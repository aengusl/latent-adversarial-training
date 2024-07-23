import copy
import itertools
import time
import os
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from peft import PeftModel
from tqdm import tqdm

from .utils import *
from .laa import *
from .laa.attacks import GDAdversary, WhitenedGDAdversary
from .lat_helpers import *
from .fit_pca import get_pcas_of_acts, get_pcas_of_acts_diff


try:
    import deepspeed
    IS_USING_DEEPSPEED = True
except ImportError:
    IS_USING_DEEPSPEED = False


def is_deepspeed_model(model):
    if not IS_USING_DEEPSPEED:
        return False
    return isinstance(model, deepspeed.DeepSpeedEngine)


def projected_gradient_descent(
    batch,
    model,
    model_layers_module,
    layer,
    epsilon,
    learning_rate,
    pgd_iterations,
    loss_coefs,
    l2_regularization=0,
    log_loss=True,
    return_loss_over_time=False,
    device="cuda",
    clip_grad=None,
    pca_kwargs=None,
    add_completions_pgd=False,
):

    # Clear and initialize the adversary
    clear_hooks(model)
    if type(layer) == int:
        layer = [layer,]
    
    if "prompt_mask" not in batch: # Only the case for RMU
        attack_mask = batch["adv_labels_mask"].to(device)
    elif add_completions_pgd:
        completions_mask = torch.any(torch.stack([batch["adv_labels_mask"], batch["def_labels_mask"]]), dim=0)
        #completions_mask = limit_ones(completions_mask, 10)
        attack_mask = torch.any(torch.stack([batch["prompt_mask"], completions_mask]), dim=0)
        attack_mask = attack_mask.to(device)
    else:
        attack_mask = batch["prompt_mask"].to(device)
    
    if pca_kwargs is not None:
        pca_proj = pca_kwargs["proj"]
        pca_unproj = pca_kwargs["unproj"]
        create_adversary=lambda x: WhitenedGDAdversary(
            dim=model.config.hidden_size,
            device=device,
            epsilon=epsilon,
            attack_mask=attack_mask,
            proj=pca_proj,
            inv_proj=pca_unproj
        )
    else:
        create_adversary=lambda x: GDAdversary(
            dim=model.config.hidden_size,
            device=device,
            epsilon=epsilon,
            attack_mask=attack_mask,
        )
    adversary_locations = [
        (f"{model_layers_module}.{layer_i}", "mlp") for layer_i in layer if type(layer_i) == int
    ]
    if "embedding" in layer:
        adversary_locations += [(model_layers_module.replace(".layers", ""), "embed_tokens")]
    adversaries, wrappers = add_hooks(
        model,
        create_adversary=create_adversary,
        adversary_locations=adversary_locations
    )
    params = []
    for adv in adversaries:
        params += list(adv.parameters())
    
    # Define optimization utils
    adv_optim = torch.optim.AdamW(params, lr=learning_rate)
    if return_loss_over_time:
        loss_over_time = []
    losses = {}
    
    # Optimize adversary to elicit attack labels
    for j in range(pgd_iterations):
        adv_optim.zero_grad()
        # Compute the adversary los
        do_adversary_step(
            model=model,
            batch=batch,
            losses_dict=losses,
            coefs=loss_coefs,
            log_loss=log_loss,
            device=device,
            wrappers_to_disable_for_reference=wrappers
        )
        # Add a L2 penalty is specified
        if l2_regularization:
            reg_loss = 0
            num_el = 0
            for adv in adversaries:
                reg_loss += torch.norm(adv.attack)
                num_el = torch.numel(adv.attack)
            (l2_regularization * reg_loss / math.sqrt(num_el)).backward()
            losses["adv_l2_norm"] = reg_loss.item() / math.sqrt(num_el)
        # Do an optimizer step
        zero_nan_grads(adv)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                adv.parameters(), clip_grad)
        adv_optim.step()
        for adv in adversaries:
            adv.clip_attack()

        if return_loss_over_time:
            loss_over_time.append(copy.deepcopy(losses))

    if return_loss_over_time:
        return loss_over_time, wrappers
    else:
        return losses, wrappers


def train_supervised_wrapper(
    dataloader,
    model,
    model_layers_module,
    layer,
    lora_wrapper_fn,
    num_steps,
    learning_rate,
    weight_decay,
    loss_coefs,
    return_loss_over_time=False,
    max_batch_per_acc=None,
    clip_grad=None,
    pca_kwargs=None
):
    # Clear and initialize the adversary
    clear_hooks(model)
    if type(layer) == int:
        layer = [layer,]
    adversaries, wrappers = add_hooks(
        model,
        create_adversary=lora_wrapper_fn,
        adversary_locations = [
            (f"{model_layers_module}.{layer_i}", "mlp") for layer_i in layer
        ]
    )
    params = []
    for adv in adversaries:
        params += list(adv.parameters())
    
    # Define optimization utils
    adv_optim = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if return_loss_over_time:
        loss_over_time = []
    losses = {}
    
    # Train steering vector
    for step in tqdm(range(num_steps)):
        batch = next(dataloader)
        adv_optim.zero_grad()
        if max_batch_per_acc is not None:
            batch_size = batch["def_tokens"].shape[0]
            acc_steps = list(range(0, batch_size, max_batch_per_acc))
            # Train advesaries for each sub-batch
            for start_idx in acc_steps:
                # Load a subset of the batch
                mini_batch = get_minibatch(batch, start_idx, max_batch_per_acc)
                do_adversary_step(
                    model=model,
                    batch=mini_batch,
                    losses_dict=losses,
                    coefs=loss_coefs,
                    log_loss=start_idx==acc_steps[-1],
                    wrappers_to_disable_for_reference=wrappers
                )   
        else:
            # Load batched data
            do_adversary_step(
                model=model,
                batch=batch,
                losses_dict=losses,
                coefs=loss_coefs,
                log_loss=True,
                wrappers_to_disable_for_reference=wrappers
            )
        zero_nan_grads(adv)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                adv.parameters(), clip_grad)
        adv_optim.step()
        if return_loss_over_time:
            loss_over_time.append(copy.deepcopy(losses))

    if return_loss_over_time:
        return loss_over_time, wrappers
    else:
        return losses, wrappers


class LATBaseClass:
    
    def __init__(
        self,
        model,
        dataloader,
        model_layers,
        init_callback=None,
        post_adv_callback=None,
        post_def_callback=None,
        model_layers_module="model.layers",
        only_train_lora=None,
    ):
        self.model = model
        self.dataloader = itertools.cycle(dataloader)
        
        if type(model_layers) == int:
            model_layers = [model_layers,]
        self.model_layers = model_layers
        
        self.init_callback = init_callback
        self.post_adv_callback = post_adv_callback
        self.post_def_callback = post_def_callback
        
        if only_train_lora is None:
            self.only_train_lora = isinstance(self.model, PeftModel)
        else:
            self.only_train_lora = only_train_lora
        self.model_layers_module = model_layers_module
    
    def disable_model_gradients(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def enable_model_gradients(self):
        n_layers = self.model.config.num_hidden_layers
        for i in range(n_layers):
            if i in self.model_layers:
                if self.only_train_lora:
                    for name, param in self.model.get_submodule(self.model_layers_module)[i].named_parameters():
                        if "lora_" in name:
                            param.requires_grad_(True)
                else:
                    self.model.get_submodule(self.model_layers_module)[i].requires_grad_(True)

    def train(self, project_name, name=None, additional_wandb_kwargs=None):
        config = self.__dict__
        if additional_wandb_kwargs is not None:
            config.update(additional_wandb_kwargs)
        wandb.init(
            project=project_name,
            config=config,
            config_exclude_keys=[
                "model",
                "dataloader",
                "sft_dataloader",
                "init_callback",
                "post_adv_callback",
                "post_def_callback"
            ],
            name=name
        )
        clear_hooks(self.model)
    

class ProjectedGradLAT(LATBaseClass):

    def __init__(
        self,
        model,
        dataloader,
        pgd_layers,
        model_layers,
        epsilon,
        init_callback=None,
        post_adv_callback=None,
        post_def_callback=None,
        outer_learning_rate=2e-5,
        inner_learning_rate=5e-2,
        num_steps=100,
        pgd_iterations_per_step=16,
        l2_regularization=0,
        model_iterations_per_step=1,
        model_layers_module="model.layers",
        only_train_lora=None,
        sft_dataloader=None,
        adv_loss_coefs={"away":0, "toward": 1},
        def_loss_coefs={"away":0, "toward": 1, "sft": 0},
        max_batch_per_acc=None,
        clip_grad=1.0,
        reinitialize_dev_optim=False,
        add_completions_pgd=False,
        time_limit=None,
        device="cuda",
        N_checkpoints=None, # *includes* the final checkpoint
        checkpoint_dir=None,
        pca_kwargs=None
    ):
        
        super().__init__(
            model=model,
            dataloader=dataloader,
            model_layers=model_layers,
            init_callback=init_callback,
            post_adv_callback=post_adv_callback,
            post_def_callback=post_def_callback,
            model_layers_module=model_layers_module,
            only_train_lora=only_train_lora,
        )

        self.pgd_layers = pgd_layers
        self.epsilon = epsilon
        self.l2_regularization = l2_regularization 
        self.outer_learning_rate = outer_learning_rate
        self.inner_learning_rate = inner_learning_rate
        self.num_steps = num_steps
        self.pgd_iterations_per_step = pgd_iterations_per_step
        self.model_iterations_per_step = model_iterations_per_step
        self.max_batch_per_acc = max_batch_per_acc
        self.clip_grad = clip_grad
        self.reinitialize_dev_optim = reinitialize_dev_optim
        self.add_completions_pgd = add_completions_pgd
        self.time_limit = time_limit
        self.device = device
        self.N_checkpoints = N_checkpoints # *includes* the final checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.pca_kwargs = pca_kwargs

        if sft_dataloader is not None:
            assert dataloader.batch_size == sft_dataloader.batch_size
            self.sft_dataloader = itertools.cycle(sft_dataloader)
        else:
            assert def_loss_coefs["sft"] == 0
            self.sft_dataloader = None  

        self.adv_loss_coefs = normalize_dict(adv_loss_coefs)
        self.def_loss_coefs = normalize_dict(def_loss_coefs)
        
        self.def_optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.outer_learning_rate
        )
        
        self.attack_type = "pgd"
        
    def train_adversary(self, batch, acc_step, pca_kwargs=None):
        return projected_gradient_descent(
            batch=batch,
            model=self.model,
            model_layers_module=self.model_layers_module,
            layer=self.pgd_layers,
            epsilon=self.epsilon,
            l2_regularization=self.l2_regularization,
            learning_rate=self.inner_learning_rate,
            pgd_iterations=self.pgd_iterations_per_step,
            loss_coefs=self.adv_loss_coefs,
            log_loss=not acc_step,
            device=self.device,
            pca_kwargs=pca_kwargs,
            add_completions_pgd=self.add_completions_pgd
        )

    def train_defense(self, batch, sft_batch, wrappers, zero_grad, grad_step):
        # Initialize optimizer and loss
        losses = {}
        if zero_grad:
            self.def_optim.zero_grad()
        # Compute the defense        
        do_defense_step(
            model=self.model,
            batch=batch,
            losses_dict=losses,
            wrappers=wrappers,
            sft_batch=sft_batch,
            coefs=self.def_loss_coefs,
            log_loss=grad_step,
            device=self.device
        )
        zero_nan_grads(self.model)
        # Do gradient step
        if grad_step:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad)
            self.def_optim.step()
        return losses
    
    def lat_training_step(self, epoch, batch, sft_batch, pca_kwargs=None):
        # Train Adversary
        self.disable_model_gradients()
        losses, wrappers = self.train_adversary(
            batch=batch,
            acc_step=False,
            pca_kwargs=pca_kwargs
        )
        if self.post_adv_callback is not None:
            self.post_adv_callback(losses, epoch=epoch)
        # Train model against adversary
        self.enable_model_gradients()
        for _ in range(self.model_iterations_per_step):
            def_losses = self.train_defense(
                batch=batch,
                sft_batch=sft_batch,
                wrappers=wrappers,
                zero_grad=True,
                grad_step=True,
            )
        # Log results
        losses.update(def_losses)
        clear_hooks(self.model)
        if self.post_def_callback is not None:
            self.post_def_callback(losses, epoch)

    def lat_training_step_with_accumulation(self, epoch, batch, sft_batch, pca_kwargs=None):
        # Train gradient accumulation version
        batch_size = batch["def_tokens"].shape[0]
        acc_steps = list(range(0, batch_size, self.max_batch_per_acc))
        acc_wrappers = []
        # Train advesaries for each sub-batch
        for start_idx in acc_steps:
            # Load a subset of the batch
            mini_batch = get_minibatch(batch, start_idx, self.max_batch_per_acc)
            # print(f"MINI BATCH: {mini_batch}")

            # Train Adversary
            self.disable_model_gradients()
            losses, wrappers = self.train_adversary(
                batch=mini_batch,
                acc_step=start_idx!=acc_steps[-1],
                pca_kwargs=pca_kwargs
            )
            acc_wrappers.append(wrappers)
            for wrapper in wrappers:
                wrapper.enabled = False
        if self.post_adv_callback is not None:
            self.post_adv_callback(losses, epoch=epoch)
        # Train defense for each sub-batch
        for _ in range(self.model_iterations_per_step):
            for i, start_idx in enumerate(acc_steps):
                # Load in things associated with subbatch
                mini_batch = get_minibatch(batch, start_idx, self.max_batch_per_acc)
                sft_mini_batch = get_minibatch(sft_batch, start_idx, self.max_batch_per_acc)
                wrappers = acc_wrappers[i]
                for wrapper in wrappers:
                    wrapper.enabled = True                    
                # Train model against adversary
                self.enable_model_gradients()
                def_losses = self.train_defense(
                    batch=mini_batch,
                    sft_batch=sft_mini_batch,
                    wrappers=wrappers,
                    zero_grad=start_idx==acc_steps[0],
                    grad_step=start_idx==acc_steps[-1],
                )
                for wrapper in wrappers:
                    wrapper.enabled = False
        # Log results
        losses.update(def_losses)
        if self.post_def_callback is not None and start_idx == acc_steps[-1]:
            self.post_def_callback(losses, epoch)
        clear_hooks(self.model)

    def train_epoch(self, epoch, new_pca_projs=None):
        # Load batched data
        batch = next(self.dataloader)
        if self.sft_dataloader is not None:
            sft_batch = next(self.sft_dataloader)
        else:
            sft_batch = None
        # Reinitialize optimizer every LAT step
        if self.reinitialize_dev_optim:
            self.def_optim = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.outer_learning_rate
            )
        # Start training loop
        if self.max_batch_per_acc is not None:
            self.lat_training_step_with_accumulation(
                epoch=epoch,
                batch=batch,
                sft_batch=sft_batch,
                pca_kwargs=new_pca_projs
            )
        else:
            self.lat_training_step(
                epoch=epoch,
                batch=batch,
                sft_batch=sft_batch,
                pca_kwargs=new_pca_projs
            )

    def save_checkpoint(self, checkpoint_num):
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.model.save_pretrained(f"{self.checkpoint_dir}/checkpoint_{checkpoint_num}")

    def train(self, project_name, name=None, additional_wandb_kwargs=None):
        super().train(project_name, name=name, additional_wandb_kwargs=additional_wandb_kwargs)
        if self.init_callback is not None:
            self.init_callback({}, -1)
        
        use_pca = self.pca_kwargs is not None
        if use_pca:
            refresh_pca_every = self.pca_kwargs.get("refresh_every", None)
            pca_proj = self.pca_kwargs.get("proj", None)
            pca_unproj = self.pca_kwargs.get("unproj", None)

        epoch_iter = tqdm(range(self.num_steps)) if self.num_steps is not None else tqdm(itertools.count())
        start_time = time.time()

        next_checkpoint = 1

        for epoch in epoch_iter:
            if use_pca and (refresh_pca_every is not None) and (epoch % refresh_pca_every == 0):
                print("Refreshing PCA")
                if self.pca_kwargs.get("use_act_diff", False):
                    print("Using act diff")
                    pcas = get_pcas_of_acts_diff(
                        model=self.model,
                        tokenizer=self.pca_kwargs["tokenizer"],
                        dataset=self.pca_kwargs["dataset"],
                        device=self.pca_kwargs.get("device", "cuda"),
                        gen_batch_size=self.pca_kwargs.get("gen_batch_size", 32),
                        pca_batch_size=self.pca_kwargs.get("pca_batch_size", 128),
                        num_batches=self.pca_kwargs.get("num_batches", 1024),
                        cache_locations=[(f"{self.model_layers_module}.{self.pgd_layers}", "mlp")],
                        dims=self.model.config.hidden_size,
                        verbose=False,
                        dataset_text_cols=self.pca_kwargs.get("dataset_text_cols", ["text_1", "text_2"]),  # Adjusted for dual text columns
                        dataset_tokens_cols=self.pca_kwargs.get("dataset_tokens_cols", [None, None]),  # Adjusted for potential dual token columns
                        max_ctx_len=self.pca_kwargs.get("max_ctx_len", 2048),
                        index_last_poss=self.pca_kwargs.get("index_last_poss", [None, None]),  # Adjusted for potential dual index_last_pos
                        indices_cols=self.pca_kwargs.get("indices_cols", [None, None]),  # Adjusted for potential dual indices columns
                    )
                else:
                    print("Not using act diff")
                    pcas = get_pcas_of_acts(
                        model=self.model,
                        tokenizer=self.pca_kwargs["tokenizer"],
                        dataset=self.pca_kwargs["dataset"],
                        device=self.pca_kwargs.get("device", "cuda"),
                        gen_batch_size=self.pca_kwargs.get("gen_batch_size", 32),
                        pca_batch_size=self.pca_kwargs.get("pca_batch_size", 128),
                        num_batches=self.pca_kwargs.get("num_batches", 100),
                        cache_locations = [(f"{self.model_layers_module}.{self.pgd_layers}", "mlp"),],
                        dims=self.model.config.hidden_size,
                        verbose=False,
                        max_ctx_len=self.pca_kwargs.get("max_ctx_len", 2048),
                        index_last_pos=self.pca_kwargs.get("index_last_pos", None),
                        dataset_text_col=self.pca_kwargs.get("dataset_text_col", "text"),
                        dataset_tokens_col=self.pca_kwargs.get("dataset_tokens_col", None),
                        indices_col=self.pca_kwargs.get("indices_col", None),
                    )
                pca_proj, pca_unproj = pcas[(f'{self.model_layers_module}.{self.pgd_layers}', 'mlp')].get_projections()

            if use_pca:
                new_pca_projs = {"proj": pca_proj, "unproj": pca_unproj}
            else:
                new_pca_projs = None

            try:
                self.train_epoch(epoch, new_pca_projs=new_pca_projs)
            except Exception as e:
                print(f"Error at epoch {epoch} of {name}: {e}")
                os.makedirs("logs", exist_ok=True)
                with open(f"logs/{name}_errors.txt", "a") as f:
                    f.write(f"Error at epoch {epoch} of {name}: {e}\n")
            elapsed_time = time.time() - start_time
            # Checkpointing
            if self.N_checkpoints:
                step_checkpoint = self.num_steps is not None and (epoch+1)/self.num_steps >= next_checkpoint/self.N_checkpoints
                time_checkpoint = self.time_limit is not None and elapsed_time/self.time_limit >= next_checkpoint/self.N_checkpoints
                if step_checkpoint or time_checkpoint:
                    print(f"Saving checkpoint at epoch {epoch}")
                    self.save_checkpoint(next_checkpoint)
                    next_checkpoint += 1
            # Time limit
            if self.time_limit is not None and elapsed_time > self.time_limit:
                print(f"Reached {elapsed_time} seconds at epoch {epoch}")
                break
        wandb.finish()


def run_rmu(
        model,
        make_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        alpha=1200.0,
        layer_ids=[6,7,8], # layers to train
        layer_id=8, # layers to do RMU in
        param_ids=[6],
        lr=5.0e-5,
        module_str="{model_name}.model.layers[{layer_id}]",
        steering_coef=6.5,
        model_iterations_per_step=4,
        max_num_batches=200,
        use_pgd=True,
        pgd_layers=7,  # layers to attack
        epsilon=2,
        inner_learning_rate=5.0e-2,
        pgd_iterations_per_step=16,
        adv_loss_coefs={'toward': 1, 'away': 1},
        num_epochs=1,
):
    def disable_model_gradients() -> None:
        for param in model.parameters():
            param.requires_grad_(False)
    def enable_model_gradients() -> None:
        for i in range(model.config.num_hidden_layers):
            if i in range(model.config.num_hidden_layers):
                model.get_submodule("model.layers")[i].requires_grad_(True)

    def train_attack(
            model,
            batch,
            do_grad_step,
            epsilon,
            inner_learning_rate,
            pgd_iterations_per_step,
            adv_loss_coefs,
    ):
        return projected_gradient_descent(
            batch=batch,
            model=model,
            model_layers_module="model.layers",
            layer=pgd_layers,
            epsilon=epsilon,
            learning_rate=inner_learning_rate,
            pgd_iterations=pgd_iterations_per_step,
            loss_coefs=adv_loss_coefs,
            log_loss=do_grad_step,
            device=model.device,
        )

    frozen_model = make_model()
    model = model.train()
    params = get_params(model, layer_ids, param_ids)
    optimizer = torch.optim.AdamW(params, lr=lr)
    frozen_module = eval(
        module_str.format(model_name="frozen_model", layer_id=layer_id)
    )
    updated_module = eval(
        module_str.format(model_name="model", layer_id=layer_id)
    )

    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(1, 1, model.config.hidden_size, dtype=model.dtype,
                                   device=model.device)
        control_vec = random_vector / torch.norm(random_vector) * steering_coef
        control_vectors_list.append(control_vec)

    num_batches = max_num_batches
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for epoch in range(num_epochs):
        for idx in tqdm(range(num_batches)):

            clear_hooks(model)

            control_vec = control_vectors_list[idx]
            unlearn_batch = forget_data_list[idx]
            retain_batch = retain_data_list[idx]

            max_length = 512 if idx == 0 else 768
            unlearn_inputs = tokenizer(
                unlearn_batch, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length
            ).to(model.device)
            retain_inputs = tokenizer(
                retain_batch, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length
            ).to(model.device)

            if use_pgd:
                adv_labels_mask = torch.zeros_like(unlearn_inputs["input_ids"], dtype=bool)
                def_labels_mask = torch.zeros_like(retain_inputs["input_ids"], dtype=bool)
                for b, example in enumerate(retain_batch):
                    len_example = len(tokenizer(example)["input_ids"])
                    def_labels_mask[b, :len_example] = True
                for b, example in enumerate(unlearn_batch):
                    len_example = len(tokenizer(example)["input_ids"])
                    adv_labels_mask[b, :len_example] = True

                # prompt_mask = torch.zeros(len(unlearn_batch), pad_length + 1, dtype=torch.bool)
                pgd_batch = {"def_tokens": retain_inputs["input_ids"].to(model.device),
                             "adv_tokens": unlearn_inputs["input_ids"].to(model.device),
                             "adv_labels_mask": adv_labels_mask.to(model.device),
                             "def_labels_mask": def_labels_mask.to(model.device)}
                disable_model_gradients()
                losses, hooks = train_attack(
                    model,
                    batch=pgd_batch,
                    do_grad_step=True,
                    epsilon=epsilon,
                    inner_learning_rate=inner_learning_rate,
                    pgd_iterations_per_step=pgd_iterations_per_step,
                    adv_loss_coefs=adv_loss_coefs,
                )
                enable_model_gradients()

                for hook in hooks:
                    hook.enabled = False
                for _ in range(model_iterations_per_step):
                    for hook in hooks:
                        hook.enabled = True
                    # Unlearning loss
                    updated_forget_activations = forward_with_cache(
                        model, unlearn_inputs, module=updated_module, no_grad=False
                    ).to(model.device)
                    unlearn_loss = torch.nn.functional.mse_loss(
                        updated_forget_activations, control_vec
                    )

                    # Retain loss
                    for hook in hooks:
                        hook.enabled = False
                    updated_retain_activations = forward_with_cache(
                        model, retain_inputs, module=updated_module, no_grad=False
                    ).to(model.device)
                    frozen_retain_activations = forward_with_cache(
                        frozen_model, retain_inputs, module=frozen_module, no_grad=True
                    ).to(model.device)
                    retain_loss = torch.nn.functional.mse_loss(
                        updated_retain_activations, frozen_retain_activations
                    )
                    retain_loss *= alpha

                    # Update model
                    loss = unlearn_loss + retain_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            else:
                # Unlearning loss
                updated_forget_activations = forward_with_cache(
                    model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(model.device)
                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec
                )

                # Retain loss
                updated_retain_activations = forward_with_cache(
                    model, retain_inputs, module=updated_module, no_grad=False
                ).to(model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(model.device)
                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                retain_loss *= alpha

                # Update model
                loss = unlearn_loss + retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    tokenizer.truncation_side = truncation_side

    return model