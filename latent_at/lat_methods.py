import copy
import itertools
import time
import os
import math

import torch
import wandb
from peft import PeftModel
from tqdm import tqdm

from .utils import *
from .laa import *
from .laa.attacks import GDAdversary
from .lat_helpers import *

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
    accelerator=None,
):

    # Clear and initialize the adversary
    clear_hooks(model)
    if type(layer) == int:
        layer = [layer,]

    create_adversary=lambda x: GDAdversary(
        # dim=model.config.hidden_size,
        dim=4096,
        device=device,
        epsilon=epsilon,
        attack_mask=batch["prompt_mask"].to(device)
    )

    adversary_locations = [
        (f"{model_layers_module}.{layer_i}", "mlp") for layer_i in layer if type(layer_i) == int
    ]
    if "embedding" in layer:
        adversary_locations += [(model_layers_module.replace(".layers", ""), "embed_tokens")]
    if is_deepspeed_model(model):
        adversaries, wrappers = deepspeed_add_hooks(
            model,
            create_adversary=create_adversary,
            adversary_locations=adversary_locations
        )
    else:
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
            wrappers_to_disable_for_reference=wrappers,
            accelerator=accelerator,
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
        time_limit=None,
        device="cuda",
        N_checkpoints=None, # *includes* the final checkpoint
        checkpoint_dir=None,
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
        self.time_limit = time_limit
        self.device = device
        self.N_checkpoints = N_checkpoints # *includes* the final checkpoint
        self.checkpoint_dir = checkpoint_dir

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
        
    def train_adversary(self, batch, acc_step):
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
    
    def lat_training_step(self, epoch, batch, sft_batch):
        # Train Adversary
        self.disable_model_gradients()
        losses, wrappers = self.train_adversary(
            batch=batch,
            acc_step=False,
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

    def lat_training_step_with_accumulation(self, epoch, batch, sft_batch):
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

    def train_epoch(self, epoch):
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
            )
        else:
            self.lat_training_step(
                epoch=epoch,
                batch=batch,
                sft_batch=sft_batch,
            )

    def save_checkpoint(self, checkpoint_num):
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.model.save_pretrained(f"{self.checkpoint_dir}/checkpoint_{checkpoint_num}")

    def train(self, project_name, name=None, additional_wandb_kwargs=None):
        super().train(project_name, name=name, additional_wandb_kwargs=additional_wandb_kwargs)
        if self.init_callback is not None:
            self.init_callback({}, -1)


        epoch_iter = tqdm(range(self.num_steps)) if self.num_steps is not None else tqdm(itertools.count())
        start_time = time.time()

        next_checkpoint = 1

        for epoch in epoch_iter:

            try:
                self.train_epoch(epoch)
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
