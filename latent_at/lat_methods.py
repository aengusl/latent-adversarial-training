import copy
import itertools
import time
import os
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from peft import AutoPeftModelForCausalLM, PeftModel
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from tasks.inference_utils import get_batched_generations
from tasks.general_capabilities.multiple_choice_tasks import MMLUTask
from tasks.wmdp.WMDP_MCTask import WMDP_MCTask


import torch.distributed as dist

from .utils import *
from .laa import *
from .laa.attacks import GDAdversary, WhitenedGDAdversary
from .lat_helpers import *
from .fit_pca import get_pcas_of_acts, get_pcas_of_acts_diff

import deepspeed

def is_deepspeed_model(model):
    return isinstance(model, deepspeed.DeepSpeedEngine)

def projected_gradient_descent(
    batch: dict[str, torch.Tensor],
    model: nn.Module,
    model_layers_module: List[str],
    layer: Union[int, List[int]],
    epsilon: float,
    learning_rate: float,
    pgd_iterations: int,
    loss_coefs: dict[str, float],
    l2_regularization: float = 0,
    device: str = "cuda",
    log_loss: Optional[bool] = True,
    return_loss_over_time: Optional[bool] = False,
    clip_grad: Optional[bool] = None,
    pca_kwargs: Optional[dict[str, Any]] = None,
    accelerator: Any = None,
) -> tuple[Union[list[dict], dict], list[nn.Module]]:
    """
    Add hooks and return the adversaries and hooks.
    Create adversary optimizer.
    Run the PGD for as many iterations as specified by pgd_iterations.
    Zero grads, backward loss, step, clip grads (if specified True).

    Args:
        batch: Standardised format of dictionary of strings followed by data. 
            Possible keys: ["tokens", "def_tokens", "adv_tokens", "adv_labels", "def_labels", "prompt_mask"]
        loss_coefs: Set to zero for away or toward loss to remove from loss function.
            Possible keys: ["away", "toward"]
    
    Returns:
        losses or losses_over_time: Dictionary of losses.
        wrappers: List of hook instances. These subclass nn.Module.
    """
    # Clear and initialize the adversary
    clear_hooks(model)
    if type(layer) == int:
        layer = [layer,]
    
    if pca_kwargs is not None:
        pca_proj = pca_kwargs["proj"]
        pca_unproj = pca_kwargs["unproj"]
        create_adversary=lambda x: WhitenedGDAdversary(
            # dim=model.config.hidden_size,
            dim=4096,
            device=device,
            epsilon=epsilon,
            attack_mask=batch["prompt_mask"] if "prompt_mask" in batch else batch["adv_labels_mask"].to(device),
            proj=pca_proj,
            inv_proj=pca_unproj
        )
    else:
        create_adversary=lambda x: GDAdversary(
            # dim=model.config.hidden_size,
            dim=4096,
            device=device,
            epsilon=epsilon,
            attack_mask=batch["prompt_mask"] if "prompt_mask" in batch else batch["adv_labels_mask"].to(device)
        )

    if is_deepspeed_model(model):
        adversaries, wrappers = deepspeed_add_hooks(
            model,
            create_adversary=create_adversary,
            adversary_locations = [
                (f"{model_layers_module}.{layer_i}", "mlp") for layer_i in layer
            ]
        )
    else:
        adversaries, wrappers = add_hooks(
            model,
            create_adversary=create_adversary,
            adversary_locations = [
                (f"{model_layers_module}.{layer_i}", "mlp") for layer_i in layer
            ]
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
            pca_kwargs=pca_kwargs
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


class WMDPProjectedGradLAT(LATBaseClass):

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        pgd_layers: List[int],
        model_layers: List[int],
        epsilon: float,
        config: OmegaConf,
        init_callback: Optional[Callable] = None,
        post_adv_callback: Callable = None,
        post_def_callback: Callable = None,
        outer_learning_rate: float = 1e-4,
        inner_learning_rate: float = 5e-2,
        num_steps: int = 50,
        l2_regularization: float = 0,
        pgd_iterations_per_step: int = 16,
        model_iterations_per_step: int = 1,
        model_layers_module: List[str] = "model.layers",
        only_train_lora: bool = None,
        sft_dataloader: torch.utils.data.DataLoader = None,
        adv_loss_coefs: dict[str, float] = {"away": 0, "toward": 1},
        def_loss_coefs: dict[str, float] = {"away": 0, "toward": 1, "sft": 0},
        max_batch_per_acc: int = None,
        clip_grad: float= 1.0,
        reinitialize_dev_optim: bool = True,
        time_limit: int = None,
        device: str = "cuda",
        checkpoint_dir=None,
        pca_kwargs = None,
    ):
        """
        Args used for pgd:
            pgd_layers: e.g. range(earliest_layer, llama.config.num_hidden_layers).
                Layers to train adversary for. Passed into projected_gradient_descent.
            epsilon: Attack clip maxmimum distance. Passed into projected_gradient_descent.
            init_callback: For logging.
            inner_learning_rate: PGD learning rate. Passed into projected_gradient_descent.
            pgd_iterations_per_epoch: Passed into projected_gradient_descent.
            model_layers_module: used in projected_gradient_descent:         
                adversary_locations = [(f"{model_layers_module}.{layer_i}", "mlp") for layer_i in layer]
            only_train_lora: Passed into projected_gradient_descent.
            adv_loss_coefs: Set to zero for away or toward loss to remove from loss function.
                Passed into projected_gradient_descent.

        Args used for defence:
            outer_learning_rate: Defence learning rate.
            model_iterations_per_step: Should be mostly 1.
            def_loss_coefs: If supervised fine-tuning loss term not used, "sft" should be set to 0.   
                Passed into do_def_step.
                Set to zero for away or toward loss to remove from loss function.
            clip_grad: Gradient clipping value in defence step.
        
        Other args:
            post_adv_callback: For logging adversary loss as fn of epoch.
            post_def_callback: "" for model.
            num_steps: Number training 'epochs', used to make iterable of training epoch value.
                Epochs are passed into lat_training_step and lat_training_step_with_accumulate.
                Set to None to train until wallclock time runs out.
            reinitialize_dev_optim: Whether to reinitialize optimizer for model every LAT step. Default True.
            sft_dataloader: Batches sft_batch from this passed into the two training functions, which then call do_defense_step.
            max_batch_per_acc: Minibatch size in gradient accumulation training.
            time_limit: Units seconds. Used to terminate training when wallclock time runs out, when num_steps is not specified.
        """
        
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
        self.l2_regularization = l2_regularization 
        self.epsilon = epsilon
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
        self.checkpoint_dir = checkpoint_dir
        self.pca_kwargs = pca_kwargs
        self.config = config

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

        # Mapping from YAML config model names to HuggingFace paths
        self.model_dict = model_dict = {
            "LLAMA_7B": "meta-llama/Llama-2-7b-chat-hf",
            "LLAMA_34B": "meta-llama/Llama-2-34b-chat-hf",
            "ZEPHYR_7B": "HuggingFaceH4/zephyr-7b-beta",
            "ZEPHYR_34B": "HuggingFaceH4/zephyr-34b-beta",
        }
        
    def train_adversary(
        self, 
        batch: dict[str, torch.Tensor], 
        acc_step: bool, 
        pca_kwargs: dict = None,
    ) -> None:
        return projected_gradient_descent(
            batch=batch,
            model=self.model,
            model_layers_module=self.model_layers_module,
            layer=self.pgd_layers,
            l2_regularization=self.l2_regularization,
            epsilon=self.epsilon,
            learning_rate=self.inner_learning_rate,
            pgd_iterations=self.pgd_iterations_per_step,
            loss_coefs=self.adv_loss_coefs,
            log_loss=not acc_step,
            device=self.device,
            pca_kwargs=pca_kwargs,
        )

    def train_defense(
        self, 
        batch: dict[str, torch.Tensor], 
        wrappers: list[CustomHook], 
        sft_batch: dict[str, torch.Tensor],
        zero_grad: bool,
        grad_step: bool,
    ) -> dict[str, float]:
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.def_optim.step()

        return losses
    
    def lat_training_step(self, epoch, batch, sft_batch, pca_kwargs=None):
        # Train adversary
        self.disable_model_gradients()
        losses, wrappers = self.train_adversary(
            batch=batch,
            acc_step=False,
            pca_kwargs=pca_kwargs,
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
    
    def evaluate_during_training(self) -> dict:
        """Quick evaluation function during training. All our official results use lm-eval-harness, but it is slow. This offers a ball-park evaluation which can be used during the training process.
        
        Create right padded tokenizer specific to this evaluation method.

        Edit MLMLUTask parameters (bottom of fn) to change example count.
        Edit WMDP_MCTask batch size and num_iters as needed.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dict[self.config.model_name], 
            trust_remote_code=True, 
            use_fast=False,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        tokenizer.mask_token_id = tokenizer.eos_token_id
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.cls_token_id = tokenizer.eos_token_id

        wmdp_scores = {}
        num_iters = 30
        print(f"evaluating on sample size {16*num_iters}")
        subtasks = ["wmdp-bio", "wmdp-chem", "wmdp-cyber"]
        for subtask in subtasks:
            wmdp = WMDP_MCTask(batch_size=16, tokenizer=tokenizer, subset=subtask)
            unlearned_accuracy = 0
            for _ in tqdm(range(num_iters)):
                unlearned_accuracy += wmdp.get_test_accuracy(self.model)
            wmdp_scores[subtask] = unlearned_accuracy / num_iters
        avg_wmdp_score = sum([abs(v) for v in wmdp_scores.values()])/len(wmdp_scores)
        print(wmdp_scores)
        
        left_tokenizer = AutoTokenizer.from_pretrained(
            self.model_dict[self.config.model_name], 
            trust_remote_code=True, 
            use_fast=False,
        )
        left_tokenizer.pad_token_id = tokenizer.eos_token_id
        left_tokenizer.padding_side = "left"

        mmlu = MMLUTask()
        mmlu = mmlu.get_accuracy(self.model, tokenizer=left_tokenizer, temperature=0, batch_size=25, n_batches=40, verbose=False)
        combined_evals = wmdp_scores | {"MMLU": mmlu} | {"pareto_score": mmlu/ avg_wmdp_score}

        return combined_evals

    def lat_training_step_with_accumulation(
        self, 
        batch: dict[str, torch.Tensor],
        sft_batch: Optional[dict[str, torch.Tensor]] = None,
        pca_kwargs: dict = None,
    ) -> None:
        # Train gradient accumulation version
        batch_size = batch["def_tokens"].shape[0]
        acc_steps = list(range(0, batch_size, self.max_batch_per_acc))
        acc_wrappers = []

        # Train advesaries for each sub-batch with minibatch gradient accumulation
        for start_idx in acc_steps:
            # Load a subset of the batch
            mini_batch = get_minibatch(batch, start_idx, self.max_batch_per_acc)

            # Train adversary
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
            self.post_adv_callback(losses, epoch=self.idx)
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
            self.post_def_callback(losses, self.idx)
        clear_hooks(self.model)

    def train_epoch(self, epoch, new_pca_projs=None):
        # Load batched data
        batch = next(self.dataloader)
        sft_batch = next(self.sft_dataloader) if self.sft_dataloader else None

        # Reinitialize optimizer every LAT step
        if self.reinitialize_dev_optim:
            self.def_optim = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.schedule[self.idx] if self.config.use_cosine_schedule else self.config.def_config.outer_learning_rate,
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
        super().train(project_name, additional_wandb_kwargs=additional_wandb_kwargs)
        if self.init_callback is not None:
            self.init_callback({}, -1)
        
        use_pca = self.pca_kwargs is not None
        if use_pca:
            refresh_pca_every = self.pca_kwargs.get("refresh_every", None)
            pca_proj = self.pca_kwargs.get("proj", None)
            pca_unproj = self.pca_kwargs.get("unproj", None)

        epoch_iter = tqdm(range(self.num_steps)) if self.num_steps is not None else tqdm(itertools.count())
        start_time = time.time()
        self.idx = 0

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
                # WMDP selective saving based on WMDP versus MMLU performance
                if self.idx % self.config.eval_steps == 0:
                    with torch.inference_mode():
                        eval_accs = self.evaluate_during_training(self.model)
                    wandb.log(eval_accs, step=epoch)

                    subset = "bio" if self.config.use_just_bio else "cyber"
                    # Requirement for saving model (will be logged): WMDP evaluation decreases, and MMLU above save threshold set in config
                    if abs(eval_accs[f"wmdp-{subset}"]-0.25) < abs(self.best_accs["wmdp"]-0.25) and eval_accs["MMLU"] >= self.config.save_threshold:
                        self._save_model()
                        self.best_accs = {"wmdp": eval_accs[f"wmdp-{subset}"], "mmlu": eval_accs["MMLU"]}
                        os.makedirs("logs", exist_ok=True)
                        with open(f"logs/{name}_metrics.txt", "a") as f:
                            f.write(f"Saved at {epoch} of {name}: with wmdp {subset} {eval_accs[f'wmdp-{subset}']} and mmlu {eval_accs['MMLU']}\n")
                self.idx += 1
            except Exception as e:
                print(f"Error at epoch {epoch} of {name}: {e}")
                os.makedirs("logs", exist_ok=True)
                with open(f"logs/{name}_errors.txt", "a") as f:
                    f.write(f"Error at epoch {epoch} of {name}: {e}\n")
            elapsed_time = time.time() - start_time
            if self.time_limit is not None and elapsed_time > self.time_limit:
                print(f"Reached {elapsed_time} seconds at epoch {epoch}")
                break
        
        for attr in list(self.__dict__.keys()):
            del attr
        torch.cuda.empty_cache()

        log_artifact = wandb.Artifact('run_logs', type='log')
        log_artifact.add_file(f"logs/{self.run_name}_metrics.txt")
        wandb.log_artifact(log_artifact)
        wandb.finish()