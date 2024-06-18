# LAT implemented as a subclass of Huggingface's `Trainer` class

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from datasets import load_dataset, Dataset
from typing import List, Union, Dict, Any, Tuple, Optional, Callable
import torch
from dataclasses import dataclass, field
from latent_at.laa.attacks import GDAdversary
from accelerate import Accelerator
from latent_at.utils import log_1_minus_p_loss
import tqdm
import wandb
import itertools

def zero_nan_grads(parameters):
    for param in parameters:
        if param.grad is not None:
            param.grad[torch.isnan(param.grad)] = 0.

@dataclass
class PGDConfig:
    inner_learning_rate: float = 5e-2

    # Llama hparams
    outer_learning_rate: float = 2e-5
    outer_weight_decay: float = 0.1

    inner_steps_per_batch: int = 16
    outer_steps_per_batch: int = 5

    pgd_epsilon: float = 1.0
    pgd_criterion: str = "cross_entropy"

    towards_criterion: str = "cross_entropy"
    away_criterion: str = "log_1_minus_p"
    sft_criterion: str = "cross_entropy"

    sft_ratio: float = 1.0
    away_ratio: float = 1.0
    towards_ratio: float = 1.0

    per_device_max_batch: int = 8

    locations: List[str] = field(default_factory=lambda : [])
    dimensions: List[str] = field(default_factory=lambda : [])

    clip_grad: float = 1.0
    reinit_def_optim: bool = False # unimplemented
    num_steps: int = 50

@dataclass
class PGDScheduler:
    epsilon_schedule: Optional[Callable[[int], float]] = None
    lr_schedule: Optional[Callable[[int], float]] = None

    towards_ratio_schedule: Optional[Callable[[int], float]] = None
    away_ratio_schedule: Optional[Callable[[int], float]] = None
    sft_ratio_schedule: Optional[Callable[[int], float]] = None

@dataclass
class DatasetConfig:
    prompt_mask: str = "prompt_mask"
    adv_tokens: str = "adv_tokens"
    def_tokens: str = "def_tokens"
    adv_labels: str = "adv_labels"
    def_labels: str = "def_labels"
    sft_tokens: str = "tokens"
    sft_labels: str = "labels"
    is_nonempty_completion: Optional[str] = None

class PGDWrapper(torch.nn.Module):
    """
    Wraps a torch.nn.Module with a set of PGD attacks for each subbatch.
    Fullfils the role of the multiple `CustomHook`s in the old code.
    Works with FSDP via Accelerate. Probably will not work with others.
    Definitely will not work with TPUs, XLA, torch.compile, etc.
    """

    def __init__(
        self,
        pgd_config: PGDConfig,
        module: torch.nn.Module,
        tensor_dim: int,
        device=None,
    ):
        super().__init__()
        self.config = pgd_config
        self.module = module
        self.tensor_dim = tensor_dim
        self.selected_attack = None
        self.device = device
        self.epsilon_scale = 1.0
        
        self.pgd_attacks = []
    
    def init_new_attack_from_mask(
        self,
        prompt_mask: torch.Tensor,
    ):
        self.pgd_attacks.append(GDAdversary(
            dim=self.tensor_dim,
            epsilon=self.config.pgd_epsilon * self.epsilon_scale,
            attack_mask=prompt_mask,
            device=self.device,
        ))

    def perturb(
        self,
        x
    ):
        if isinstance(x, tuple):
            return (self.perturb(x[0]),) + x[1:]
        
        assert self.selected_attack is not None, "FATAL: No attack specified! This should not happen."
        
        return self.pgd_attacks[self.selected_attack](x)
    
    def forward(
        self,
        inputs
    ):
        outputs = self.module(inputs)

        if self.selected_attack is not None:
            return self.perturb(outputs)
        else:
            return outputs

    def clip_selected_attack(self):
        assert self.selected_attack is not None, "FATAL: No attack specified! This should not happen."

        self.pgd_attacks[self.selected_attack].clip_attack()

    def freeze_selected_attack(self):
        assert self.selected_attack is not None, "FATAL: No attack specified! This should not happen."

        for param in self.pgd_attacks[self.selected_attack].parameters():
            param.requires_grad = False
            param.grad = None

    def clear_attacks(self):
        self.pgd_attacks = []

    def selected_pgd_parameters(self):
        assert self.selected_attack is not None, "FATAL: No attack specified! This should not happen."

        return self.pgd_attacks[self.selected_attack].parameters()

class PGDModel(torch.nn.Module):
    """
    Wraps a PreTrainedModel with a set of PGD attacks for each subbatch.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        pgd_config: PGDConfig,
    ):
        super().__init__()
        self.model = model
        self.pgd_config = pgd_config

        self.wrappers = []
        self.wrapper_locations = []

        # iterate over all modules in the model
        for name, module in model.named_modules():
            for child_name, child_module in module.named_children():
                matching_locations = [
                    loc for loc in self.pgd_config.locations if f"{name}.{child_name}".endswith(loc)
                ]

                if len(matching_locations) > 0:
                    matched_loc =  matching_locations[0]

                    print("Found match:", f"{name}.{child_name}", matched_loc)

                    tensor_dim = self.pgd_config.dimensions[self.pgd_config.locations.index(matched_loc)]

                    wrapper = PGDWrapper(
                        pgd_config=self.pgd_config,
                        module=child_module,
                        tensor_dim=tensor_dim
                    )
                    self.wrappers.append(wrapper)
                    self.wrapper_locations.append((name, child_name))
                    setattr(module, child_name, wrapper)

    def unwrap_model(self):
        for wrapper, (name, child_name) in zip(self.wrappers, self.wrapper_locations):
            setattr(getattr(self.model, name), child_name, wrapper.module)
        
        self.wrappers = []
        self.wrapper_locations = []

        return self.model
    
    def selected_pgd_parameters(self):
        return [param for wrapper in self.wrappers for param in wrapper.selected_pgd_parameters()]

    def select_attack(self, attack_idx: Optional[int] = None):
        for wrapper in self.wrappers:
            wrapper.selected_attack = attack_idx
    
    def set_epsilon_scale(self, epsilon_scale: float):
        for wrapper in self.wrappers:
            wrapper.epsilon_scale = epsilon_scale

    def init_new_attacks_from_mask(
        self,
        prompt_mask: torch.Tensor,
    ):
        for wrapper in self.wrappers:
            wrapper.init_new_attack_from_mask(prompt_mask)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)
    
    def clip_selected_attacks(self):
        for wrapper in self.wrappers:
            wrapper.clip_selected_attack()

    def freeze_selected_attacks(self):
        for wrapper in self.wrappers:
            wrapper.freeze_selected_attack()

    def clear_attacks(self):
        self.select_attack(None)

        for wrapper in self.wrappers:
            wrapper.clear_attacks()

    @property
    def current_num_subbatches(self):
        assert len(self.wrappers) > 0
        return len(self.wrappers[0].pgd_attacks)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def __setattr__(self, name, value):
        try:
            super().__setattr__(name, value)
        except AttributeError:
            setattr(self.model, name, value)

def accumulate_losses(
    dicts: List[Dict[str, float]],
    mean: bool = True
):
    losses = {
        k: 0
        for k in set().union(*dicts)
    }

    for d in dicts:
        for k, v in d.items():
            losses[k] += v / len(dicts) if mean else v
    
    return losses

class PGDTrainer:
    """
    A trainer for PGD-LAT suitable for use with DDP or FSDP.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: PGDConfig,
        dataset_config: DatasetConfig,
        lat_dataloader,
        sft_dataloader=None,
        accelerator_args: Dict[str, Any] = {},
        init_tracker_args: Dict[str, Any] = {},
        pgd_schedule: Optional[PGDScheduler] = None,
        train_modules: List[str] = ["model"],
        train_lora: bool = False,
    ):
        self.config = config
        self.dataset_config = dataset_config

        pgd_model = PGDModel(model, config)

        self.trainable_params = []

        if train_lora:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    self.trainable_params.append(param)
        else:
            for name, module in model.named_modules():
                if any([name.endswith(m) for m in train_modules]):
                    self.trainable_params.extend(module.parameters())

        for param in model.parameters():
            param.requires_grad = False
        
        for param in self.trainable_params:
            param.requires_grad = True

        model_optim = torch.optim.AdamW(
            self.trainable_params,
            lr=config.outer_learning_rate,
            weight_decay=config.outer_weight_decay
        )

        self.pgd_schedule = pgd_schedule
        self.loss_schedules = {
            "towards": 1.0,
            "away": 1.0,
            "sft": 1.0,
        }

        self.accelerator = Accelerator(**accelerator_args)
        self.accelerator.init_trackers(**init_tracker_args)

        self.pgd_model, self.model_optim, self.lat_dataloader, self.sft_dataloader = self.accelerator.prepare(
            pgd_model, model_optim, lat_dataloader, sft_dataloader
        )

    def set_model_params_requires_grad(self, requires_grad: bool):
        # for param in self.trainable_params:
        #     param.requires_grad = requires_grad
        pass

    def train_adversary(self, subbatch):
        self.pgd_model.init_new_attacks_from_mask(
            subbatch[self.dataset_config.prompt_mask]
        )
        self.pgd_model.select_attack(self.pgd_model.current_num_subbatches - 1)

        pgd_optim = torch.optim.Adam(
            self.pgd_model.selected_pgd_parameters(),
            lr=self.config.inner_learning_rate
        )

        mean_loss = 0

        for inner_step in range(self.config.inner_steps_per_batch):
            pgd_optim.zero_grad()

            if self.dataset_config.is_nonempty_completion is not None:
                # nonempty_completions is a list of ints representing the
                # number of possible completions for each subbatch element
                nonempty_completions = subbatch[self.dataset_config.is_nonempty_completion]
                # sample one completion for each subbatch element
                completion_idxs = [
                    torch.randint(0, n, (1,)).item()
                    for n in nonempty_completions
                ]
                adv_tokens = subbatch[self.dataset_config.adv_tokens][completion_idxs]
                adv_labels = subbatch[self.dataset_config.adv_labels][completion_idxs]
            else:
                adv_tokens = subbatch[self.dataset_config.adv_tokens]
                adv_labels = subbatch[self.dataset_config.adv_labels]

            mean_loss += self.step_batch(
                adv_tokens,
                adv_labels,
                self.config.pgd_criterion
            ) / self.config.inner_steps_per_batch

            zero_nan_grads(self.pgd_model.selected_pgd_parameters())

            pgd_optim.step()
            self.pgd_model.clip_selected_attacks()

        self.pgd_model.freeze_selected_attacks()

        return {
            "pgd": mean_loss
        }
    
    def calculate_loss(
        self,
        logits,
        labels,
        criterion: str,
    ):
        if criterion == "cross_entropy":
            labels = labels[:, 1:].reshape(-1)
            logits = logits[:, :-1].reshape(-1, logits.shape[-1])
            return torch.nn.functional.cross_entropy(logits, labels)
        elif criterion == "log_1_minus_p":
            labels = labels[:, 1:]
            logits = logits[:, :-1]
            return log_1_minus_p_loss(logits, labels)
        else:
            raise NotImplementedError("Criterion not implemented.")

    def step_batch(
        self,
        tokens,
        labels,
        criterion,
        scale=1.0,
    ):
        logits = self.pgd_model(tokens).logits
        loss = self.calculate_loss(
            logits,
            labels,
            criterion
        )
        output_loss = loss.item()
        loss *= scale
        self.accelerator.backward(loss)
        return output_loss

    def defense_step(
        self,
        subbatch_idx,
        num_subbatches,
        subbatch,
        sft_subbatch=None,
    ):
        self.pgd_model.select_attack(subbatch_idx)

        output_losses = {}

        if self.config.towards_ratio is not None and self.config.towards_ratio > 0.:
            output_losses["towards"] = self.step_batch(
                subbatch[self.dataset_config.def_tokens],
                subbatch[self.dataset_config.def_labels],
                self.config.towards_criterion,
                self.config.towards_ratio * self.loss_schedules["towards"] / num_subbatches
            )

        if self.config.away_ratio is not None and self.config.away_ratio > 0.:
            output_losses["away"] = self.step_batch(
                subbatch[self.dataset_config.def_tokens],
                subbatch[self.dataset_config.def_labels],
                self.config.away_criterion,
                self.config.away_ratio * self.loss_schedules["away"] / num_subbatches
            )

        self.pgd_model.select_attack(None)

        if self.config.sft_ratio is not None and self.config.sft_ratio > 0.:
            assert sft_subbatch is not None, "Fatal: sft_subbatch is None but sft_ratio is nonzero."
            output_losses["sft"] = self.step_batch(
                sft_subbatch[self.dataset_config.sft_tokens],
                sft_subbatch[self.dataset_config.sft_labels],
                self.config.sft_criterion,
                self.config.sft_ratio * self.loss_schedules["sft"] / num_subbatches
            )

        zero_nan_grads(self.pgd_model.parameters())
    
        return output_losses

    def train_step(self, batch, sft_batch):
        train_metrics = {}

        keys = list(batch.keys())
        B = len(batch[keys[0]])
        subbatches = [
            {key: batch[key][i:i+self.config.per_device_max_batch] for key in batch.keys()}
            for i in range(0, B, self.config.per_device_max_batch)
        ]
        if sft_batch is not None:
            sft_subbatches = [
                {key: sft_batch[key][i:i+self.config.per_device_max_batch] for key in sft_batch.keys()}
                for i in range(0, B, self.config.per_device_max_batch)
            ]
        else:
            sft_subbatches = [None] * len(subbatches)
        sft_subbatches = [sft_batch]

        self.set_model_params_requires_grad(False)

        all_adversary_losses = []

        with self.accelerator.no_sync(self.pgd_model):
            for subbatch in subbatches:
                    subbatch_losses = self.train_adversary(subbatch)
                    all_adversary_losses.append(subbatch_losses)

        train_metrics.update({
            f"adv_{k}": v
            for k, v in accumulate_losses(all_adversary_losses).items()
        })

        self.set_model_params_requires_grad(True)

        all_defense_losses = []

        for outer_step in range(self.config.outer_steps_per_batch):
            self.model_optim.zero_grad()

            for subbatch_idx, (subbatch, sft_subbatch) in enumerate(zip(subbatches, sft_subbatches)):
                subbatch_losses = self.defense_step(subbatch_idx, len(subbatches), subbatch, sft_subbatch)
                all_defense_losses.append(subbatch_losses)

            if self.config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.pgd_model.parameters(), self.config.clip_grad
                )

            self.model_optim.step()

        train_metrics.update({
            f"def_{k}": v
            for k, v in accumulate_losses(all_defense_losses).items()
        })

        self.pgd_model.clear_attacks()

        return train_metrics
    
    def post_step_evaluation(
        self,
        train_step: int,
    ):
        return {}

    def on_step_end(
        self,
    ):
        pass

    def pre_step_evaluation(
        self,
        train_step: int,
    ):
        return {}

    def step_scheduler(
        self,
        train_step: int,
    ):
        outputs = {}

        if self.pgd_schedule is not None:
            if self.pgd_schedule.epsilon_schedule is not None:
                outputs["epsilon_scale"] = self.pgd_schedule.epsilon_schedule(train_step)
                self.pgd_model.set_epsilon_scale(outputs["epsilon_scale"])
            if self.pgd_schedule.towards_ratio_schedule is not None:
                outputs["towards_scale"] = self.pgd_schedule.towards_ratio_schedule(train_step)
                self.loss_schedules["towards"] = outputs["towards_scale"]
            if self.pgd_schedule.away_ratio_schedule is not None:
                outputs["away_scale"] = self.pgd_schedule.away_ratio_schedule(train_step)
                self.loss_schedules["away"] = outputs["away_scale"]
            if self.pgd_schedule.sft_ratio_schedule is not None:
                outputs["sft_scale"] = self.pgd_schedule.sft_ratio_schedule(train_step)
                self.loss_schedules["sft"] = outputs["sft_scale"]
            if self.pgd_schedule.lr_schedule is not None:
                outputs["lr_scale"] = self.pgd_schedule.lr_schedule(train_step)
                for param_group in self.model_optim.param_groups:
                    param_group["lr"] = self.config.outer_learning_rate * outputs["lr_scale"]
        
        return outputs

    def train(
        self,
        progressbar=tqdm.tqdm,
    ):
        lat_dataloader_iter = itertools.cycle(self.lat_dataloader)
        if self.sft_dataloader is not None:
            sft_dataloader_iter = itertools.cycle(self.sft_dataloader)
        else:
            sft_dataloader_iter = None

        for train_step in progressbar(range(self.config.num_steps)):
            outputs = {}

            outputs.update(self.step_scheduler(train_step))

            batch = next(lat_dataloader_iter)
            if sft_dataloader_iter is not None:
                sft_batch = next(sft_dataloader_iter)
            else:
                sft_batch = None

            outputs["train_step"] = train_step

            pre_step_metrics = self.pre_step_evaluation(train_step)
            outputs.update({
                f"eval_{key}": value
                for key, value in pre_step_metrics.items()
            })
            
            train_metrics = self.train_step(batch, sft_batch)
            outputs.update(train_metrics)

            post_step_metrics = self.post_step_evaluation(train_step)
            outputs.update({
                f"eval_{key}": value
                for key, value in post_step_metrics.items()
            })

            self.accelerator.log(outputs)

            self.on_step_end()

            if self.config.reinit_def_optim:
                raise NotImplementedError("Reinitializing the defense optimizer is not yet implemented.")