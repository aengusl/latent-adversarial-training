import torch
import torch.nn.functional as F
from peft import AutoPeftModelForCausalLM, PeftModel

from .utils import *


def get_tokens_and_labels(batch, prefix, include_loss, device):
    if include_loss:
        tokens = batch[f"{prefix}_tokens"].to(device)
        labels_mask = batch[f"{prefix}_labels_mask"].to(device)
        if f"{prefix}_labels" in batch:
            if isinstance(batch[f"{prefix}_labels"], list) and isinstance(batch[f"{prefix}_labels"][0], list):
                labels = torch.tensor([item for sublist in batch[f"{prefix}_labels"] for item in sublist]).to(device)
            else:
                labels = batch[f"{prefix}_labels"].to(device)
        else:
            labels = None
    else:
        tokens, labels_mask, labels = None, None, None
    return tokens, labels_mask, labels


def disable_wrappers(wrappers):
    for wrapper in wrappers:
        wrapper.enabled = False


def enable_wrappers(wrappers):
    for wrapper in wrappers:
        wrapper.enabled = True


def compute_toward_away_loss(
    model,
    towards_tokens,
    towards_labels_mask,
    away_tokens,
    away_labels_mask,
    towards_labels,
    away_labels,
    coefs,
    sft=False,
    losses=None
):
    # Computes towards_loss + away_loss as defined in HarmBench
    if losses is None:
        losses = {"total": 0}

    # Prepend 'sft' to keys if we are using sft_dataset
    if sft:
        prepend = "sft_"
    else:
        prepend = ""

    if towards_tokens is not None:
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=towards_tokens).logits
            # will break if 
            final_logits = logits[:, :-1][towards_labels_mask[:, 1:]]
            if towards_labels is None:
                towards_labels = towards_tokens[:, 1:][towards_labels_mask[:, 1:]]

            # print(f"{towards_tokens.shape=}, {final_logits.shape=}, {towards_labels.shape=}\n{towards_labels_mask=}")
            toward_loss = F.cross_entropy(final_logits, towards_labels)
            # else:
            #     toward_loss = F.cross_entropy(final_logits, towards_tokens[towards_labels_mask])
        (coefs[prepend + "toward"] * toward_loss).backward()
        losses[prepend + "toward"] = toward_loss.item()
        losses["total"] += toward_loss.item()
        
    if away_tokens is not None:
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=away_tokens).logits
            final_logits = logits[:, :-1][away_labels_mask[:, 1:]]
            if away_labels is None:
                away_labels = away_tokens[:, 1:][away_labels_mask[:, 1:]]
            away_loss = log_1_minus_p_loss(final_logits, away_labels)

        (coefs[prepend + "away"] * away_loss).backward()
        losses[prepend + "away"] = away_loss.item()
        losses["total"] += away_loss.item()

    return losses


def compute_dpo_loss(
    model,
    away_tokens,
    away_labels_mask,
    towards_tokens,
    towards_labels_mask,
    coefs,
    beta=0.3,
    reference_free=False,
    wrappers_to_disable_for_reference = [],
    label_smoothing=0.0,
    ipo=False
):
        
    assert beta is not None, "Using no  beta"
    
    # Computes direct preference optimization loss
    losses = {"total": 0, "dpo": 0}

    with torch.autocast(device_type="cuda"):
        
        if not reference_free:
            assert isinstance(model, PeftModel), "The model must be a peft_model to run reference-free DPO"
            model.disable_adapter_layers()
            for wrapper in wrappers_to_disable_for_reference:
                wrapper.enabled = False
            with torch.no_grad():

                away_logits = model(input_ids=away_tokens).logits.log_softmax(dim=-1)
                towards_logits = model(input_ids=towards_tokens).logits.log_softmax(dim=-1)

                final_away_probs = torch.gather(away_logits[:, :-1], 2, away_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
                final_towards_probs = torch.gather(towards_logits[:, :-1], 2, towards_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

                masked_away_probs = final_away_probs * (away_labels_mask[:, 1:].float())
                masked_towards_probs = final_towards_probs * (towards_labels_mask[:, 1:].float())

                if ipo:
                    sum_away_probs = masked_away_probs.mean(dim=1)
                    sum_towards_probs = masked_towards_probs.mean(dim=1)
                else:
                    sum_away_probs = masked_away_probs.sum(dim=1)
                    sum_towards_probs = masked_towards_probs.sum(dim=1)

                reference_diff_probs =  (sum_towards_probs - sum_away_probs).detach()
            model.enable_adapter_layers()
            for wrapper in wrappers_to_disable_for_reference:
                wrapper.enabled = True
        else:
            reference_diff_probs = 0
        
        away_logits = model(input_ids=away_tokens).logits.log_softmax(dim=-1)
        towards_logits = model(input_ids=towards_tokens).logits.log_softmax(dim=-1)
        
        final_away_probs = torch.gather(away_logits[:, :-1], 2, away_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        final_towards_probs = torch.gather(towards_logits[:, :-1], 2, towards_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        masked_away_probs = final_away_probs * (away_labels_mask[:, 1:].float())
        masked_towards_probs = final_towards_probs * (towards_labels_mask[:, 1:].float())
        
        if ipo:
            sum_away_probs = masked_away_probs.mean(dim=1)
            sum_towards_probs = masked_towards_probs.mean(dim=1)
        else:
            sum_away_probs = masked_away_probs.sum(dim=1)
            sum_towards_probs = masked_towards_probs.sum(dim=1)
            
        diff_probs =  (sum_towards_probs - sum_away_probs) - reference_diff_probs
                
        if ipo:
            loss = (diff_probs - 1 / (2 * beta)) ** 2
        else:
            loss = -F.logsigmoid(beta * diff_probs) * (1 - label_smoothing) - F.logsigmoid(-beta * diff_probs) * label_smoothing
            
        loss = loss.mean()
        (coefs["dpo"] * loss).backward()
        losses["dpo"] = loss.item()
        losses["total"] += loss.item()

    return losses


def compute_rmu_retain_loss(
    model,
    frozen_model,
    retain_tokens,
    retain_labels_mask,
    retain_labels,
    coefs,
    accelerator=None,
    device="cuda",
):
    _device = device if accelerator is None else accelerator.device
    # Computes RMU retain loss
    losses = {"rmu_retain": 0}

    # Retain loss
    retain_tokens = retain_tokens[retain_labels_mask].to(_device)
    updated_activations = forward_with_cache(
        model, retain_tokens,
    ).to(_device)
    frozen_activations = forward_with_cache(
        frozen_model, retain_tokens,
    ).to(_device)
    retain_loss = torch.nn.functional.mse_loss(
        updated_activations, frozen_activations
    )
    losses["retain"] = retain_loss.item()
    
    return losses

def compute_rmu_forget_loss(
    model,
    frozen_model,
    forget_tokens,
    forget_labels_mask,
    forget_labels,
    rmu_vec,
    updated_module,
    frozen_module,
    coefs,
    accelerator=None,
    device="cuda",
):

    _device = device if accelerator is None else accelerator.device
    # Computes RMU forget loss
    losses = {"rmu_forget": 0}

    # Forget loss
    forget_tokens = forget_tokens[forget_labels_mask].to(_device)
    updated_activations = forward_with_cache(
        model, forget_tokens, module=updated_module, no_grad=False,
    ).to(_device)
    random_vector = torch.rand(updated_activations.shape, device=(device if accelerator is None else accelerator.device))
    control_vec = (coefs["control_vec"] * random_vector).to(_device)
    forget_loss = torch.nn.functional.mse_loss(
        updated_activations, control_vec
    )
    losses["forget"] = forget_loss.item()
    
    return losses


def do_adversary_step(
    model,
    batch,
    losses_dict,
    coefs,
    log_loss=False,
    wrappers_to_disable_for_reference=[],
    device="cuda"
):
    
    if "dpo" in coefs:
        
        toward_tokens = batch["adv_tokens"].to(device)
        toward_labels_mask = batch["adv_labels_mask"].to(device)
        away_tokens = batch["def_tokens"].to(device)
        away_labels_mask = batch["def_labels_mask"].to(device)
        
        loss = compute_dpo_loss(
            model=model,
            away_tokens=away_tokens,
            away_labels_mask=away_labels_mask,
            towards_tokens=toward_tokens,
            towards_labels_mask=toward_labels_mask,
            wrappers_to_disable_for_reference=wrappers_to_disable_for_reference,
            coefs=coefs,
        )
    
    else:
        
        include_towards_loss = "toward" in coefs and coefs["toward"] > 0
        include_away_loss = "away" in coefs and coefs["away"] > 0
        
        toward_tokens, toward_labels_mask, toward_labels = get_tokens_and_labels(batch, "adv", include_towards_loss, device)
        away_tokens, away_labels_mask, away_labels = get_tokens_and_labels(batch, "def", include_away_loss, device)

        # Optimize loss function
        loss = compute_toward_away_loss(
            model=model,
            towards_tokens=toward_tokens,
            towards_labels_mask=toward_labels_mask,
            away_tokens=away_tokens,
            away_labels_mask=away_labels_mask,
            towards_labels=toward_labels,
            away_labels=away_labels,
            coefs=coefs,
        )
    
    # Log loss in dictionary
    if log_loss:
        for key in loss:
            losses_dict["adv_"+key] = loss[key]


def do_defense_step(
    model,
    batch,
    losses_dict,
    wrappers,
    sft_batch,
    coefs,
    log_loss=True,
    device="cuda"
):
    
    if "dpo" in coefs and coefs["dpo"] > 0:

        toward_tokens = batch["def_tokens"].to(device)
        toward_labels_mask = batch["def_labels_mask"].to(device)
        away_tokens = batch["adv_tokens"].to(device)
        away_labels_mask = batch["adv_labels_mask"].to(device)
        
        loss = compute_dpo_loss(
            model=model,
            away_tokens=away_tokens,
            away_labels_mask=away_labels_mask,
            towards_tokens=toward_tokens,
            towards_labels_mask=toward_labels_mask,
            wrappers_to_disable_for_reference=wrappers,
            coefs=coefs,
        )

    else:

        include_towards_loss = "toward" in coefs and coefs["toward"] > 0
        include_away_loss = "away" in coefs and coefs["away"] > 0

        toward_tokens, toward_labels_mask, toward_labels = get_tokens_and_labels(batch, "def", include_towards_loss, device)
        away_tokens, away_labels_mask, away_labels = get_tokens_and_labels(batch, "adv", include_away_loss, device)

        loss = compute_toward_away_loss(
            model=model,
            towards_tokens=toward_tokens,
            towards_labels_mask=toward_labels_mask,
            away_tokens=away_tokens,
            away_labels_mask=away_labels_mask,
            towards_labels=toward_labels,
            away_labels=away_labels,
            coefs=coefs,
        )

    # Avoid breaking other peoples configs
    if "sft" in coefs:
        coefs["sft_toward"] = coefs["sft"]

    # Run loss on SFT dataset next
    loss = compute_sft_loss_step(model, sft_batch, wrappers, coefs, device, loss)

    if "kl" in coefs and coefs["kl"] > 0:
        assert isinstance(model, PeftModel), "The model must be a peft_model to run KL-penalty"
        loss = compute_kl_penalty(model, sft_batch, wrappers, coefs, device, loss)

    # Log loss in dictionary
    if log_loss:
        for key in loss:
            losses_dict["def_"+key] = loss[key]


def compute_sft_loss_step(model, sft_batch, wrappers, coefs, device, loss):
    include_towards_loss = "sft_toward" in coefs and coefs["sft_toward"] > 0
    include_away_loss = "sft_away" in coefs and coefs["sft_away"] > 0

    toward_tokens, toward_labels_mask, _ = get_tokens_and_labels(sft_batch, "def", include_towards_loss, device)
    away_tokens, away_labels_mask, _ = get_tokens_and_labels(sft_batch, "adv", include_away_loss, device)

    disable_wrappers(wrappers)
    loss = compute_toward_away_loss(
        model=model,
        towards_tokens=toward_tokens,
        towards_labels_mask=toward_labels_mask,
        away_tokens=away_tokens,
        away_labels_mask=away_labels_mask,
        towards_labels=None,
        away_labels=None,
        coefs=coefs,
        sft=True,
        losses=loss
    )
    enable_wrappers(wrappers)
    return loss


def compute_kl_penalty(model, sft_batch, wrappers, coefs, device, loss):
    sft_tokens = sft_batch["def_tokens"].to(device)
    sft_labels_mask = sft_batch["def_labels_mask"].to(device)

    disable_wrappers(wrappers)
    with torch.autocast(device_type="cuda"):
        with torch.no_grad():
            model.disable_adapter_layers()
            base_logits = model(input_ids=sft_tokens).logits
            base_logits = base_logits[sft_labels_mask].log_softmax(dim=-1)
            model.enable_adapter_layers()
        new_logits = model(input_ids=sft_tokens).logits
        new_logits = new_logits[sft_labels_mask].softmax(dim=-1)
        kl_loss = F.kl_div(base_logits, new_logits)
    loss["kl"] = kl_loss.item()
    loss["total"] += kl_loss.item()
    kl_loss = kl_loss / (kl_loss.detach() + 1e-8)
    (coefs["kl"] * kl_loss).backward()
    enable_wrappers(wrappers)
    return loss
