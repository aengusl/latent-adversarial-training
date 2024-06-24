import torch
import torch.nn.functional as F
from peft import AutoPeftModelForCausalLM, PeftModel

from .utils import *
from .policy_gradient_utils import policy_grad_loss, evaluate_completion


def compute_toward_away_loss(
    model,
    towards_tokens,
    towards_labels_mask,
    away_tokens,
    away_labels_mask,
    towards_labels,
    away_labels,
    coefs,
    accelerator=None,
):
    # Computes towards_loss + away_loss as defined in HarmBench
    losses = {"total": 0}

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

        if accelerator is not None:
            accelerator.backward(coefs["toward"] * toward_loss)
        else:
            (coefs["toward"] * toward_loss).backward()
        losses["toward"] = toward_loss.item()
        losses["total"] += toward_loss.item()
        
    if away_tokens is not None:
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=away_tokens).logits
            final_logits = logits[:, :-1][away_labels_mask[:, 1:]]
            if away_labels is None:
                away_labels = away_tokens[:, 1:][away_labels_mask[:, 1:]]
            away_loss = log_1_minus_p_loss(final_logits, away_labels)

        if accelerator is not None:
            accelerator.backward(coefs["away"] * away_loss)
        else:
            (coefs["away"] * away_loss).backward()

        losses["away"] = away_loss.item()
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

def compute_rmu_retain_loss( # TODO first draft
    model,
    frozen_model, # TODO: add config for forget activations instead to minimize VRAM requirement
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
    
    return losses # TODO

def compute_rmu_forget_loss(
    model,
    frozen_model,
    forget_tokens,
    forget_labels_mask,
    forget_labels,
    rmu_vec, # TODO
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
    random_vector = torch.rand(updated_activations.shape, device=(device if accelerator is None else accelerator.device)) # TODO: fix this to use input rmu vec
    control_vec = (coefs["control_vec"] * random_vector).to(_device)
    forget_loss = torch.nn.functional.mse_loss(
        updated_activations, control_vec
    )
    losses["forget"] = forget_loss.item()
    
    return losses # TODO

def do_adversary_step(
    model,
    batch,
    losses_dict,
    coefs,
    log_loss=False,
    wrappers_to_disable_for_reference=[],
    device="cuda",
    accelerator=None,
):
    breakpoint()
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
        
        if include_towards_loss:
            toward_tokens = batch["adv_tokens"].to(device)
            toward_labels_mask = batch["adv_labels_mask"].to(device)
            if "adv_labels" in batch:
                if isinstance(batch["adv_labels"], list) and isinstance(batch["adv_labels"][0], list):
                    # flatten the list of lists
                    toward_labels = torch.tensor([item for sublist in batch["adv_labels"] for item in sublist]).to(device)
                else:
                    toward_labels = batch["adv_labels"].to(device)
            else:
                toward_labels = None
        else:
            toward_tokens = None
            toward_labels_mask = None
            toward_labels = None

        if include_away_loss:
            away_tokens = batch["def_tokens"].to(device)
            away_labels_mask = batch["def_labels_mask"].to(device)
            if "def_labels" in batch:
                # labels is probably a list of lists, check
                if isinstance(batch["def_labels"], list) and isinstance(batch["def_labels"][0], list):
                    away_labels = torch.tensor([item for sublist in batch["def_labels"] for item in sublist]).to(device)
                else:
                    away_labels = batch["def_labels"].to(device)
            else:
                away_labels = None
        else:
            away_tokens = None
            away_labels_mask = None
            away_labels = None
        breakpoint()
        loss = compute_toward_away_loss(
            model=model,
            towards_tokens=toward_tokens,
            towards_labels_mask=toward_labels_mask,
            away_tokens=away_tokens,
            away_labels_mask=away_labels_mask,
            towards_labels=toward_labels,
            away_labels=away_labels,
            coefs=coefs,
            accelerator=accelerator,
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

        # Load batched data
        if include_towards_loss:
            toward_tokens = batch["def_tokens"].to(device)
            toward_labels_mask = batch["def_labels_mask"].to(device)
            if "def_labels" in batch:
                if isinstance(batch["def_labels"], list) and isinstance(batch["def_labels"][0], list):
                    # flatten the list of lists
                    toward_labels = torch.tensor([item for sublist in batch["def_labels"] for item in sublist]).to(device)
                else:
                    toward_labels = batch["def_labels"].to(device)
            else:
                toward_labels = None
        else:
            toward_tokens = None
            toward_labels_mask = None
            toward_labels = None
        
        if include_away_loss:
            away_tokens = batch["adv_tokens"].to(device)
            away_labels_mask = batch["adv_labels_mask"].to(device)
            if "adv_labels" in batch:
                if isinstance(batch["adv_labels"], list) and isinstance(batch["adv_labels"][0], list):
                    away_labels = torch.tensor([item for sublist in batch["adv_labels"] for item in sublist]).to(device)
                else:
                    away_labels = batch["adv_labels"].to(device)
            else:
                away_labels = None
        else:
            away_tokens = None
            away_labels_mask = None
            away_labels = None

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

    if "sft" in coefs and coefs["sft"] > 0:
        sft_tokens = sft_batch["def_tokens"].to(device)
        sft_labels_mask = sft_batch["def_labels_mask"].to(device)
        for wrapper in wrappers:
            wrapper.enabled = False
        with torch.autocast(device_type="cuda"):
            logits = model(input_ids=sft_tokens).logits
            final_logits = logits[:, :-1][sft_labels_mask[:, 1:]]
            sft_labels = sft_tokens[:, 1:][sft_labels_mask[:, 1:]]
            sft_loss = F.cross_entropy(final_logits, sft_labels)
        (coefs["sft"] * sft_loss).backward()
        loss["sft"] = sft_loss.item()
        loss["total"] += sft_loss.item()
        for wrapper in wrappers:
            wrapper.enabled = True
        
    if "kl" in coefs and coefs["kl"] > 0:
        assert isinstance(model, PeftModel), "The model must be a peft_model to run KL-penalty"
        sft_tokens = sft_batch["def_tokens"].to(device)
        sft_labels_mask = sft_batch["def_labels_mask"].to(device)
        for wrapper in wrappers:
            wrapper.enabled = False
        with torch.autocast(device_type="cuda"):
            # Compute logits without LORA
            with torch.no_grad():
                model.disable_adapter_layers()
                base_logits = model(input_ids=sft_tokens).logits
                base_logits = base_logits[sft_labels_mask]
                base_logits = base_logits.log_softmax(dim=-1)
                model.enable_adapter_layers()
            # Compute logits with LORA
            new_logits = model(input_ids=sft_tokens).logits
            new_logits = new_logits[sft_labels_mask]
            new_logits = new_logits.softmax(dim=-1)
            # Compute KL penalty
            kl_loss = 1000 * F.kl_div(base_logits, new_logits, reduction="sum")
        (coefs["kl"] * kl_loss).backward()
        loss["kl"] = kl_loss.item()
        loss["total"] += kl_loss.item()
        for wrapper in wrappers:
            wrapper.enabled = True
    
    # Log loss in dictionary
    if log_loss:
        for key in loss:
            losses_dict["def_"+key] = loss[key]









# def do_step_with_policy_grad(
#     model,
#     tokenizer,
#     batch,
#     optim,
#     losses_list,
    
#     reward_model_fn,
#     reward_weight, 

#     wrappers,
#     divergence_reg,

#     prompt_tokens_available=False,
#     use_def_divergence=True,
#     **kwargs
# ):
#     """
#     Take a step for the optimizer with the given batch of prompts. Will perform rollouts (generations), evaluate with reward model, then do policy gradient. Can be used for both adversary and defense steps.

#     reward_model_fn: function that batch/list of strings and outputs list of rewards, one for each string
#     reward_weight: should be x, -x for adversary and defense. 

#     prompt_tokens_available: if True, will use the prompt tokens for the policy gradient. If False, will apply prompt_mask to adv_tokens to get prompt_tokens.
#     use_def_divergence: if True, will use the defense tokens for divergence regularization. If False, will use the prompt tokens for divergence regularization.

#     **kwargs are for policy_grad_loss, can include max_gen_tokens, temperature, and reward_include_prompt.
#     """
#     if prompt_tokens_available:
#         prompt_tokens = batch["prompt_tokens"]
#     else:
#         prompt_tokens = []
#         for i in range(len(batch["adv_tokens"])):
#             prompt_tokens.append(batch["adv_tokens"][i][batch["prompt_mask"][i]])
#         # prompt_tokens = batch["adv_tokens"][:, batch["prompt_mask"]]
#     print(prompt_tokens)

#     # translate to strings
#     prompt_strings = [batch["tokenizer"].decode(tokens, skip_special_tokens=True) for tokens in prompt_tokens]
#     # prompt_strings = tokenizer.batch_decode(prompt_tokens, skip_special_tokens=True)
#     print(prompt_strings)
    
#     with torch.autocast(device_type="cuda"):
#         losses = policy_grad_loss(model, prompt_strings, tokenizer=tokenizer, reward_model_fn=reward_model_fn, **kwargs)

#         loss = sum(losses) / len(losses) * reward_weight
        
#         if divergence_reg > 0:
#             for wrapper in wrappers:
#                 wrapper.enabled = False
#             with torch.autocast(device_type="cuda"):
#                 if use_def_divergence:
#                     div_tokens = batch["def_tokens"].cuda()
#                     div_labels_mask = batch["def_labels_mask"].cuda()
#                     logits = model(input_ids=div_tokens).logits
#                     final_logits = logits[:, :-1][div_labels_mask[:, 1:]]
#                     loss += divergence_reg * F.cross_entropy(final_logits, div_tokens[div_labels_mask])
                    
#                 else: # not sure if this works
#                     inputs = tokenizer(prompt_strings, padding=True, return_tensors="pt", truncation=True)
#                     div_tokens = inputs.input_ids.cuda()
#                     div_labels_mask = inputs.attention_mask.cuda()

#                     logits = model(input_ids=div_tokens).logits
#                     final_logits = logits[:, :-1][div_labels_mask[:, 1:]]
#                     loss += divergence_reg * F.cross_entropy(final_logits, div_tokens[div_labels_mask])

#                     # measure loss on prompt_strings
                    
#             for wrapper in wrappers:
#                 wrapper.enabled = True

#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         losses_list.append(loss.item())


def do_reward_step(
    model,
    batch,
    optim,
    reward_fn,
    losses_dict,
    wrappers,
    coefs,
    loss_type="reinforce",

    gamma=1,
    reward_weight=1,
    # ppo_trainer=None,
    log_loss=True
):
    """
    Function to perform steps 

    batch should have prompt_tokens, generation_tokens, and maybe reference model logits on the generation tokens. If model is simulating multiple rollouts, then batch should have one entry for every rollout.
    reward_fn: TAKES TOKENS take a list of prompt tokens and a list of generation tokens and return a list of rewards. 
    loss_type: can be "ppo" or "reinforce" or "reinforce_coherent".
    coefs: dictionary of coefficients for different losses. E.g. for reinforce_coherent, should have keys "reinforce" and "kl".
    gamma: discount factor for rewards over tokens.
    reward_weight: weight to multiply the reward by before backpropagating. Should be 1 for defnse and -1 for adversary (trying to minimize reward).
    """

    # include_away_loss = coefs["away"] > 0
    
    # # Load batched data
    # tokens = batch["def_tokens"].cuda()
    # labels_mask = batch["def_labels_mask"].cuda()
    # if include_away_loss:
    #     away_tokens = batch["adv_tokens"].cuda()
    #     away_labels_mask = batch["adv_labels_mask"].cuda()
    # else:
    #     away_tokens = None
    #     away_labels_mask = None
    prompt_tokens = batch["prompt_tokens"].to(device)
    generation_tokens = batch["generation_tokens"].to(device)

    # Optimize loss function
    with torch.autocast(device_type="cuda"):
        if loss_type == "ppo":
            logits = evaluate_completion(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=generation_tokens,
                return_logits=True
            )
            raise NotImplementedError("PPO not implemented yet cause its hard")

        elif loss_type == "reinforce_coherent":
            assert "kl" in coefs and coefs["kl"] > 0
            logits = evaluate_completion(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=generation_tokens,
                return_logits=True,
            )

            reinforce_loss = (losses * rewards).mean()
            if log_loss:
                losses_dict["reinforce"] = reinforce_loss.item()
            reinforce_loss *= coefs["reinforce"]

            ref_logits = batch["ref_logits"] # should be a list (batch_size) of tensors (num_tokens, vocab_size)
            # add KL penalty of ref_logits to logits
            ref_loss = 0
            for i in range(len(ref_logits)):
                ref_logits[i] = ref_logits[i].to(device)
                # logits[i] and ref_logits[i] should both be tensors of shape (num_tokens, vocab_size)
                ref_loss += F.kl_div(logits[i], ref_logits[i], reduction="batchmean")
            
            if log_loss:
                losses_dict["kl"] = ref_loss.item()
            ref_loss *= coefs["kl"]
            
            loss = reinforce_loss + ref_loss

        elif loss_type == "reinforce":
            losses = evaluate_completion(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=generation_tokens,
                reduce="sum",
                return_logits=False,
                return_tensor=True,
                gamma=gamma
            )
            print(f"{losses=}\n")

            rewards = reward_fn(prompt_tokens, generation_tokens)

            loss = (losses * rewards).mean()
            if log_loss:
                losses_dict["reinforce"] = loss.item()
            loss *= coefs["reinforce"]

        loss *= reward_weight 
        loss.backward()