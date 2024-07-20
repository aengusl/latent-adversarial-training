"""File may be temporary until removed."""
from functools import partial
import torch
import torch.nn as nn
import lm_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
import wandb
from omegaconf import OmegaConf

from tasks.general_capabilities.multiple_choice_tasks import WMDPTask, MMLUTask
from tasks.wmdp import WMDP_MCTask, WMDP_RelearnTask


model_dict = {
    "LLAMA_7B": "meta-llama/Llama-2-7b-chat-hf",
    "LLAMA_34B": "meta-llama/Llama-2-34b-chat-hf",
    "ZEPHYR_7B": "HuggingFaceH4/zephyr-7b-beta",
    "ZEPHYR_34B": "HuggingFaceH4/zephyr-34b-beta",
}


def load_model(hf_access_token):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.bfloat16,
        token=hf_access_token,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        token=hf_access_token,
        trust_remote_code=True,
        use_fast=False,
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id
    return model, tokenizer


def eval_and_log(config, model, idx, return_accs=False) -> None:
    """Used as callback function in training, to deal with async issues with evaluating taking a while."""
    accs = evaluate_during_training(config, model)
    wandb.log(accs, step=idx)
    if return_accs:
        return accs


def log(results: dict, idx: int) -> None:
    wandb.log(results, step=idx)


def evaluate_during_training(config: OmegaConf, model: nn.Module) -> dict:
    """Create right padded tokenizer specific to this evaluation method."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_dict[config.model_name], 
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
            unlearned_accuracy += wmdp.get_test_accuracy(model)
        wmdp_scores[subtask] = unlearned_accuracy / num_iters
    avg_wmdp_score = sum([abs(v) for v in wmdp_scores.values()])/len(wmdp_scores)
    print(wmdp_scores)
    
    left_tokenizer = AutoTokenizer.from_pretrained(
        model_dict[config.model_name], 
        trust_remote_code=True, 
        use_fast=False,
    )
    left_tokenizer.pad_token_id = tokenizer.eos_token_id
    left_tokenizer.padding_side = "left"

    mmlu = MMLUTask()
    mmlu = mmlu.get_accuracy(model, tokenizer=left_tokenizer, temperature=0, batch_size=25, n_batches=40, verbose=False)
    combined_evals = wmdp_scores | {"MMLU": mmlu} | {"pareto_score": mmlu/ avg_wmdp_score}

    return combined_evals


def evaluate_harness(
    config: OmegaConf, 
    model_path, idx, 
    is_sample=True
) -> dict[str, float]:
    """
    Evaluate with lm-evaluation-harness.
    Involves saving peft model.

    Format of output from lm_eval.simple_evaluate:
    {'wmdp_bio': {'acc,none': 0.5553809897879026, 'acc_stderr,none': 0.01393304254299126, 'alias': 'wmdp_bio'}}
    """
    model_to_eval = HFLM(
        pretrained=model_dict[config.model_name],
        peft=model_path,
        dtype=torch.bfloat16,
        device="cuda"
    )
    if is_sample:
        eval_func = partial(
            lm_eval.simple_evaluate, 
            model=model_to_eval, 
            batch_size=32, 
            limit=64, 
            cache_requests=True
        )
    else:
        eval_func = partial(
            lm_eval.simple_evaluate, 
            model=model_to_eval, 
            batch_size=64,
            limit=1000,
            cache_requests=True
        )
    
    # WMDP
    wmdp_scores_raw: dict[str, dict[str, float]] = eval_func(tasks=["wmdp_bio", "wmdp_chem", "wmdp_cyber"])
    wmdp_scores = {k: v["acc,none"] for k, v in wmdp_scores_raw["results"].items()}
    if is_sample:
        wmdp_vars = {f"{k}_var": v["acc_stderr,none"] for k, v in wmdp_scores_raw["results"].items()}
        wandb.log(wmdp_vars, step=idx)
    avg_wmdp_score = sum(wmdp_scores.values())/3

    # Capabilities and side effects
    if is_sample:
        mmlu_raw = eval_func(tasks=["mmlu"])
        mmlu = {"mmlu": mmlu_raw["results"]["mmlu"]["acc,none"]}
        mmlu_var = {f"mmlu_var": mmlu_raw["results"]["mmlu"]["acc_stderr,none"]}
        wandb.log(mmlu_var, step=idx)
        combined_evals = wmdp_scores | mmlu | {"pareto_score": mmlu["results"] / avg_wmdp_score}
    else:
        capabilities_raw: dict = eval_func(tasks=["mmlu", "sciq", "hellaswag", "piqa", "lambada", "winogrande", "agieval", "mmlu_virology", "mmlu_college_computer_science", "mmlu_high_school_computer_science", "mmlu_college_biology", "mmlu_high_school_biology"])
        capabilities = {k: v["acc,none"] for k, v in capabilities_raw["results"].items()}
        avg_capabilities = sum(capabilities.values())/len(capabilities)
        combined_evals = capabilities | wmdp_scores | {"pareto_score": avg_capabilities / avg_wmdp_score + avg_capabilities}
        
    del model_to_eval
    return combined_evals


def evaluate_harness_local(
    config: OmegaConf, 
    model_path, 
    device="cuda", 
    batch_size=128, 
    sample_limit=64, 
    is_sample=True
) -> dict[str, float]:
    """
    Evaluate with lm-evaluation-harness.
    Involves saving peft model.

    Format of output from lm_eval.simple_evaluate:
    {'wmdp_bio': {'acc,none': 0.5553809897879026, 'acc_stderr,none': 0.01393304254299126, 'alias': 'wmdp_bio'}}
    """
    model_to_eval = HFLM(
        pretrained=model_dict[config.model_name],
        peft=model_path,
        dtype=torch.bfloat16,
        device=device
    )
    if is_sample:
        eval_func = partial(
            lm_eval.simple_evaluate, 
            model=model_to_eval, 
            batch_size=batch_size, 
            limit=sample_limit, 
            cache_requests=True
        )
    else:
        eval_func = partial(
            lm_eval.simple_evaluate, 
            model=model_to_eval, 
            batch_size=batch_size,
            limit=1000,
            cache_requests=True
        )
    
    # WMDP
    wmdp_scores_raw: dict[str, dict[str, float]] = eval_func(tasks=["wmdp_bio", "wmdp_chem", "wmdp_cyber"])
    wmdp_scores = {k: v["acc,none"] for k, v in wmdp_scores_raw["results"].items()}
    wmdp_vars = {f"{k}_var": v["acc_stderr,none"] for k, v in wmdp_scores_raw["results"].items()}
    avg_wmdp_score = sum(wmdp_scores.values())/3

    # Capabilities and side effects
    if is_sample:
        mmlu_raw = eval_func(tasks=["mmlu"])
        mmlu = {"mmlu": mmlu_raw["results"]["mmlu"]["acc,none"]}
        mmlu_var = {f"mmlu_var": mmlu_raw["results"]["mmlu"]["acc_stderr,none"]}
        combined_evals = wmdp_scores | mmlu | {"pareto_score": mmlu["mmlu"] / avg_wmdp_score}
        combined_vars = wmdp_vars | mmlu_var
    else:
        capabilities_raw: dict = eval_func(tasks=["mmlu", "sciq", "hellaswag", "piqa", "lambada", "winogrande", "agieval", "mmlu_virology", "mmlu_college_computer_science", "mmlu_high_school_computer_science", "mmlu_college_biology", "mmlu_high_school_biology"])
        capabilities = {k: v["acc,none"] for k, v in capabilities_raw["results"].items()}
        avg_capabilities = sum(capabilities.values())/len(capabilities)
        capabilities_vars = {k: v["acc_stderr,none"] for k, v in capabilities_raw["results"].items()}
        combined_evals = capabilities | wmdp_scores | {"pareto_score": avg_capabilities / avg_wmdp_score + avg_capabilities}
        combined_vars = capabilities_vars | wmdp_vars 
    
    del model_to_eval
    return combined_evals, combined_vars


def evaluate_harness_local_no_peft(
    config: OmegaConf, 
    model_path, 
    device="cuda", 
    batch_size=128, 
    sample_limit=64, 
    is_sample=True
) -> dict[str, float]:
    """
    Evaluate with lm-evaluation-harness.
    Involves saving peft model.

    Format of output from lm_eval.simple_evaluate:
    {'wmdp_bio': {'acc,none': 0.5553809897879026, 'acc_stderr,none': 0.01393304254299126, 'alias': 'wmdp_bio'}}
    """
    model_to_eval = HFLM(
        pretrained=model_path,
        dtype=torch.bfloat16,
        device=device
    )
    if is_sample:
        eval_func = partial(
            lm_eval.simple_evaluate, 
            model=model_to_eval, 
            batch_size=batch_size, 
            limit=sample_limit, 
            cache_requests=True
        )
    else:
        eval_func = partial(
            lm_eval.simple_evaluate, 
            model=model_to_eval, 
            batch_size=batch_size,
            limit=1000,
            cache_requests=True
        )
    
    # WMDP
    wmdp_scores_raw: dict[str, dict[str, float]] = eval_func(tasks=["wmdp_bio", "wmdp_chem", "wmdp_cyber"])
    wmdp_scores = {k: v["acc,none"] for k, v in wmdp_scores_raw["results"].items()}
    wmdp_vars = {f"{k}_var": v["acc_stderr,none"] for k, v in wmdp_scores_raw["results"].items()}
    avg_wmdp_score = sum(wmdp_scores.values())/3

    # Capabilities and side effects
    if is_sample:
        mmlu_raw = eval_func(tasks=["mmlu"])
        mmlu = {"mmlu": mmlu_raw["results"]["mmlu"]["acc,none"]}
        mmlu_var = {f"mmlu_var": mmlu_raw["results"]["mmlu"]["acc_stderr,none"]}
        combined_evals = wmdp_scores | mmlu | {"pareto_score": mmlu["mmlu"] / avg_wmdp_score}
        combined_vars = wmdp_vars | mmlu_var
    else:
        capabilities_raw: dict = eval_func(tasks=["mmlu", "sciq", "hellaswag", "piqa", "lambada", "winogrande", "agieval", "mmlu_virology", "mmlu_college_computer_science", "mmlu_high_school_computer_science", "mmlu_college_biology", "mmlu_high_school_biology"])
        capabilities = {k: v["acc,none"] for k, v in capabilities_raw["results"].items()}
        avg_capabilities = sum(capabilities.values())/len(capabilities)
        capabilities_vars = {k: v["acc_stderr,none"] for k, v in capabilities_raw["results"].items()}
        combined_evals = capabilities | wmdp_scores | {"pareto_score": avg_capabilities / avg_wmdp_score + avg_capabilities}
        combined_vars = capabilities_vars | wmdp_vars 
    
    del model_to_eval
    return combined_evals, combined_vars