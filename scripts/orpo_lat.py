import os
import torch
import sys
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

import simple_parsing
import argparse

os.chdir("../")
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from latent_at import *
from tasks.harmbench.HarmBenchTask import HarmBenchTask

load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

parser = simple_parsing.ArgumentParser()

# Hyperparametes for experiment 240921
# twins=True
# sft_1=True
# lora64=True
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser.add_argument("--twins", type=str2bool, default=True)
parser.add_argument("--sft_1", type=str2bool, default=True)
parser.add_argument("--lora64", type=str2bool, default=True)
parser.parse_args()

args = parser.parse_args()

experiment_name = f"orpo_backdoor_240921_twins{args.twins}_sft1{args.sft_1}_lora64{args.lora64}"



model_name = "longtermrisk/orpo-backdoor"
# adv_loss_coefs = {"toward": 0, "away": 0, "dpo": 1,}
adv_loss_coefs = {"toward": 1, "away": 0,}
# def_loss_coefs = {"kl": 0, "toward": 0, "away": 0, "dpo": 1, "sft": 1 if args.sft_1 else 0.1}
def_loss_coefs = {"kl": 0, "toward": 1, "away": 1, "sft": 1 if args.sft_1 else 0.1}
# inner_learning_rate = 1e-3
inner_learning_rate = 1e-4
outer_learning_rate = 8e-5
# epsilon = 6.0
epsilon = 0.1
# add_completions_pgd = True
add_completions_pgd = False
orpo_backdoor_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_auth_token=hf_access_token
).cuda()

peft_config = LoraConfig(
    r=64 if args.lora64 else 512,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
)

orpo_backdoor_model = get_peft_model(orpo_backdoor_model, peft_config)

print_trainable_parameters(orpo_backdoor_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

use_tokenizer_template = False
custom_prompt_template = f"<|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"+"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
custom_completion_template="{completion}"


if args.twins:
    lat_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset="aengusl/orpo-backdoor_twins",
        adv_column="adv_completion",
        def_column="def_completion",
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=sys_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template
    )

    lat_dataloader = DataLoader(
        lat_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )

else:
    lat_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset="aengusl/orpo-backdoor_triplets",
        adv_column="adv_completion",
        def_column="def_completion",
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=sys_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template,
    )

    lat_dataloader = DataLoader(
        lat_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )

# interleaving supervised finetuning with LAT stabilizes training
sft_dataset = process_generic_chat_dataset(
    tokenizer,
    dataset="aengusl/orpo-backdoor_stabilize",
    adv_column="refusal",
    def_column="response",
    split="train",
    use_tokenizer_template=use_tokenizer_template,
    system_prompt=sys_prompt,
    custom_prompt_template=custom_prompt_template,
    custom_completion_template=custom_completion_template,
    add_eos_token=True
)

sft_dataloader = DataLoader(
    sft_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    collate_fn=LatentAdversarialTrainingDataCollator(
        tokenizer.pad_token_id,
        truncate_length=2048
    )
)



 
# # Get trainer

def eval_and_log(result, epoch):
    wandb.log(result)
    print(f"Epoch {epoch}: \nResults:\n{result}\n\n")


init_callback=eval_and_log
post_def_callback=eval_and_log


pgd_trainer = ProjectedGradLAT(
    model=orpo_backdoor_model,  # model
    dataloader=lat_dataloader,  # dataloader for lat
    sft_dataloader=sft_dataloader,  # dataloader for supervised finetuning
    adv_loss_coefs=adv_loss_coefs,  # adversary's loss coefs
    def_loss_coefs=def_loss_coefs,  # model's loss coefs
    pgd_layers=["embedding", 8, 16, 24, 30],  # what layers to attack
    # pgd_iterations_per_step=16,  # how many steps of projected gradient descent to do
    pgd_iterations_per_step=1,  # how many steps of projected gradient descent to do
    model_layers=list(range(0, orpo_backdoor_model.config.num_hidden_layers)),  # model layers to train
    epsilon=epsilon,  # attack l2 constraint
    inner_learning_rate=inner_learning_rate,  # adversary lr
    outer_learning_rate=outer_learning_rate,  # model lr
    model_iterations_per_step=4,  # how many times to train on each step
    num_steps=500,  # number of epochs
    init_callback=init_callback,
    post_def_callback=post_def_callback,
    # num_steps=10,  # number of epochs
    max_batch_per_acc=2,  # max size of a minibatch
    only_train_lora=True,  # train using low rank adapters
    l2_regularization=0,  # coef for l2 weight regularization
    model_layers_module="base_model.model.model.layers",  # where the model layers are
    reinitialize_dev_optim=True,  # whether to reinitialize optimizer every lat step,
    add_completions_pgd=add_completions_pgd,  # Whether to add PGD over the completion tokens
    N_checkpoints=10,
    checkpoint_dir=f"latent-adversarial-training/models/{experiment_name}",
    # huggingface_folder=experiment_name,
    # huggingface_token=hf_access_token,
)


 
# # Run it


pgd_trainer.train(project_name="orpo_backdoor_240921")