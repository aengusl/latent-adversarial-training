# %%
import matplotlib.pyplot as plt
import os
import random
import itertools
import torch
import tqdm as tqdm
import wandb

from datasets import load_dataset
from dotenv import load_dotenv
from itertools import cycle
from peft import AutoPeftModelForCausalLM
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from latent_at import *

import argparse

load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

# %%
dtype = torch.bfloat16
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_access_token, torch_dtype=dtype).cuda()
# llama = AutoModelForCausalLM.from_pretrained("models/now", token=hf_access_token, torch_dtype=dtype).cuda()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.padding_side = "left"
device="cuda"

#%%

parser = argparse.ArgumentParser(description='Set variables from the command line.')

# Add arguments
parser.add_argument('--pca_type', type=str, help='Type of PCA to use', default=None)
parser.add_argument('--refresh_every', type=int, help='Refresh rate', default=10)
parser.add_argument('--num_pca_batches', type=int, help='Number of batches', default=5)
parser.add_argument('--data_type', type=str, help='Data type for LAT, can be "WHP", "SAQ"', default="WHP")
parser.add_argument('--epsilon', type=float, help='Epsilon for PGD', default=1.0)
parser.add_argument('--num_steps', type=int, help='Number of steps', default=100)
parser.add_argument('--pgd_iterations_per_step', type=int, help='Number of PGD iterations per step', default=16)
parser.add_argument('--model_iterations_per_step', type=int, help='Number of PGD iterations per step', default=4)
# parser.add_argument('--pgd_layer', type=int, help='PGD Layer', default=8)
parser.add_argument('--pgd_layer', type=int, nargs='+', help='PGD Layer(s)', default=[8])
parser.add_argument('--def_layer', type=int, help='PGD Layer', default=0)
parser.add_argument('--sft_type', type=str, help='Type of SFT to use', default="harmless")
parser.add_argument('--def_towards_coef', type=int, help='Coefficient for def towards loss', default=1)
parser.add_argument('--def_away_coef', type=int, help='Coefficient for def away loss', default=1)
parser.add_argument('--def_sft_coef', type=int, help='Coefficient for def sft loss', default=1)
parser.add_argument('--hf_run_id', type=int, help='Huggingface run id (for saving multiple models)', default=-1)

# Parse the arguments
args = parser.parse_args()

data_type = args.data_type
# %%
# load tasks/hp/data/generic_predictions.hf as huggingface dataset
import datasets
if data_type == "WHP":
    hp_generic_dataset = datasets.load_from_disk("tasks/hp/data/generic_predictions.hf")
    def add_label_indices(example):
        # don't want first label since first label isn't label for any part of sentence (label for prediction of first token)
        example['labels'] = example['labels'][1:]
        example['label_indices'] = list(range(len(example['tokens']) - 1))
        return example

    hp_generic_dataset = hp_generic_dataset.map(add_label_indices)

    hp_generic_dataset = process_pretokenized_dataset(
        tokenizer=tokenizer, 
        dataset=hp_generic_dataset, 
        prompt_column="tokens", 
        adv_labels_column=None, # adversary steers towards the prompt tokens
        # don't need to set adv_labels_column since adv_labels_column is None, handles by default
        def_labels_column="labels", # unlearned model steers towards generic labels
        def_labels_indices_column="label_indices", # indices of the generic labels, since labels of 
    )

    hp_dataloader = DataLoader(
        hp_generic_dataset,
        # hp_generic_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=PretokenizedLatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )


# %%
import pickle
with open('tasks/hp/data/msr_data/generic_hp_text.pkl', 'rb') as f:
    generic_hp_text = pickle.load(f)

# %%
# saq_easy
sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
use_tokenizer_template = True
custom_prompt_template = None
custom_completion_template = None

if data_type == "SAQ":
 
    hp_data = load_dataset("quirky-lats-at-mats/hp_saq_easy", split="train")
    from latent_at.lat_datasets import tokenized_behavior_dataset

    hp_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset="quirky-lats-at-mats/hp_saq_easy",
        prompt_column="question",
        adv_column="true_answer",
        def_column="false_answer",
        split="train",
        use_tokenizer_template=True,
        system_prompt=sys_prompt,
    )

    hp_dataloader = DataLoader(
        hp_dataset,
        # hp_generic_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )

if args.sft_type == "harmless":
    sft_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset="VH1213141516/benign_data_v1",
        adv_column=None,
        def_column="response",
        split="train",
        use_tokenizer_template=True,
        system_prompt=sys_prompt
    )

elif args.sft_type == "alpaca":
    sft_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset="abhayesian/alpaca-cleaned",
        adv_column=None,
        def_column="output",
        split="train",
        use_tokenizer_template=True,
        system_prompt=sys_prompt
    )    


elif args.sft_type == "ultrachat":
    sft_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset="abhayesian/ultrachat-first-response",
        adv_column=None,
        def_column="response",
        split="train_sft",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=sys_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template
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

elif args.sft_type == "wikitext":
    sft_dataset = process_generic_sft_dataset(
        tokenizer,
        dataset="wikitext",
        text_column="text",
        split="train",
        config="wikitext-103-v1",
        num_examples=100000,
    )

    sft_dataloader = DataLoader(
        sft_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )

elif args.sft_type == "pile":
    sft_dataset = process_generic_sft_dataset(
        tokenizer,
        dataset="JeanKaddour/minipile",
        text_column="text",
        split="train",
        num_examples=100000,
    )

    sft_dataloader = DataLoader(
        sft_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=1024
        )
    )

# %%
import pickle
import numpy as np
from datasets import Dataset


pca_type = args.pca_type

gen_batch_size = 32
pca_batch_size = 128
default_pca_kwargs = {
    "tokenizer": tokenizer,
    "device": device,
    "gen_batch_size": gen_batch_size,
    "pca_batch_size": pca_batch_size,
    "refresh_every": args.refresh_every,
    "num_batches": args.num_pca_batches,
}
if pca_type == "hp_verbatim_text":
    with open("tasks/hp/data/all_processed_sentences.pkl", "rb") as f:
        hp_verbatim_sentences = pickle.load(f)
    hp_verbatim_passages = [hp_verbatim_sentences[i:i+10] for i in range(0, len(hp_verbatim_sentences)-10)]
    random.shuffle(hp_verbatim_passages)

    passages_data = [" ".join(passage) for passage in hp_verbatim_passages]

    train_passages = passages_data[:10000]
    test_passages = passages_data[10000:11000]

    # transform passages_data into huggingface dataset
    from datasets import Dataset
    train_passages_dataset = Dataset.from_dict({"text": train_passages})
    test_passages_dataset = Dataset.from_dict({"text": test_passages})

    pca_kwargs = {
        "dataset": train_passages_dataset,
        "max_ctx_len": 512,
        "index_last_pos": 100,
    }
    # combine defualt_pca_kwargs with pca_kwargs, pca_kwargs has higher priority
    pca_kwargs = {**default_pca_kwargs, **pca_kwargs}

elif pca_type == "pile":
    pile_data = datasets.load_dataset("JeanKaddour/minipile", split="train[:10000]")
    pca_kwargs = {
        "dataset": pile_data,
        "max_ctx_len": 2048,
        "num_batches": 5,
    }
    pca_kwargs = {**default_pca_kwargs, **pca_kwargs}

elif pca_type == "hp_only_all" or pca_type == "hp_only_hp_indices" or pca_type == "genericized_diff_hp_indices" or pca_type == "genericized_diff_all":
    generic_indices_data = np.load("tasks/hp/data/msr_data/processed_data.npy", allow_pickle=True).item()
    original_sentences = [tokenizer.decode(generic_indices_data['original_tokens'][i]) for i in tqdm(range(len(generic_indices_data['original_tokens'])))]
    edited_sentences = [tokenizer.decode(generic_indices_data['edited_tokens'][i]) for i in tqdm(range(len(generic_indices_data['edited_tokens'])))]
    indices_1 = []
    indices_2 = []
    for i in range(len(generic_indices_data['index_map'])):
        indices_1.append([x[0] for x in generic_indices_data['index_map'][i]])
        indices_2.append([x[1] for x in generic_indices_data['index_map'][i]])
    
    filtered_indices_1 = []
    filtered_indices_2 = []

    for sentence_idx in tqdm(range(len(generic_indices_data['index_map']))):
        new_original_indices = []
        new_edited_indices = []
        original_tokens = generic_indices_data['original_tokens'][sentence_idx]
        edited_tokens = generic_indices_data['edited_tokens'][sentence_idx]
        original_indices = [x[0] for x in generic_indices_data['index_map'][sentence_idx]]
        edited_indices = [x[1] for x in generic_indices_data['index_map'][sentence_idx]]
        for index_idx in range(len(original_indices)):
            if original_tokens[original_indices[index_idx]] != edited_tokens[edited_indices[index_idx]]:
                new_original_indices.append(original_indices[index_idx])
                new_edited_indices.append(edited_indices[index_idx])

        filtered_indices_1.append(new_original_indices)
        filtered_indices_2.append(new_edited_indices)
    diff_dataset = Dataset.from_dict({"original_text": original_sentences, "edited_text": edited_sentences, 
                                  "original_tokens": generic_indices_data['original_tokens'], "edited_tokens": generic_indices_data['edited_tokens'],
                                  "original_indices": indices_1, "edited_indices": indices_2,
                                  "filtered_original_indices": filtered_indices_1, "filtered_edited_indices": filtered_indices_2})
    
    if pca_type == "hp_only_all":
        pca_kwargs = {
            "dataset": diff_dataset,
            "dataset_tokens_col": "original_tokens",
            "max_ctx_len": 2048,
        }
    elif pca_type == "hp_only_hp_indices":
        pca_kwargs = {
            "dataset": diff_dataset,
            "dataset_tokens_col": "original_tokens",
            "max_ctx_len": 2048,
            "indices_col": "filtered_original_indices",
        }
        
    elif pca_type == "genericized_diff_all":
        pca_kwargs = {
            "dataset": diff_dataset,
            "dataset_tokens_cols": ["original_tokens", "edited_tokens"],
            "indices_cols": ["original_indices", "edited_indices"],
            "max_ctx_len": 2048,
            "use_act_diff": True,
        }

    elif pca_type == "genericized_diff_hp_indices":
        pca_kwargs = {
            "dataset": diff_dataset,
            "dataset_tokens_cols": ["original_tokens", "edited_tokens"],
            "indices_cols": ["filtered_original_indices", "filtered_edited_indices"],
            "max_ctx_len": 2048,
            "use_act_diff": True,
        }

    pca_kwargs = {**default_pca_kwargs, **pca_kwargs}


else:
    pca_kwargs = None

print(pca_kwargs)

# %%
peft_config = LoraConfig(
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
)

llama = get_peft_model(llama, peft_config)

# %%



def eval_and_log(result, epoch):
    wandb.log(result)

epsilon = args.epsilon
# pgd_layer = args.pgd_layer
pgd_layer = args.pgd_layer
if len(pgd_layer) == 1:
    pgd_layer = pgd_layer[0]
def_layer = args.def_layer
if def_layer is None:
    def_layer = pgd_layer

print(f"{pgd_layer=}, {def_layer=}")

pgd_trainer = ProjectedGradLAT(
    model=llama,
    dataloader=hp_dataloader,
    sft_dataloader=sft_dataloader,
    def_loss_coefs={
        "toward": args.def_towards_coef,
        "away":   args.def_away_coef,
        "sft":    args.def_sft_coef,
    },
    pgd_layers=pgd_layer,
    model_layers=list(range(def_layer, llama.config.num_hidden_layers)),
    epsilon=args.epsilon,
    outer_learning_rate=5e-5,
    init_callback=eval_and_log,
    post_def_callback=eval_and_log,
    pgd_iterations_per_step=args.pgd_iterations_per_step,
    model_iterations_per_step=args.model_iterations_per_step,
    num_steps=args.num_steps,
    max_batch_per_acc=4,
    only_train_lora=True,
    model_layers_module="base_model.model.model.layers",
    pca_kwargs=pca_kwargs,
)
# get dict of args
additional_wandb_args = {}
for arg in vars(args):
    additional_wandb_args[arg] = getattr(args, arg)

pgd_trainer.train(project_name="HP-LAT", additional_wandb_kwargs=additional_wandb_args)

import time
# include time and date in model name
llama.save_pretrained(f"models/hp-lat-llama-{pca_type}-{epsilon=}-{pgd_layer=}-{args.sft_type}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")
# push to hub
if epsilon == 0 and args.pgd_iterations_per_step == 0 and pgd_layer == 0:
    # push WHP replication
    save_name = f"PhillipGuo/hp-whp_repl-towards{args.def_towards_coef}-away{args.def_away_coef}-sft{args.def_sft_coef}-{args.sft_type}"
else:
    if isinstance(pgd_layer, list):
        pgd_layer_str = "_".join(map(str, pgd_layer))
    else:
        pgd_layer_str = str(pgd_layer)

    pca_type = "PCA" if pca_type is not None else "No_PCA"
    if args.def_layer is not None:
        save_name = f"PhillipGuo/hp-lat-llama-{pca_type}-epsilon{epsilon}-pgd_layer{pgd_layer_str}-def_layer{args.def_layer}-{args.sft_type}"
    else:
        save_name = f"PhillipGuo/hp-lat-llama-{pca_type}-epsilon{epsilon}-pgd_layer{pgd_layer_str}_{args.sft_type}"
    
    if args.def_away_coef == 0 or args.def_towards_coef == 0 or args.def_sft_coef == 0:
        save_name += f"-towards{args.def_towards_coef}-away{args.def_away_coef}-sft{args.def_sft_coef}"
    # if args.def_layer is not None:
    #     save_name = f"PhillipGuo/hp-lat-llama-{pca_type}-epsilon{epsilon}-pgd_layer{pgd_layer}-def_layer{args.def_layer}-{args.sft_type}"
    # else:
    #     save_name = f"PhillipGuo/hp-lat-llama-{pca_type}-epsilon{epsilon}-pgd_layer{pgd_layer}_{args.sft_type}"

if args.hf_run_id != -1:
    save_name += f"-{args.hf_run_id}"

llama.push_to_hub(save_name, use_temp_dir=True)

# %%
