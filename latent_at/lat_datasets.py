import os
import json
from typing import List, Optional, Union
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")


class LatentAdversarialTrainingDataset(Dataset):

    def __init__(self, hf_dataset):
        self.adv_tokens = hf_dataset["adv_tokens"]
        self.def_tokens = hf_dataset["def_tokens"]
        self.prompt_tokens = hf_dataset["prompt_tokens"]

        try:
            if "adv_strs" in hf_dataset.column_names:
                self.adv_strs = hf_dataset["adv_strs"]
                self.def_strs = hf_dataset["def_strs"]
                self.prompt_strs = hf_dataset["prompt_strs"]
        except:
            pass

        self.prompt_lengths = torch.tensor([len(x) for x in self.prompt_tokens])
        self.adv_lengths = torch.tensor([len(x) for x in self.adv_tokens])
        self.def_lengths = torch.tensor([len(x) for x in self.def_tokens])
        self.length = self.adv_lengths.shape[0]

        try:
            if "adv_labels" in hf_dataset.column_names:
                self.adv_labels = hf_dataset["adv_labels"]
                self.def_labels = hf_dataset["def_labels"]
                self.adv_indices = hf_dataset["adv_indices"]
                self.def_indices = hf_dataset["def_indices"]
        except:
            pass
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # if self.adv_strs exists, return it
        return_dict = {
            "adv_tokens": self.adv_tokens[idx],
            "def_tokens": self.def_tokens[idx],
            "prompt_tokens": self.prompt_tokens[idx],
            "adv_length": self.adv_lengths[idx],
            "def_length": self.def_lengths[idx],
            "prompt_length": self.prompt_lengths[idx],
        }
        if hasattr(self, 'adv_strs'):
            return_dict["adv_strs"] = self.adv_strs[idx]
            return_dict["def_strs"] = self.def_strs[idx]
            return_dict["prompt_strs"] = self.prompt_strs[idx]
        if hasattr(self, 'adv_labels'):
            return_dict["adv_labels"] = self.adv_labels[idx]
            return_dict["def_labels"] = self.def_labels[idx]
            return_dict["adv_indices"] = self.adv_indices[idx]
            return_dict["def_indices"] = self.def_indices[idx]
        return return_dict
                # "adv_strs": self.adv_strs[idx],
                # "def_strs": self.def_strs[idx],
                # "prompt_strs": self.prompt_strs[idx]


class LatentAdversarialTrainingDataCollator:
    def __init__(self, pad_token_id, truncate_length=None):
        assert pad_token_id is not None, "pad_token_id must be specified"

        self.pad_token_id = pad_token_id
        self.truncate_length = truncate_length

    def __call__(self, batch):
        B = len(batch)

        prompt_lengths = []
        adv_prompt_lengths = []
        def_prompt_lengths = []

        for i in range(B):
            prompt_lengths.append(batch[i]["prompt_length"])
            adv_prompt_lengths.append(batch[i]["prompt_length"] + batch[i]["adv_length"])
            def_prompt_lengths.append(batch[i]["prompt_length"] + batch[i]["def_length"])
        
        pad_length = max(adv_prompt_lengths + def_prompt_lengths)

        adv_prompt_tokens = torch.zeros(B, pad_length, dtype=torch.long)
        def_prompt_tokens = torch.zeros(B, pad_length, dtype=torch.long)

        prompt_mask = torch.zeros(B, pad_length, dtype=torch.bool)
        adv_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)
        def_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)

        for i in range(B):
            adv_prompt_tokens[i] = torch.tensor(batch[i]["prompt_tokens"] + batch[i]["adv_tokens"] + [self.pad_token_id] * (pad_length - adv_prompt_lengths[i]))
            def_prompt_tokens[i] = torch.tensor(batch[i]["prompt_tokens"] + batch[i]["def_tokens"] + [self.pad_token_id] * (pad_length - def_prompt_lengths[i]))

            prompt_mask[i, :prompt_lengths[i]] = True
            adv_labels_mask[i, prompt_lengths[i]:adv_prompt_lengths[i]] = True
            def_labels_mask[i, prompt_lengths[i]:def_prompt_lengths[i]] = True

        if self.truncate_length is not None:
            if any([prompt_length > self.truncate_length for prompt_length in prompt_lengths]):
                print(f"WARNING: Prompt length (at least one of {prompt_lengths}) is less than truncate length ({self.truncate_length})")

            adv_prompt_tokens = adv_prompt_tokens[:, :self.truncate_length]
            def_prompt_tokens = def_prompt_tokens[:, :self.truncate_length]
            prompt_mask = prompt_mask[:, :self.truncate_length]
            adv_labels_mask = adv_labels_mask[:, :self.truncate_length]
            def_labels_mask = def_labels_mask[:, :self.truncate_length]

        
        if "adv_strs" in batch[0]:
            return {
                "adv_tokens": adv_prompt_tokens,
                "def_tokens": def_prompt_tokens,
                "prompt_mask": prompt_mask,
                "adv_labels_mask": adv_labels_mask,
                "def_labels_mask": def_labels_mask,
                "adv_strs": [x["adv_strs"] for x in batch],
                "def_strs": [x["def_strs"] for x in batch],
                "prompt_strs": [x["prompt_strs"] for x in batch]
            }
        return {
            "adv_tokens": adv_prompt_tokens,
            "def_tokens": def_prompt_tokens,
            "prompt_mask": prompt_mask,
            "adv_labels_mask": adv_labels_mask,
            "def_labels_mask": def_labels_mask,
            # "adv_strs": [x["adv_strs"] for x in batch],
            # "def_strs": [x["def_strs"] for x in batch],
            # "prompt_strs": [x["prompt_strs"] for x in batch]
        }


def apply_chat_formatting(
    tokenizer,
    prompt,
    def_completion,
    adv_completion,
    use_tokenizer_template,
    system_prompt,
    custom_prompt_template,
    custom_completion_template
):
    if use_tokenizer_template:
        if system_prompt is not None:
            prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        else:
            prompt_messages = [{"role": "user", "content": prompt}]
        
        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    else:
        if system_prompt is not None:
            prompt_str = custom_prompt_template.format(system_prompt=system_prompt, prompt=prompt)
        else:
            prompt_str = prompt
    
    if custom_completion_template is not None:
        adv_str = custom_completion_template.format(completion=adv_completion)
        def_str = custom_completion_template.format(completion=def_completion)
    else:
        adv_str = adv_completion
        def_str = def_completion
    
    return prompt_str, adv_str, def_str


def process_generic_chat_dataset(
    tokenizer,
    dataset="Baidicoot/comic_villain_completions",
    prompt_column="prompt",
    adv_column="adv_completion",
    def_column="clean_completion",
    use_tokenizer_template=True,
    custom_prompt_template=None,
    custom_completion_template=None,
    system_prompt=None,
    system_prompt_column=None,
    filter_len=None,
    num_adv_words=None,
    map_fn=None,
    add_eos_token=False,
    **dataset_kwargs,
):
    # loader for generic datasets of the form (prompt, positive_completion, negative_completion)
    assert not (system_prompt is not None and system_prompt_column is not None), "Only one of system_prompt and system_prompt_column can be specified"

    dataset = load_dataset(dataset, **dataset_kwargs)

    if prompt_column != "prompt":
        dataset = dataset.rename_column(prompt_column, "prompt")

    if adv_column != "adv_completion":
        if adv_column is None:
             dataset = dataset.map(lambda x: {"adv_completion": "not available"})
        else:
            dataset = dataset.rename_column(adv_column, "adv_completion")
    
    if def_column != "def_completion":
        dataset = dataset.rename_column(def_column, "def_completion")

    if system_prompt_column is not None:
        dataset = dataset.rename_column(system_prompt_column, "system_prompt")
    
    if map_fn is not None:
        dataset = dataset.map(map_fn, batched=True)
    
    def preprocess_example_batch(examples):
        for i in range(len(examples["prompt"])):
            if system_prompt_column is not None:
                _system_prompt = examples["system_prompt"][i]
            elif system_prompt is not None:
                _system_prompt = system_prompt
            else:
                _system_prompt = None
                
            prompt, adv_completion, def_completion= apply_chat_formatting(
                tokenizer=tokenizer,
                prompt=examples["prompt"][i],
                def_completion=examples["def_completion"][i],
                adv_completion=examples["adv_completion"][i],
                use_tokenizer_template=use_tokenizer_template,
                system_prompt=_system_prompt,
                custom_prompt_template=custom_prompt_template,
                custom_completion_template=custom_completion_template
            )
            
            examples["prompt"][i] = prompt
            
            if num_adv_words is None:
                examples["adv_completion"][i] = adv_completion
            else:
                examples["adv_completion"][i] = " ".join(adv_completion.split(" ")[:num_adv_words])

            if add_eos_token:
                examples["def_completion"][i] = def_completion + tokenizer.eos_token
            else:
                examples["def_completion"][i] = def_completion
        
        return examples

    dataset = dataset.map(
        preprocess_example_batch,
        batched=True,
        remove_columns=set(dataset.column_names) - {"prompt", "adv_completion", "def_completion"}
    )

    def remove_duplicate_bos_batched(batch_of_sequences, bos_token):
        def process_sequence(sequence):
            # Find the index of the last BOS token at the start of the sequence
            last_bos_index = 0
            for i, token in enumerate(sequence):
                if token != bos_token:
                    break
                last_bos_index = i

            # Return a single BOS token followed by the rest of the sequence
            return [bos_token] + sequence[last_bos_index + 1:]

        return [process_sequence(seq) for seq in batch_of_sequences]

    def tokenize_batch(examples):
        examples["prompt_tokens"] = remove_duplicate_bos_batched(
            tokenizer(examples["prompt"], add_special_tokens=True).input_ids,
            tokenizer.bos_token_id
        )
        examples["adv_tokens"] = tokenizer(examples["adv_completion"], add_special_tokens=False).input_ids
        examples["def_tokens"] = tokenizer(examples["def_completion"], add_special_tokens=False).input_ids
        return examples
    
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns={"prompt", "adv_completion", "def_completion"}
    )

    if filter_len is not None:
        start_len = len(dataset)
        dataset = dataset.filter(
            lambda x: len(x["prompt_tokens"]) + max(len(x["adv_tokens"]), len(x["def_tokens"])) <= filter_len
        )
        end_len = len(dataset)
        print(f"Filtered out {(start_len - end_len) / start_len * 100:.2f}% of the dataset")

    return LatentAdversarialTrainingDataset(dataset)


def process_generic_sft_dataset(
        tokenizer,
        dataset="wikitext",
        text_column="text",
        split="train",
        config=None,
        num_examples=100000,
        **kwargs
):

    def tokenize_batch(examples):
        examples["prompt_tokens"] = []
        examples["adv_tokens"] = []
        examples["def_tokens"] = []
        for i in range(len(examples[text_column])):
            text = examples[text_column][i]
            tokenized = tokenizer(text, add_special_tokens=False).input_ids
            examples["prompt_tokens"].append([])
            examples["adv_tokens"].append(tokenized)
            examples["def_tokens"].append(tokenized)
        return examples

    if config is not None:
        dataset = load_dataset(dataset, config, split=split, **kwargs)
    else:
        dataset = load_dataset(dataset, split=split, **kwargs)

    if num_examples is not None:
        dataset = dataset.take(num_examples)  # think this works for streaming
    # filter dataset for empty text rows
    dataset = dataset.filter(lambda x: len(x[text_column]) > 0)

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
    )

    dataset = LatentAdversarialTrainingDataset(dataset)
    return dataset

def tokenized_behavior_dataset(
    behaviors_list,
    tokenizer,
    use_tokenizer_template=True,
    custom_prompt_template=None,
    custom_completion_template=None,
    system_prompt=None,
    good_refuse=True,
    bad_refuse=False,
):

    final_dataset = {"prompt_tokens":[], "adv_tokens":[], "def_tokens":[], "prompt_strs": [], "adv_strs": [], "def_strs": []}
    for entry in behaviors_list:
        
        prompt, target, refusal = apply_chat_formatting(
            tokenizer=tokenizer,
            prompt=entry["behavior"],
            def_completion=entry["refusal"],
            adv_completion=entry["target"],
            use_tokenizer_template=use_tokenizer_template,
            system_prompt=system_prompt,
            custom_prompt_template=custom_completion_template,
            custom_completion_template=custom_completion_template
        )

        prompt_tokens = tokenizer(prompt).input_ids
        target_tokens = tokenizer(target, add_special_tokens=False).input_ids
        refusal_tokens = tokenizer(refusal, add_special_tokens=False).input_ids
                
        # remove bos token
        if target_tokens[0] == tokenizer.bos_token_id:
            target_tokens = target_tokens[1:]
        if refusal_tokens[0] == tokenizer.bos_token_id:
            refusal_tokens = refusal_tokens[1:]
        
        final_dataset["prompt_tokens"].append(prompt_tokens)
        final_dataset["prompt_strs"].append(prompt)
        if good_refuse:
            final_dataset["def_tokens"].append(refusal_tokens)    
            final_dataset["def_strs"].append(refusal)        
        else:
            final_dataset["def_tokens"].append(target_tokens)
            final_dataset["def_strs"].append(target)

        if bad_refuse:
            final_dataset["adv_tokens"].append(refusal_tokens)
            final_dataset["adv_strs"].append(refusal)
        else:
            final_dataset["adv_tokens"].append(target_tokens)
            final_dataset["adv_strs"].append(target)
        
    return LatentAdversarialTrainingDataset(final_dataset)


def process_pretokenized_dataset(
    tokenizer,
    dataset,
    prompt_column="prompt_tokens",
    adv_labels_column=None, #"adv_labels",
    adv_labels_indices_column=None,
    def_labels_column=None, #"def_labels",
    def_labels_indices_column=None,
):
    """
    If adv_labels_column == None, assume prompt_tokens should also be used as adv_labels. 

    Rows of adv_labels_indices_column gives a **list** of indices that the labels correspond to. For indexing:
        For one row, assume next-token-prediction at prompt_tokens[adv_labels_indices[i]] == adv_labels[i]

    Does not support system prompt, add system prompt before tokens
    """

    if prompt_column != "prompt_tokens":
        dataset = dataset.rename_column(prompt_column, "prompt_tokens")
    
    # add adv_tokens and def_tokens, which are just prompt_tokens
    dataset = dataset.map(lambda examples: {"adv_tokens": examples["prompt_tokens"], "def_tokens": examples["prompt_tokens"]}, batched=True)

    # adv and def shouldn't both be None
    assert adv_labels_column is not None or def_labels_column is not None, "At least one of adv_labels_column or def_labels_column should almost certainly be specified, error for now"

    def create_default_indices(examples, adv=True):
        # by default, labels are the next token in the prompt_tokens
        if adv:
            examples["adv_labels"] = [None] * len(examples["prompt_tokens"])
            examples["adv_indices"] = [None] * len(examples["prompt_tokens"])

        for i in range(len(examples["prompt_tokens"])):
            labels = examples["prompt_tokens"][i]
            labels = labels[1:]
            indices = list(range(len(labels))) # next_token_prediction at last token is not a label
            if adv:
                examples["adv_labels"][i] = labels
                examples["adv_indices"][i] = indices
            else:
                examples["def_labels"][i] = labels
                examples["def_indices"][i] = indices
        return examples

    if adv_labels_column is None:
        assert adv_labels_indices_column is None, f"If adv_labels_column is None, adv_labels_indices_column must also be None, instead got {adv_labels_indices_column}"
        dataset = dataset.map(lambda examples: create_default_indices(examples, adv=True), batched=True)

    else:
        if adv_labels_column != "adv_labels":
            dataset = dataset.rename_column(adv_labels_column, "adv_labels")
        assert adv_labels_indices_column is not None, f"If adv_labels_column is not None, adv_labels_indices_column must also be specified, instead got {adv_labels_indices_column}"
        if adv_labels_indices_column != "adv_indices":
            dataset = dataset.rename_column(adv_labels_indices_column, "adv_indices")
    
    if def_labels_column is None:
        assert def_labels_indices_column is None, f"If def_labels_column is None, def_labels_indices_column must also be None, instead got {def_labels_indices_column}"
        dataset = dataset.map(lambda examples: create_default_indices(examples, adv=False), batched=True)

    else:
        if def_labels_column != "def_labels":
            dataset = dataset.rename_column(def_labels_column, "def_labels")
        assert def_labels_indices_column is not None, f"If def_labels_column is not None, def_labels_indices_column must also be specified, instead got {def_labels_indices_column}"
        if def_labels_indices_column != "def_indices":
            dataset = dataset.rename_column(def_labels_indices_column, "def_indices")

    print("Completed adding/renaming columns, performing checks")
    # do final checks
    def check_labels_lengths(examples):
    # Assuming examples is a batch of examples
        for i in range(len(examples["adv_labels"])):
            assert len(examples["adv_labels"][i]) == len(examples["adv_indices"][i]), f"Length of adv_labels and adv_indices should be the same, got {len(examples['adv_labels'][i])} and {len(examples['adv_indices'][i])}"
            assert len(examples["def_labels"][i]) == len(examples["def_indices"][i]), f"Length of def_labels and def_indices should be the same, got {len(examples['def_labels'][i])} and {len(examples['def_indices'][i])}"
        # Return the examples unmodified, as we're only performing checks
        return examples

    # Apply the checks to the dataset using map
    dataset.map(check_labels_lengths, batched=True)

    return LatentAdversarialTrainingDataset(dataset) # lengths will look weird


def load_targeted_wmdp_data(
        retain_corpora: List[str]=['bio-forget-corpus', 'cyber-forget-corpus'],
        forget_corpora: List[str]=['wikitext', 'wikitext'])-> list[Dataset, Dataset]:
    """
    Forget and retain datasets differ. Load forget and retain then recombine into columns
    'adv_tokens' and 'def_tokens' in new Huggingface Dataset object.

    Supports bio/cyber WMDP retain corpora and WikiText (paired with both bio and cyber unlearn)
    as options for retain dataset. See config line 'retain_corpora'.
    """

    # Load and rename datasets for 'forget' corpora
    # Only bio forget needs to be locally loaded
    hf_location = "cais/wmdp-corpora"
    forget_data, retain_data, all_data = [], [], []

    for d in forget_corpora:
        if d == "bio-forget-corpus": # wmdp bio must be downloaded
            dataset_path = f"data/{d}.jsonl"
            forget_dataset = load_dataset('json', data_files=dataset_path, split='train')
            forget_dataset = forget_dataset.rename_column('text', 'adv_tokens')
            forget_data.append(forget_dataset)
        elif d == "cyber-forget-corpus":
            forget_dataset = load_dataset(hf_location, name=d, split='train')
            forget_dataset = forget_dataset.rename_column('text', 'adv_tokens')
        else:
            raise NotImplementedError
        forget_data.append(forget_dataset)

    for d in retain_corpora:
        if d in ['bio-retain-corpus', 'cyber-retain-corpus']:
            retain_dataset = load_dataset(hf_location, name=d, split='train')
            retain_dataset = retain_dataset.rename_column('text', 'def_tokens')
        elif d in ['wikitext']:
            retain_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            retain_dataset = retain_dataset.rename_column('text', 'def_tokens')
        else:
            raise NotImplementedError
        retain_data.append(retain_dataset)

    def merge_rows(example1, example2):
        return {'adv_tokens': example1['adv_tokens'], 'def_tokens': example2['def_tokens']}

    for fd, rd in zip(forget_data, retain_data):
        min_length = min(len(fd), len(rd))
        dset = fd.select(range(min_length)).map(
            lambda x,
                   idx: merge_rows(x, rd[idx]),
            with_indices=True,
        )
        if 'title' in dset:  # if bio
            dset = dset.remove_columns(['title', 'abstract', 'doi'])
        all_data.append(dset)

    return all_data


def make_targeted_wmdp_dataloader(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    lat_batch_size: int=4,
    data_truncate_length: int=600,
) -> tuple[DataLoader, DataLoader]:

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=lat_batch_size,
        collate_fn=WMDPLATTargetedDataCollator(tokenizer, truncate_length=data_truncate_length),
        drop_last=True,
    )
    return dataloader


def make_untargeted_wmdp_dataloaders(
        data: Union[list[str], list[list[str]]],
        tokenizer: AutoTokenizer,
        sft: Optional[bool] = True,
        sft_batch_size: int=4,
        lat_batch_size: int=4,
        data_truncate_length: int=600,
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    """
    Don't use for anything except WMDP unlearn corpora exactly as loaded with tuple list [bio, cyber].
    Used for two things: creating SFT dataloaders, and creating WMDP dataloaders in the case where defence and attacker both train on the WMDP unlearn corpora.

    In the paper, by default, we use targeted, where the defence trains toward WikiText and the attack trains toward WMDP unlearn corpora. The away losses are gradient ascent on these same datasets but swapped between attack and defence.

    Args:
        config: OmegaConf object created from yaml file.
        data: Each list this contains will be one of the bio/cyber datasets.
        sft: If True, data list is two copies of the same dataset. This will only be used to generate supervised fine tuning dataloader for SFT portion of R2D2 loss in LAT.
    Returns:
        Dataloaders.
    """
    if sft:
        return DataLoader(
            data,
            shuffle=True,
            batch_size=sft_batch_size,
            collate_fn=WMDPLATDataCollator(tokenizer, truncate_length=data_truncate_length),
            drop_last=True,
        )

    wmdp_bio_dataloader = DataLoader(
        data[0],
        shuffle=True,
        batch_size=lat_batch_size,
        collate_fn=WMDPLATDataCollator(tokenizer, truncate_length=data_truncate_length),
        drop_last=True,
    )
    wmdp_cyber_dataloader = DataLoader(
        data[1],
        shuffle=True,
        batch_size=lat_batch_size,
        collate_fn=WMDPLATDataCollator(tokenizer, truncate_length=data_truncate_length),
        drop_last=True,
    )
    return wmdp_bio_dataloader, wmdp_cyber_dataloader


def load_sft_dataset(sft_corpora: str='alpaca') -> list[str]:
    """Works for wikitext and alpaca."""
    if sft_corpora == "wikitext":
        raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    elif sft_corpora == "alpaca":
        raw_data = load_dataset("tatsu-lab/alpaca", "default", split="train")
    else:
        raise NotImplementedError
    data = []
    for x in raw_data:
        data.append(str(x['text']))
    return data


class PretokenizedLatentAdversarialTrainingDataCollator:
    def __init__(self, pad_token_id, truncate_length=None):
        assert pad_token_id is not None, "pad_token_id must be specified"

        self.pad_token_id = pad_token_id
        self.truncate_length = truncate_length

    def __call__(self, batch):
        B = len(batch)

        prompt_lengths = []
        # adv_prompt_lengths and def_prompt_lengths are equal to prompt_lengths

        for i in range(B):
            prompt_lengths.append(batch[i]["prompt_length"])
        
        pad_length = max(prompt_lengths)

        # adv_prompt_tokens = torch.zeros(B, pad_length, dtype=torch.long)
        # def_prompt_tokens = torch.zeros(B, pad_length, dtype=torch.long)
        
        # add 1 to mask lengths to allow label at last token
        adv_prompt_tokens = torch.zeros(B, pad_length+1, dtype=torch.long) 
        def_prompt_tokens = torch.zeros(B, pad_length+1, dtype=torch.long)
        # both are same, just prompt

        prompt_mask = torch.zeros(B, pad_length+1, dtype=torch.bool) 
        adv_labels_mask = torch.zeros(B, pad_length+1, dtype=torch.bool)
        def_labels_mask = torch.zeros(B, pad_length+1, dtype=torch.bool)

        for i in range(B):
            adv_prompt_tokens[i] = torch.tensor(batch[i]["prompt_tokens"] + [self.pad_token_id] * (pad_length - prompt_lengths[i]+1))
            def_prompt_tokens[i] = torch.tensor(batch[i]["prompt_tokens"] + [self.pad_token_id] * (pad_length - prompt_lengths[i]+1))
            # set prompt_mask to True everywhere except for the padding
            prompt_mask[i, :prompt_lengths[i]] = True

            for adv_index in batch[i]["adv_indices"]:
                # assert adv_index+1 < prompt_lengths[i], f"adv_index {adv_index} is within 1 of prompt_length {prompt_lengths[i]}"
                if adv_index+1 >= prompt_lengths[i]:
                    print(f"adv_index {adv_index} is within 1 of prompt_length {prompt_lengths[i]}, will not check labels for this index")
                    continue
                adv_labels_mask[i, adv_index + 1] = True # + 1 because indexing logits by final_logits = logits[:, :-1][towards_labels_mask[:, 1:]], need to shift by 1
            
            for def_index in batch[i]["def_indices"]:
                assert def_index+1 < prompt_lengths[i], f"def_index {def_index} is within 1 of prompt_length {prompt_lengths[i]}"
                def_labels_mask[i, def_index + 1] = True
            
        if self.truncate_length is not None:
            adv_prompt_tokens = adv_prompt_tokens[:, :self.truncate_length]
            def_prompt_tokens = def_prompt_tokens[:, :self.truncate_length]
            prompt_mask = prompt_mask[:, :self.truncate_length]
            adv_labels_mask = adv_labels_mask[:, :self.truncate_length]
            def_labels_mask = def_labels_mask[:, :self.truncate_length]

        return_dict = {
            "adv_tokens": adv_prompt_tokens,
            "def_tokens": def_prompt_tokens,
            "prompt_mask": prompt_mask,
            "adv_labels_mask": adv_labels_mask,
            "def_labels_mask": def_labels_mask,

            "adv_labels": [x["adv_labels"] for x in batch],
            "def_labels": [x["def_labels"] for x in batch],
            # "adv_indices": [x["adv_indices"] for x in batch],
            # "def_indices": [x["def_indices"] for x in batch],
        }

        return return_dict


class WMDPLATTargetedDataCollator:
    """
    Targeted version of below class, which returns *different* adv_labels and def_labels
    using wmdp retain and unlearn corpora.
    Specifically designed to WMDP corpora data, working with data loading methods from jsonl in wmdp/cut/utils.py,
    with batching removed as batching is done here instead.
    This class is not used for SFT.
    """

    def __init__(self, tokenizer, truncate_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.eos_token_id
        self.truncate_length = truncate_length

    def __call__(self, batch: List[str]):
        B = len(batch)
        tokenized_def_inputs = [self.tokenizer(example["def_tokens"])["input_ids"] for example in batch]
        tokenized_adv_inputs = [self.tokenizer(example["adv_tokens"])["input_ids"] for example in batch]
        def_lengths = [len(x) for x in tokenized_def_inputs]
        adv_lengths = [len(x) for x in tokenized_adv_inputs]
        pad_length = max(max(adv_lengths), max(def_lengths))

        def_tokens = torch.zeros(B, pad_length, dtype=torch.long)
        adv_tokens = torch.zeros(B, pad_length, dtype=torch.long)
        def_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)
        adv_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)

        for i, (def_inputs, adv_inputs) in enumerate(zip(tokenized_def_inputs, tokenized_adv_inputs)):
            def_tokens[i] = torch.tensor(def_inputs + [self.pad_token_id] * (pad_length - def_lengths[i]),
                                         dtype=torch.long)
            adv_tokens[i] = torch.tensor(adv_inputs + [self.pad_token_id] * (pad_length - adv_lengths[i]),
                                         dtype=torch.long)
            def_labels_mask[i, :def_lengths[i]] = True
            adv_labels_mask[i, :adv_lengths[i]] = True

        if self.truncate_length is not None:
            def_tokens = def_tokens[:, :self.truncate_length]
            adv_tokens = adv_tokens[:, :self.truncate_length]
            def_labels_mask = def_labels_mask[:, :self.truncate_length]
            adv_labels_mask = adv_labels_mask[:, :self.truncate_length]

        return {
            "def_tokens": def_tokens,
            "adv_tokens": adv_tokens,
            "def_labels_mask": def_labels_mask,
            "adv_labels_mask": adv_labels_mask,
        }


class WMDPLATDataCollator:
    """
    Specifically designed to WMDP corpora data, working with data loading methods from jsonl in wmdp/cut/utils.py,
    with batching removed as batching is done here instead.

    Note adv_labels == def_labels because we just do a 1-p loss for the defence on the 'bad corpora'.
    This class is used for both SFT and WMDP unlearn corpora.
    For SFT, it suffices to have the labels mask be created as per usual.
    """

    def __init__(self, tokenizer, truncate_length):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.eos_token_id
        self.truncate_length = truncate_length

    def __call__(self, batch: List[str]):
        B = len(batch)
        tokenized_inputs = [self.tokenizer(example)["input_ids"] for example in batch]
        lengths = [len(example) for example in tokenized_inputs]
        pad_length = max(lengths)

        tokens = torch.zeros(B, pad_length, dtype=torch.long)
        adv_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)
        def_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)

        for i, example in enumerate(tokenized_inputs):
            l = lengths[i]
            tokens[i] = torch.tensor(example + [self.pad_token_id] * (pad_length - l), dtype=torch.long)
            adv_labels_mask[i, :l] = True
            def_labels_mask[i, :l] = True

        if self.truncate_length is not None:
            tokens = tokens[:, :self.truncate_length]
            def_labels_mask = def_labels_mask[:, :self.truncate_length]
            adv_labels_mask = adv_labels_mask[:, :self.truncate_length]

        return {
            "tokens": tokens,
            "def_labels_mask": def_labels_mask,
            "adv_labels_mask": adv_labels_mask,
        }
