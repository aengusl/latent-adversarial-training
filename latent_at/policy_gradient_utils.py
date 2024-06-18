from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import random
from itertools import cycle
from dotenv import load_dotenv
import os
import tqdm.notebook as tqdm
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from tasks.inference_utils import generate_completions


from tasks.inference_utils import custom_generate
def get_classify_text_fn(model, tokenizer, toxic_index=0):
    """
    Get a reward model using a text classifier. (for instance works with unitary/toxic-bert)
    """
    # Tokenize the input texts
    def classify_text(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Move the input tensors to the model's device
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # print(outputs.logits.shape)
        # Process the model's output (logits)
        # predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = outputs.logits
        return predictions[:, toxic_index]
    return classify_text


def evaluate_reward(model, strs, tokenizer, reward_model_fn,
        max_gen_tokens=10,
        temperature=.7,
        reward_include_prompt=False, # whether to include the prompt during input to reward model):
        verbose=False,
        device="cuda"
):
    # # generate 10 tokens
    # tokenizer.padding_side = "left"
    # tokenized_inputs = tokenizer.batch_encode_plus(
    #     strs, return_tensors="pt", padding=True,
    # )
    # start_len = tokenized_inputs['input_ids'].shape[1]
    
    # tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}  # Move to model's device
    # try:
    #     outputs = model.generate(
    #         **tokenized_inputs,
    #         max_length=tokenized_inputs['input_ids'].shape[1] + max_gen_tokens,
    #         temperature=temperature,
    #         return_dict_in_generate=True,
    #         output_scores=True,
    #         pad_token_id=tokenizer.pad_token_id,
    #     )
    #     sequences = outputs.sequences
    #     scores = outputs.scores 

    # except Exception as e:

    #     print(f"Falling back to custom generation due to exception: {e}\nRunning model as a model inference function instead of a huggingface model.")

    #     custom_output = custom_generate(
    #         model_inference_fn=model,
    #         input=tokenized_inputs['input_ids'],
    #         num_new_tokens=max_gen_tokens,
    #         temperature=temperature,
    #         stop_tokens=[tokenizer.eos_token_id],
    #         verbose=False  # Set to True for progress bar
    #     )
    #     sequences = custom_output["sequences"]
    #     scores = custom_output["scores"]
    
    # if not reward_include_prompt:
    #     decode_sequences = [ids[start_len:] for ids in sequences]
    # else:
    #     decode_sequences = sequences

    # decoded_sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in decode_sequences]

    decoded_sentences, scores = generate_completions(model, strs, tokenizer, device, max_gen_tokens, temperature, return_decoded=True, include_prompt=reward_include_prompt)
    
    if verbose:
        print(strs)
        print(decoded_sentences)
    
    rewards = reward_model_fn(decoded_sentences)
    return rewards, decoded_sentences, scores


from tasks.inference_utils import batch_text_to_tokens, process_model_output
def evaluate_completion(model, tokenizer, prompt_tokens, completion_tokens, criterion="cross_entropy", device="cuda", return_tensor=False, return_logits=False, gamma=None, reduce="sum"):
    """
    prompt_tokens is list of batch_size lists of tokens (not necessary same length), with no padding. completion_tokens is same. ith element of prompt_tokens and completion_tokens should be from the same context, with completion_tokens coming right after prompt_tokens. Make sure completion_tokens does not have a start token.
    Evaluate the model's ability to complete the prompt by evaluating each token in completion_tokens. 
    """
    # tokenize the whole context
    if criterion == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    tokens = [torch.tensor(prompt_tokens[i] + completion_tokens[i]) for i in range(len(prompt_tokens))]
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    
    # losses = torch.zeros(len(tokens)).to(self.device)
    losses = []
    # pad tokens
    logits = process_model_output(model(tokens.cuda()))

    all_logits = []
    
    for i in range(len(tokens)): # for each sentence in the batch
        # print(f"{len(prompt_tokens[i])=}")
        # print(f"{len(prompt_tokens[i])+len(completion_tokens[i])-1=}")
        completion_start_index = len(prompt_tokens[i])
        completion_end_index = len(prompt_tokens[i])+len(completion_tokens[i])-1
        loss_logits = logits[i, completion_start_index:completion_end_index,]
        target_labels = torch.tensor(completion_tokens[i][1:]).to(device)

        print(f"Evaluating loss on logits at tokens {tokens[i][completion_start_index:completion_end_index]}\nlabels {target_labels}")

        if return_logits:
            all_logits.append(loss_logits)
        else:
            loss = criterion(loss_logits, target_labels)
            if gamma is not None:
                # make tensor of discount weights, discounting early token losses more
                discount_weights = torch.tensor([gamma**(len(loss) - i - 1) for i in range(len(loss))]).to(device)
                print(f"{discount_weights=}")
                loss = loss * discount_weights
                if reduce == "sum":
                    loss = loss.sum()
                elif reduce == "mean":
                    loss = loss.mean()
            losses.append(loss)

    if return_logits:
        return all_logits

    if return_tensor:
        return torch.stack(losses)
    return losses

def policy_grad_loss(model, strs, tokenizer, reward_model_fn, max_gen_tokens=10, temperature=.7, reward_include_prompt=False, verbose=False):
    with torch.no_grad():
        rewards, decoded_sentences, _ = evaluate_reward(
            model, strs, tokenizer, reward_model_fn, max_gen_tokens, temperature, reward_include_prompt
        )
    
    # now, calculate cross entropy on decoded sentences portion of whole text by appending decoded_sentences to strs
    if reward_include_prompt:
        # want only the ending part not including the original strs, but decoded_sentences includes the original strs
        gen_strings = []
        for i in range(len(strs)):
            orig_str_len = len(strs[i])+1
            gen_strings.append(decoded_sentences[i][orig_str_len:])
    else:
        gen_strings = decoded_sentences
    
    prompt_tokens = tokenizer(strs).input_ids
    completion_tokens_temp = tokenizer(gen_strings).input_ids
    completion_tokens = []
    for completion in completion_tokens_temp:
        if completion[0] == tokenizer.bos_token_id:
            completion_tokens.append(completion[1:])
        else:
            completion_tokens.append(completion)
    
    if verbose:
        print(prompt_tokens)
        print(completion_tokens)
    losses = evaluate_completion(model, tokenizer, prompt_tokens, completion_tokens, criterion="cross_entropy")
    reward_weighted_losses = [rewards[i] * loss for i, loss in enumerate(losses)]
    return reward_weighted_losses

