import torch

from typing import List, Tuple, Union

from latent_at.laa import add_hooks
from tqdm.auto import tqdm
import time

class BatchedPCA:
    def __init__(self, n_dims, device):
        super().__init__()
        self.n_dims = n_dims
        self.device = device

        self.cov = torch.zeros((n_dims, n_dims), device=device)
        self.mean = torch.zeros((n_dims,), device=device)
        self.n_samples = 0

    def get_mean(self):
        return self.mean

    def train_batch(self, activations):
        # activations: (batch_size, n_dims)
        batch_size = activations.shape[0]
        corrected = activations - self.mean.unsqueeze(0)
        new_mean = self.mean + torch.mean(corrected, dim=0) * batch_size / (self.n_samples + batch_size)
        cov_update = torch.einsum("bi,bj->bij", corrected, activations - new_mean.unsqueeze(0)).mean(dim=0)
        self.cov = self.cov * (self.n_samples / (self.n_samples + batch_size)) + cov_update * batch_size / (
            self.n_samples + batch_size
        )
        self.mean = new_mean
        self.n_samples += batch_size

    def get_pca(self):
        cov_symm = (self.cov + self.cov.T) / 2
        eigvals, eigvecs = torch.linalg.eigh(cov_symm)
        return eigvals, eigvecs

    def get_centering_transform(self):
        eigvals, eigvecs = self.get_pca()

        # assert torch.all(eigvals > 0), "Covariance matrix is not positive definite"
        # clamp to avoid numerical issues
        eigvals = torch.clamp(eigvals, min=1e-6)

        scaling = 1 / torch.sqrt(eigvals)

        assert torch.all(~torch.isnan(scaling)), "Scaling has NaNs"

        return self.get_mean(), eigvecs, scaling

    def get_projections(self):
        mean, eigvecs, scaling = self.get_centering_transform()

        # print(eigvecs.shape)
        # print(f"{scaling=}")

        proj = torch.einsum("dn,d->dn", eigvecs.T, scaling)
        unproj = torch.einsum("nd,d->nd", eigvecs, 1 / scaling)

        return proj, unproj

class ActivationCache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_cache = None

    def forward(self, x):
        self.activation_cache = x.detach().clone()#.cpu()
        return x
    
    def get_cache(self):
        return self.activation_cache
    
    def clear_cache(self):
        self.activation_cache = []

def tokenize_dataset_for_pca(
    dataset,
    tokenizer,
    batch_size,
    dataset_text_col="text",
    dataset_tokens_col=None,
    max_ctx_len=2048,
    indices_col=None
):
    if dataset_tokens_col is None:
        dataset = dataset.map(
            lambda x: tokenizer(x[dataset_text_col], truncation=True),
            batched=True,
            # remove_columns=set(dataset.column_names) - {"input_ids"}
        )
    else:
        dataset = dataset.map(
            lambda x: {"input_ids": x[dataset_tokens_col], "attention_mask": [[1] * len(input_ids) for input_ids in x[dataset_tokens_col]]},
            batched=True,
        )

    def dynamic_pad_data_collator(examples):
        B = len(examples)
        max_len = max(len(examples[i]["input_ids"]) for i in range(B))
        max_len = min(max_len, max_ctx_len)
        # print(examples[0]["input_ids"])
        input_ids = []
        attention_mask = []
        for i in range(B):
            if len(examples[i]["input_ids"]) > max_len:
                # undefined behavior if indices_col is not None, need to fix relative indexing then (host of issues, including paired activations losing different numbers of indices)
                assert indices_col is None, f"Truncating example {i} to {max_len} tokens, undefined indexing behavior"
                input_ids.append(examples[i]["input_ids"][-max_len:])
                attention_mask.append(examples[i]["attention_mask"][-max_len:])
            
            else:
                input_ids.append(examples[i]["input_ids"] + [tokenizer.pad_token_id] * (max_len - len(examples[i]["input_ids"])))
                attention_mask.append(examples[i]["attention_mask"] + [0] * (max_len - len(examples[i]["attention_mask"])))

        output_dict = {
            # dataset_text_col: [examples[i][dataset_text_col] for i in range(B)],
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
        if dataset_tokens_col is not None:
            output_dict[dataset_tokens_col] = [examples[i][dataset_tokens_col] for i in range(B)]
        else:
            output_dict[dataset_text_col] = [examples[i][dataset_text_col] for i in range(B)]

        if indices_col is not None:
            output_dict[indices_col] = [examples[i][indices_col] for i in range(B)]
        return output_dict
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dynamic_pad_data_collator
    )

    return dataloader

def get_pcas_of_acts(
    model,
    tokenizer,
    dataset,
    cache_locations: Union[List[Tuple[int, str]], List[Tuple[str, str]]],
    dims: Union[int, List[int]],
    gen_batch_size=64,
    pca_batch_size=128,
    num_batches=1024,
    dataset_text_col="text",
    dataset_tokens_col=None,
    device="cuda",
    verbose=False,
    max_ctx_len=2048,
    index_last_pos=None,
    indices_col=None,
):
    if isinstance(dims, int):
        dims = [dims] * len(cache_locations)

    pcas = {
        loc: BatchedPCA(dim, device)
        for loc, dim in zip(cache_locations, dims)
    }

    act_caches = {}

    def create_cache(loc):
        layer, subcomponent = loc
        cache = ActivationCache()
        act_caches[(layer, subcomponent)] = cache
        return cache
    
    add_hooks(
        model,
        create_cache,
        cache_locations 
    )

    dataloader = tokenize_dataset_for_pca(
        dataset,
        tokenizer,
        gen_batch_size,
        dataset_text_col=dataset_text_col,
        dataset_tokens_col=dataset_tokens_col,
        max_ctx_len=max_ctx_len,
        indices_col=indices_col
    )

    data_iter = enumerate(tqdm(dataloader, desc="PCA data iteration")) if verbose else enumerate(dataloader)
    tot_trained_on = 0

    for batch_idx, batch in data_iter:

        # print(batch)
        # print(f"{len(batch['text'])=}, {len(batch['input_ids'])=}, {len(batch['attention_mask'])=}")
        not_padding_mask = ~(batch["attention_mask"] == 0)
        # print(f"{not_padding_mask.shape=}, {not_padding_mask.sum()=}")
        if index_last_pos is not None:
            not_padding_mask = not_padding_mask[:, -index_last_pos:]

        start_time = time.time()
        with torch.no_grad():
            # model(tokenizer(text, return_tensors="pt", padd).to(device))
            model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        if verbose:
            print(f"Time for model gen: {time.time() - start_time}")
        
        # print(f"{act_caches=}")
        if verbose:
            print(f"{torch.cuda.memory_allocated()/1024**3=}, {torch.cuda.max_memory_allocated()/1024**3=}")

        start_time = time.time()
        if verbose:
            print(f"{len(act_caches.items())=}")
        for loc, cache in act_caches.items():
            # print(f"{cache.activation_cache.shape=}")
            activations = cache.get_cache()
            if index_last_pos is not None:
                if verbose:
                    print(f"Accessing last {index_last_pos} positions")
                    print(f"{activations.shape=}, {not_padding_mask.shape=}, {index_last_pos=}")
                activations = activations[:, -index_last_pos:]
                activations = activations[not_padding_mask]

            elif indices_col is not None and indices_col in batch:
                if verbose:
                    print(f"Accessing indices from {indices_col}")
                data_indices = batch[indices_col]
                assert len(data_indices) == activations.shape[0]
                # final mask is indices which are both data_indices and not padding
                final_mask = torch.zeros_like(not_padding_mask)

                tot_indices = 0 # for debugging
                for i in range(len(data_indices)):
                    final_mask[i, data_indices[i]] = not_padding_mask[i, data_indices[i]]
                    tot_indices += len(data_indices[i])
                activations = activations[final_mask]
                if verbose:
                    print(f"{final_mask.shape=}, {final_mask.sum()=}, {not_padding_mask.sum()=}, {tot_indices=}, {activations.shape=}")

            else:
                activations = activations[not_padding_mask]
            assert len(activations.shape) == 2
            tot_trained_on += activations.shape[0]
            # check torch memory
            # print(f"{activations[not_padding_mask].shape=}")
            for idx in range(0, activations.shape[0], pca_batch_size):
                pcas[loc].train_batch(activations[idx:idx + pca_batch_size])

            # pcas[loc].train_batch()
            cache.clear_cache()
        if verbose:
            print(f"Finished training PCA on batch, time taken: {time.time() - start_time}")
        if batch_idx >= num_batches:
            break
    if verbose:
        print(f"Trained on {tot_trained_on} samples")
    return pcas


def get_pcas_of_acts_diff(
    model,
    tokenizer,
    dataset,
    cache_locations: Union[List[Tuple[int, str]], List[Tuple[str, str]]],
    dims: Union[int, List[int]],
    gen_batch_size=64,
    pca_batch_size=128,
    num_batches=1024,
    dataset_text_cols=["text_1","text_2"],
    dataset_tokens_cols=[None,None],
    device="cuda",
    verbose=False,
    max_ctx_len=2048,
    index_last_poss=[None,None],
    indices_cols=[None,None],

):
    if isinstance(dims, int):
        dims = [dims] * len(cache_locations)

    pcas = {
        loc: BatchedPCA(dim, device)
        for loc, dim in zip(cache_locations, dims)
    }

    act_caches = {}

    def create_cache(loc):
        layer, subcomponent = loc
        cache = ActivationCache()
        act_caches[(layer, subcomponent)] = cache
        return cache
    
    add_hooks(
        model,
        create_cache,
        cache_locations
    )

    dataloader_1 = tokenize_dataset_for_pca(
        dataset,
        tokenizer,
        gen_batch_size,
        dataset_text_col=dataset_text_cols[0],
        dataset_tokens_col=dataset_tokens_cols[0],
        max_ctx_len=max_ctx_len,
        indices_col=indices_cols[0]
    )
    dataloader_2 = tokenize_dataset_for_pca(
        dataset,
        tokenizer,
        gen_batch_size,
        dataset_text_col=dataset_text_cols[1],
        dataset_tokens_col=dataset_tokens_cols[1],
        max_ctx_len=max_ctx_len,
        indices_col=indices_cols[1]
    )

    data_iter = enumerate(tqdm(zip(dataloader_1, dataloader_2), desc="PCA data iteration")) if verbose else enumerate(zip(dataloader_1, dataloader_2))

    tot_trained_on = 0
    for batch_idx, (batch_1, batch_2) in data_iter:
        activations_1 = {}
        activations_2 = {}
        for batch_type, batch in [(0, batch_1), (1, batch_2)]:
            indices_col = indices_cols[batch_type]
            index_last_pos = index_last_poss[batch_type]

            not_padding_mask = ~(batch["attention_mask"] == 0)
            # print(f"{not_padding_mask.shape=}, {not_padding_mask.sum()=}")
            if index_last_pos is not None:
                not_padding_mask = not_padding_mask[:, -index_last_pos:]

            start_time = time.time()
            with torch.no_grad():
                # model(tokenizer(text, return_tensors="pt", padd).to(device))
                model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            if verbose:
                print(f"Time for model gen: {time.time() - start_time}")
            
            # print(f"{act_caches=}")
            if verbose:
                print(f"{torch.cuda.memory_allocated()/1024**3=}, {torch.cuda.max_memory_allocated()/1024**3=}")

            start_time = time.time()
            for loc, cache in act_caches.items():
                # print(f"{cache.activation_cache.shape=}")
                activations = cache.get_cache()
                
                if index_last_pos is not None:
                    if verbose:
                        print(f"Accessing last {index_last_pos} positions")
                        print(f"{activations.shape=}, {not_padding_mask.shape=}, {index_last_pos=}")
                    activations = activations[:, -index_last_pos:]
                    activations = activations[not_padding_mask]

                elif indices_col is not None and indices_col in batch:
                    if verbose:
                        print(f"Accessing indices from {indices_col}")
                    data_indices = batch[indices_col]
                    assert len(data_indices) == activations.shape[0]
                    # final mask is indices which are both data_indices and not padding
                    final_mask = torch.zeros_like(not_padding_mask)

                    tot_indices = 0 # for debugging
                    for i in range(len(data_indices)):
                        final_mask[i, data_indices[i]] = not_padding_mask[i, data_indices[i]]
                        tot_indices += len(data_indices[i])
                    activations = activations[final_mask]
                    if verbose:
                        print(f"{final_mask.shape=}, {final_mask.sum()=}, {not_padding_mask.sum()=}, {tot_indices=}, {activations.shape=}")

                else:
                    activations = activations[not_padding_mask]

                if batch_type == 0:
                    tot_trained_on += activations.shape[0]

                if batch_type == 0:
                    activations_1[loc] = activations
                else:
                    activations_2[loc] = activations
                cache.clear_cache()

        # iterate through activations 1 and 2, subtract the two to get the difference, then train the pca on the difference
        assert activations_1.keys() == activations_2.keys()
        for loc in cache_locations:
            diff = activations_1[loc] - activations_2[loc]
            print(f"{diff.shape=}")
            if batch_idx % 2 == 0:
                # flip diff
                diff = -diff
            for idx in range(0, diff.shape[0], pca_batch_size):
                pcas[loc].train_batch(diff[idx:idx + pca_batch_size])

        cache.clear_cache()
        if verbose:
            print(f"Finished training PCA on batch, time taken: {time.time() - start_time}")
        if batch_idx >= num_batches:
            break
    if verbose:
        print(f"Trained on {tot_trained_on} samples")
    return pcas