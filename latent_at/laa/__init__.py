from typing import Any, Callable, List, Tuple, Union

import torch
import transformers
try:
    from transformer_lens import HookedTransformer
    IS_USING_TL = True
except ImportError:
    HookedTransformer = None
    IS_USING_TL = False

# custom hooks if we're not using transformer-lens
class CustomHook(torch.nn.Module):
    def __init__(self, module, hook_fn):
        super().__init__()
        self.module = module
        self.hook_fn = hook_fn
        self.enabled = True

    def forward(self, *args, **kwargs):
        if self.enabled:
            return self.hook_fn(self.module(*args, **kwargs))
        else:
            return self.module(*args, **kwargs)

# a hook for transformer-lens models
class TLHook(torch.nn.Module):
    def __init__(self, hook_fn):
        super().__init__()
        self.hook_fn = hook_fn
        self.enabled = True
    
    def forward(self, activations, hook):
        if self.enabled:
            return self.hook_fn(activations)
        else:
            return activations

def _remove_hook(parent, target):
    for name, module in parent.named_children():
        if name == target:
            setattr(parent, name, module.module)
            return

def insert_hook(parent, target, hook_fn):
    hook = None
    for name, module in parent.named_children():
        if name == target and hook is None:
            hook = CustomHook(module, hook_fn)
            setattr(parent, name, hook)
        elif name == target and hook is not None:
            _remove_hook(parent, target)
            raise ValueError(f"Multiple modules with name {target} found, removed hooks")
    
    if hook is None:
        raise ValueError(f"No module with name {target} found")

    return hook

def remove_hook(parent, target):
    is_removed = False
    for name, module in parent.named_children():
        if name == target and isinstance(module, CustomHook):
            setattr(parent, name, module.module)
            is_removed = True
        elif name == target and not isinstance(module, CustomHook):
            raise ValueError(f"Module {target} is not a hook")
        elif name == target:
            raise ValueError(f"FATAL: Multiple modules with name {target} found")

    if not is_removed:
        raise ValueError(f"No module with name {target} found")

def clear_hooks(model):
    for name, module in model.named_children():
        if isinstance(module, CustomHook):
            setattr(model, name, module.module)
            clear_hooks(module.module)
        else:
            clear_hooks(module)

# main function for adding adversaries
# supports both tl and pytorch models
def add_hooks(
    model: Union[HookedTransformer, torch.nn.Module],
    create_adversary: Callable[[Union[Tuple[int, str], Tuple[str, str]]], Any],
    adversary_locations: Union[List[Tuple[int, str]], List[Tuple[str, str]]]
):
    # check if we're using transformer-lens
    if IS_USING_TL:
        is_tl_model = isinstance(model, HookedTransformer)
    else:
        is_tl_model = False

    # adverseries is a list of things created by `create_adversary()`
    # hooks is a list of the hooks we've added to the model
    adversaries = []
    hooks = []

    if is_tl_model and isinstance(adversary_locations[0][0], int):
        for layer, subcomponent in adversary_locations:
            # get the hook
            hook = model.blocks[layer].get_submodule(subcomponent)
            # create an adversary
            adversaries.append(create_adversary((layer, subcomponent)))
            # add it to the hook (TLHook just wraps so that we can use the same interface)
            hooks.append(TLHook(adversaries[-1]))
            hook.add_hook(hooks[-1])
        
        return adversaries, hooks

    if len(adversary_locations) == 0:
        raise ValueError("No hook points provided")

    for layer, subcomponent in adversary_locations:
        # to add an adversary at (layer, subcomponent), we first get the layer
        parent = model.get_submodule(layer)
        # then we call `create_adversary()` to make a new adversary
        adversaries.append(create_adversary((layer, subcomponent)))
        # then we use `insert_hook` to add the adversary to the model
        hooks.append(insert_hook(parent, subcomponent, adversaries[-1]))
        # internally, `insert_hook` creates a new `CustomHook` and sets it as the subcomponent of `parent`

    return adversaries, hooks


"""
Deepspeed version of the above function
"""

from typing import List, Union, Tuple, Callable, Any
import torch
from torch.nn import Module

class AdversaryWrapper(Module):
    def __init__(self, module: Module, adversary: Any):
        super().__init__()
        self.module = module
        self.adversary = adversary

    def forward(self, *inputs, **kwargs):
        outputs = self.module(*inputs, **kwargs)
        return self.adversary(outputs)

def deepspeed_add_hooks(
    model: Module,
    create_adversary: Callable[[Union[Tuple[int, str], Tuple[str, str]]], Any],
    adversary_locations: Union[List[Tuple[int, str]], List[Tuple[str, str]]]
):
    adversaries = []
    hooks = []

    if len(adversary_locations) == 0:
        raise ValueError("No hook points provided")

    for layer, subcomponent in adversary_locations:
        # Get the parent module
        parent = model.get_submodule(layer)
        # Create an adversary
        adversary = create_adversary((layer, subcomponent))
        adversaries.append(adversary)
        # Wrap the subcomponent with the adversary
        submodule = parent.get_submodule(subcomponent)
        wrapped_module = AdversaryWrapper(submodule, adversary)
        hooks.append(wrapped_module)
        # Replace the subcomponent with the wrapped module
        setattr(parent, subcomponent, wrapped_module)

    return adversaries, hooks