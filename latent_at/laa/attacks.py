import torch
import einops

# The attacker is parameterized by a low-rank MLP (not used by default)
class LowRankAdversary(torch.nn.Module):
    
    def __init__(self, dim, rank, device, bias=False, zero_init=True):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.device = device
        self.lora_A = torch.nn.Linear(dim, rank, bias=False).to(device)
        self.lora_B = torch.nn.Linear(rank, dim, bias=bias).to(device)
        if zero_init:
            self.lora_B.weight.data.zero_()
    
    def forward(self, x):
        return self.lora_B(self.lora_A(x)) + x


# The attacker is parameterized by a full-rank MLP (not used by default)
class FullRankAdversary(torch.nn.Module):
    
    def __init__(self, dim, device, bias=False):
        super().__init__()
        self.dim = dim
        self.device = device
        self.m = torch.nn.Linear(dim, dim, bias=bias).to(device)

        self.m.weight.data.zero_()
    
    def forward(self, x):
        return self.m(x) + x


# Standard projected gradient attack (used by default)
class GDAdversary(torch.nn.Module):
    
    def __init__(self, dim, epsilon, attack_mask, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.epsilon = epsilon

        if dtype:
            self.attack = torch.nn.Parameter(torch.zeros(
                attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device, dtype=dtype))
        else:
            self.attack = torch.nn.Parameter(torch.zeros(
                attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device))
        self.clip_attack()
        self.attack_mask = attack_mask
    
    def forward(self, x):
        if x.shape[1] == 1 and self.attack.shape[1] != 1:  # generation mode (perturbation already applied)
            return x
        else:

            if self.device is None or self.device != x.device:
                with torch.no_grad():
                    self.device = x.device
                    self.attack.data = self.attack.data.to(self.device)
                    self.attack_mask = self.attack_mask.to(self.device)

            perturbed_acts = x[self.attack_mask[:, :x.shape[1]]] + self.attack[self.attack_mask[:, :x.shape[1]]].to(x.dtype)
            x[self.attack_mask[:, :x.shape[1]]] = perturbed_acts

            return x
    
    def clip_attack(self):
        with torch.no_grad():
            # clip attack norm to eps
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)

            norms = torch.norm(self.attack, dim=-1)


# Whitened adversaries train perturbations in a whitened space (not used by default)
class WhitenedGDAdversary(torch.nn.Module):

    def __init__(self, dim, device, epsilon, attack_mask, proj=None, inv_proj=None):
        super().__init__()
        self.attack = None
        self.device = device
        self.epsilon = epsilon
        self.proj = proj  # proj is a projection matrix (e.g. one obtained using PCA) to whiten the attack

        if inv_proj is None:
            self.inv_proj = torch.inverse(proj)
        else:
            self.inv_proj = inv_proj

        self.attack = torch.nn.Parameter(torch.randn(
            attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device))
        self.clip_attack()
        self.attack_mask = attack_mask
    
    def forward(self, x):
        unprojected_attack = torch.einsum("n d, batch seq n-> batch seq d", self.inv_proj, self.attack) # n is whitened dimension, d is original hidden size (technically same here)
        x[self.attack_mask[:, :x.shape[1]]] = (x + unprojected_attack.to(x.dtype))[self.attack_mask[:, :x.shape[1]]]
        return x

    
    def clip_attack(self):
        with torch.no_grad():
            # clip attack norm to eps
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)

            norms = torch.norm(self.attack, dim=-1)