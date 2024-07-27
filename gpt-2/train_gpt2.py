import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # to ensure compatibility with head size and shapes
        # key, query, and value projections for all heads at once
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask (to hide future tokens)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensions
        # calculate query, key, values for all heads simultaneously
        # nh --> "number of heads", hs --> "head size", C --> number of channels = nh * hs
        # eg. in GPT2 (124M), n_head = 12, hs = 64, nh*hs=768 channels
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each will have size n_embd along dim 2
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  
        # normalizes the attention, so it sums to 1
        # i.e. for a given token, how important each token is to it, converted to probabilities summing to 1

        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)  # wieghted sum
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        # GELU is like a slightly smoother RELU, instead of sharp turn
        # ideally we would want to use GELU, with no approximate version
        # but we use it here because GPT-2 does too

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

        # layernorm is before attention and feedforward layers, different from the original transformer paper
        # attention layer is where they communicate with other tokens
        # mlp is where they think individually based on the communicated information

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256  # context length
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6  # number of attention heads in each transformer block(layer)
    n_embd: int = 384
# dataclass automatically adds special methods to classes like init, repr, etc

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ModuleDict stores nn.Module items in a dictionary format, so we can index modules by keys
        # ModuleList is used to store an ordered list of nn.Module items
        # we want it to be compatible with gpt2 weights from huggingface transformers
        # ex: transformer.h.0.ln_1.weight torch.Size([768])