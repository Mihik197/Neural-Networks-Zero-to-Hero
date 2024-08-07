# This is the modified transformer, with the Head as just one class, unlike in the lecture

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many independent sequences we will process in parallel
block_size = 256  # max context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4  # 3x10**-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
file_path = 'final_tranformer_output.txt'  # output file path
# -----------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mappings of characters to and from integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]  # encoder: takes a string, returns a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: takes a list of integers, returns a string


# let's now encode the entire dataset and store it in a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# let's split this data into train and validation sets
n = int(0.9*len(data))  # 90% train, 10% test
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # randomly selects batch_size number of starting indices, that's why we do len(data) - block_size, so we don't choose the last few indices to begin with
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad  # doesn't track gradients for these calculations
def estimate_loss():
    out = {}
    model.eval()  # sets the model to evaluation mode. some layers like batchnorm behave differently during training and test time
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  # a tensor of 200 zeros to store losses
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # set model back to training mode
    return out
    
class Attention(nn.Module):
    """self and multi head attention in one"""

    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head  # ex: embd dim: 512, heads: 8, then head size will have to be 64

        self.combined_attn = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size).view(1, 1, block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.combined_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)  # equally split into 3 parts each of n_embd size. Ex: 512
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # we reshape them to separate the heads. Ex: 8 heads of 64 each
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # on transpose shape becomes (B, n_head, T, head_size), allows us to process all heads in parallel
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))  # (B, n_head, T, head_size) @ (B, n_head, head_size, T) = (B, n_head, T, T)
        attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # gives (B, T, n_head, head_size), and then concatenates back to original shape
        out = self.dropout(self.proj(out))
        return out

# the tokens looked at each but didn't really get a lot of time to think on what they found
# this is why we introduce these linear layers to give a place to think
# this is on the per token level, all tokens are thinking independently here
# self-attention is the communication, and this linear layer is the thinking individually part
class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # the inner layer has dimensionality of 4x the input in the paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection layer going back to the residual pathway
            nn.Dropout(dropout),  # its something you can add right before connecting back to the residual pathway
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = Attention(n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # this is slight deviation from the og paper, but it's current practice to apply layernorm before the transformation
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # the 'x +' is for residual connections
        x = x + self.ffwd(self.ln2(x))  # called the pre-norm formulation
        return x

# bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # embedding table of containing n_embd dimension vectors
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # embedding table of the position (out of 8 as block_size, for example)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # number of inputs and outputs

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)  the third dimension is the embeddings        
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C) through broadcasting
        x = self.blocks(x)  # (B, T, C) 
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            # cross_entropy expects arguments to be of shape (N, C, d1, d2, ...) where N is batch size, C is number of classes
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # 4*8=32 samples with 65 classes each
            targets = targets.view(-1)  # cross_entropy expects target of shape (N, d1, d2, ...), i.e. 4*8 = 32
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)  # pluck out the last element in the time dimension across all batches
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
    

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

lossi = []
for iter in range(max_iters):

    # every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        loss_log = f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        lossi.append(loss_log)
        print(loss_log)
        # Write the loss log to the file
        with open(file_path, 'a') as f:
            f.write(loss_log + '\n')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = decode(m.generate(context, max_new_tokens=10000)[0].tolist())
print(output)
# save output to a txt file
with open(file_path, 'a') as f:
    f.write(output)

