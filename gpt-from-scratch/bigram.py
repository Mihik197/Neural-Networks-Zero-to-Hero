import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences we will process in parallel
block_size = 8  # max context length
max_iters = 6000
eval_interval = 300
learning_rate = 1e-2  # 10**-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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


# bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)  # embedding table of containing n_embd dimension vectors

    def forward(self, idx, targets=None):

        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)        

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
            # get the predictions
            logits, loss = self(idx)
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


for iter in range(max_iters):

    # every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))