class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # buffer means it won't be counted as a parametr of the model
        self.dropout = nn.Dropout(dropout)  # randomly shuts off some subset of neurons every forward-backward pass to prevent overfitting at train time

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C) 
        q = self.query(x)  # (B, T, C) 
        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, 16) @ (B, 16, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # now the x is private to each token. the v is the thing that gets aggretated for the purpose of this single head
        out = wei @ v  # (B, T, T) @ (B, T, C) = (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # projection layer going back to the residual pathway
        return out