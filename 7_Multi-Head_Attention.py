#Multi-Head Attention: 

#The MultiHeadAttention class is a building block for the Transformer model. 
#It represents the multi-head self-attention mechanism, which allows the model to attend to different parts of the input simultaneously. 
#The class takes two arguments: num_heads, which specifies the number of attention heads to use, and head_size, which specifies the size of each attention head.

#The class first creates a list of Head instances, one for each attention head. 
#Each Head instance is initialized with the head_size argument, and has its own set of query, key, and value linear layers.

#During the forward pass, the input is split into num_heads parts, each of which is processed by a different Head instance. 
#The output from each head is concatenated and projected using a linear layer called proj. 
#The output is then passed through a dropout layer before being returned.

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
