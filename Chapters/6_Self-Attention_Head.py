#Self-Attention Head

#In the self-attention mechanism of the transformer, each input token has a query, key, and value vector. 
#These vectors are used to compute the attention scores between the input tokens, which are then used to compute a weighted sum of the values, producing the output. 
#The Head class in the code defines these three linear transformations.

#The Head class takes a head_size argument, which determines the size of the query, key, and value vectors. 
#It then defines three linear transformations, key, query, and value, using PyTorch's nn.Linear module. 
#These linear transformations take an input tensor of shape (batch_size, block_size, n_embd) and transform it into query, 
#key, and value tensors of shape (batch_size, block_size, head_size).

#In addition to the linear transformations, the Head class also defines a buffer called tril, 
#which is used to mask out the upper triangular part of the attention scores. 
#This is done to ensure that each token only attends to the tokens that come before it in the sequence, 
#and not to tokens that come after it, which would violate the causal nature of the transformer. 
#The tril buffer is initialized to a lower-triangular matrix with ones in the lower triangle and zeros in the upper triangle.

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
