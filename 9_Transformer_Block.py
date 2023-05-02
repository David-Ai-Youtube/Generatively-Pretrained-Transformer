#Transformer Block:
    
#The transformer block is the basic building block of the transformer model. 
#The code defines a class, Block, which represents a transformer block. 
#The block contains two sub-layers: a multi-head attention layer and a feedforward layer. 
#Each sub-layer is followed by a layer normalization step, which helps to stabilize the training process.

#The Block class takes the following arguments:

    #n_embd: The number of embedding dimensions.
    #n_head: The number of attention heads.
    #dropout: The dropout rate.
    #block_size: The maximum block size.

#The Block class has the following methods:

    #forward(x): This method takes an input tensor x of shape (batch_size, block_size, n_embd) and passes it through the multi-head attention and feedforward layers, 
    #followed by layer normalization. The output of the layer is returned.

#The Block class is used to create the encoder and decoder layers of the transformer model.

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ff = FeedFoward(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        out = x + self.sa(self.ln1(x))
        out = out + self.ff(self.ln2(out))
        return out
