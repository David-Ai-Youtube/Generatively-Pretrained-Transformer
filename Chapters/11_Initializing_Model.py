#Initializing_Model

#After defining the TransformerModel class, the code creates an instance of this class 
#called model by passing in the hyperparameters defined earlier (n_embd, n_head, n_layer, and dropout) to the class constructor. 
#This initializes all the weights and biases of the model's layers.

#The next step is to check if a GPU is available by calling the torch.cuda.is_available() function. 
#If a GPU is available, the code sets the device to "cuda" and moves the model to the GPU using the to() method. 
#This will enable the model to run on the GPU, which can significantly speed up training.

#If a GPU is not available, the device is set to "cpu", and the model remains on the CPU.



# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Initializing the model
model = nn.Sequential(
    MultiHeadAttention(n_head, n_embd),
    nn.LayerNorm(n_embd),
    FeedFoward(n_embd),
    nn.LayerNorm(n_embd),
)
model.to(device)
