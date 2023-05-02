#Data Loading Function: 

#The get_batch function is defined to generate small batches of data for inputs and targets. It takes in three arguments: data, idx, and bptt.

    #data is the input data as a list of integers.
    #idx is the starting index for the batch.
    #bptt (backpropagation through time) is the sequence length, or the number of time steps the model will be unrolled during training.

#The function first selects a batch of inputs and targets from the input data using the idx and bptt arguments. 
#It then creates PyTorch tensors from these inputs and targets and sends them to the GPU if it is available. 
#Finally, it returns the inputs and targets as PyTorch tensors.

#Here's the implementation of the get_batch function:

def get_batch(data, idx, bptt):
    seq_len = min(bptt, len(data) - 1 - idx)
    x = data[idx:idx + seq_len]
    y = data[idx + 1:idx + 1 + seq_len].view(-1)
    return torch.tensor(x).to(device), torch.tensor(y).to(device)

#The function first calculates the maximum sequence length based on the bptt parameter and the remaining length of the input data starting from idx. 
#It then selects the inputs x and targets y from the input data, where x is a sequence of length seq_len starting from idx, 
#and y is the sequence of the same length shifted by one character to the right. The targets are flattened into a 1D tensor using the view(-1) method. 
#Finally, the function returns x and y as PyTorch tensors on the GPU if it is available.

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y




