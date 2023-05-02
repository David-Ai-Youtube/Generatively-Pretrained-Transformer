import torch
import numpy as np

# Creating tensors
tensor0 = torch.Tensor([[1, 2], [3, 4]])   # create a 2x2 tensor with default float type
tensor1 = torch.Tensor([[1, 2], [3, 4]])   # create a 2x2 tensor with default float type
tensor2 = torch.LongTensor([5, 6, 7])      # create a 1D tensor of long integers
tensor3 = torch.randn(3, 4)                # create a 3x4 tensor with random values

# Getting information from tensors
print(tensor1.size())                      # output: torch.Size([2, 2])
print(tensor2.dtype)                       # output: torch.int64
print(tensor3.mean())                      # output: tensor(0.1108)
print()

# create two random tensors with the same shape
a = np.random.rand(2, 3)
b = np.random.rand(2, 3)

# convert the numpy arrays to PyTorch tensors
a = torch.from_numpy(a)
b = torch.from_numpy(b)

# multiply the two tensors element-wise
c = torch.mul(a, b)
print(c, " using torch")
print()

# convert the PyTorch tensor back to a numpy array
c = c.numpy()
print(c, " using numpy")
print()

# Dealing with tensor shapes
tensor7 = tensor1.view(1, 4)               # reshape tensor1 into a 1x4 tensor
tensor8 = tensor2.unsqueeze(1)             # add a new dimension to tensor2 (1D -> 2D)

# Indexing on tensors
print(tensor1[0, 1])                       # output: tensor(2.)
print(tensor3[:, 1:3])                     # output: a 3x2 tensor
print()

# Mixing PyTorch tensors and NumPy
numpy_array = np.array([[8, 9], [10, 11]])
tensor9 = torch.from_numpy(numpy_array)    # convert a NumPy array to a PyTorch tensor
numpy_array2 = tensor5.numpy()             # convert a PyTorch tensor to a NumPy array

# Reproducibility
torch.manual_seed(42)                      # set a manual seed

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")          # set device to GPU
    print("Using GPU")
else:
    device = torch.device("cpu")           # set device to CPU
    print("Using CPU")

# Create a tensor and move it to GPU
tensor1 = torch.Tensor([[1, 2], [3, 4]])
tensor1 = tensor1.to(device)
