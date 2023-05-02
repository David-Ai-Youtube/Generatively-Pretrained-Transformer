# Generatively-Pretrained-Transformer

PyTorch is a popular open-source machine learning framework used for developing and training neural networks. It is widely used in deep learning research and industry due to its flexibility, ease of use, and scalability.

The nn module in PyTorch provides a high-level interface for building neural networks. It includes classes for defining different types of layers, loss functions, and optimization algorithms. The nn module also provides pre-built models, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), which can be used for a variety of tasks.

The functional (F) module in nn provides a set of functions that can be used to define neural network layers and operations. These functions are stateless and do not have any internal parameters. They can be used to define custom layers and operations in PyTorch.

By importing these libraries, the code can utilize PyTorch's capabilities to build, train and evaluate neural network models for a variety of tasks.

```
import torch
import torch.nn as nn
from torch.nn import functional as F
```

Hyperparameters are values that define the structure and behavior of a machine learning model, but are not learned during the training process. They are set by the user prior to training and can greatly impact the performance of the model.

In this code, several hyperparameters are initialized:

    batch_size: the number of examples in each batch used during training
    block_size: the length of input sequence fed into the model
    max_iters: the maximum number of iterations to train the model
    eval_interval: the number of iterations between evaluating the model on the validation set
    learning_rate: the step size used in the optimization algorithm (in this case, stochastic gradient descent)
    device: the device on which the model will be trained (e.g. "cpu" or "cuda" for a GPU)
    eval_iters: the number of iterations used to estimate the validation loss
    n_embd: the dimensionality of the embedding layer in the model
    n_head: the number of self-attention heads in the multi-head attention layer of the model
    n_layer: the number of transformer blocks in the model
    dropout: the dropout probability used in the model for regularization

By setting these hyperparameters, the code can be customized to the specific requirements of the task and the available resources. Choosing appropriate hyperparameters is crucial for achieving good performance in a machine learning model.

```
    batch_size = 16 # how many independent sequences will we process in parallel?
    block_size = 32 # what is the maximum context length for predictions?
    max_iters = 5000 # maximum number of iterations for training
    eval_interval = 100 # how often should we evaluate the model during training?
    learning_rate = 1e-3 # learning rate for the optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # use CUDA if available, otherwise use CPU
    eval_iters = 200 # number of iterations for evaluation
    n_embd = 64 # dimensionality of the embeddings
    n_head = 4 # number of attention heads in each layer
    n_layer = 4 # number of layers in the transformer model
    dropout = 0.0 # dropout probability during training
```

The code reads input data from a text file named 'input.txt'. The text file is read using Python's built-in function, open(), and the entire text is stored as a string in a variable called text. This text is then used to create a mapping between characters and integers, which is required for feeding the data into the model.

The mapping between characters and integers is created using two dictionaries: stoi and itos. stoi is a dictionary that maps each unique character in the input text to a unique integer value, and itos is a dictionary that maps each unique integer value back to its corresponding character.

Once the mapping between characters and integers has been created, the input text is encoded as a list of integers using the encode() function. This list of integers is then split into training and validation sets using a simple slicing operation. The first 90% of the data is used for training, and the remaining 10% is used for validation.

Overall, this process of reading and preprocessing data is an important step in preparing the data for training the model. By converting the input text into a format that can be processed by the model, it allows the model to learn patterns and relationships in the data that can be used for generating new text.

The get_batch function is defined to generate small batches of data for inputs and targets. It takes in three arguments: data, idx, and bptt.

    data is the input data as a list of integers.
    idx is the starting index for the batch.
    bptt (backpropagation through time) is the sequence length, or the number of time steps the model will be unrolled during training.

The function first selects a batch of inputs and targets from the input data using the idx and bptt arguments. It then creates PyTorch tensors from these inputs and targets and sends them to the GPU if it is available. Finally, it returns the inputs and targets as PyTorch tensors.

Here's the implementation of the get_batch function:

python

def get_batch(data, idx, bptt):
    seq_len = min(bptt, len(data) - 1 - idx)
    x = data[idx:idx + seq_len]
    y = data[idx + 1:idx + 1 + seq_len].view(-1)
    return torch.tensor(x).to(device), torch.tensor(y).to(device)

The function first calculates the maximum sequence length based on the bptt parameter and the remaining length of the input data starting from idx. It then selects the inputs x and targets y from the input data, where x is a sequence of length seq_len starting from idx, and y is the sequence of the same length shifted by one character to the right. The targets are flattened into a 1D tensor using the view(-1) method. Finally, the function returns x and y as PyTorch tensors on the GPU if it is available.

The estimate_loss function takes the model, the training and validation datasets, the loss function, and the device as input. It first initializes two variables, total_loss and total_count, to zero.

Then, for each batch in the training dataset, it calls the get_batch function to get a batch of inputs and targets, and sends them to the device. It computes the model's forward pass on the inputs, calculates the loss between the predicted outputs and the targets using the specified loss function, and adds this loss to the total_loss variable. It also adds the number of elements in the batch to the total_count variable.

After processing all the batches in the training dataset, it divides the total_loss by the total_count to get the average loss per element in the dataset. It repeats the same process for the validation dataset, and returns the training and validation loss per element.

In summary, the estimate_loss function computes the average loss for a given dataset using a specified loss function and the model's predictions on that dataset. This function is useful for evaluating the model's performance during training and monitoring for overfitting or underfitting.

In the self-attention mechanism of the transformer, each input token has a query, key, and value vector. These vectors are used to compute the attention scores between the input tokens, which are then used to compute a weighted sum of the values, producing the output. The Head class in the code defines these three linear transformations.

The Head class takes a head_size argument, which determines the size of the query, key, and value vectors. It then defines three linear transformations, key, query, and value, using PyTorch's nn.Linear module. These linear transformations take an input tensor of shape (batch_size, block_size, n_embd) and transform it into query, key, and value tensors of shape (batch_size, block_size, head_size).

In addition to the linear transformations, the Head class also defines a buffer called tril, which is used to mask out the upper triangular part of the attention scores. This is done to ensure that each token only attends to the tokens that come before it in the sequence, and not to tokens that come after it, which would violate the causal nature of the transformer. The tril buffer is initialized to a lower-triangular matrix with ones in the lower triangle and zeros in the upper triangle.

The MultiHeadAttention class is a building block for the Transformer model. It represents the multi-head self-attention mechanism, which allows the model to attend to different parts of the input simultaneously. The class takes two arguments: num_heads, which specifies the number of attention heads to use, and head_size, which specifies the size of each attention head.

The class first creates a list of Head instances, one for each attention head. Each Head instance is initialized with the head_size argument, and has its own set of query, key, and value linear layers.

During the forward pass, the input is split into num_heads parts, each of which is processed by a different Head instance. The output from each head is concatenated and projected using a linear layer called proj. The output is then passed through a dropout layer before being returned.

The FeedForward class in the code represents a simple feedforward neural network. It takes the number of embedding dimensions (n_embd) as an argument and defines a sequential neural network with two linear layers, a ReLU activation function, and a dropout layer.

The first linear layer projects the input to a higher dimensional space, and the ReLU activation function applies a non-linearity to the output. The dropout layer randomly drops some of the activations to prevent overfitting. Finally, the second linear layer projects the output back to the original dimensionality.

Overall, the FeedForward class is used as part of the Transformer block to transform the output of the self-attention layer before passing it to the next layer in the network.

The transformer block is the basic building block of the transformer model. The code defines a class, Block, which represents a transformer block. The block contains two sub-layers: a multi-head attention layer and a feedforward layer. Each sub-layer is followed by a layer normalization step, which helps to stabilize the training process.

The Block class takes the following arguments:

    n_embd: The number of embedding dimensions.
    n_head: The number of attention heads.
    dropout: The dropout rate.
    block_size: The maximum block size.

The Block class has the following methods:

    forward(x): This method takes an input tensor x of shape (batch_size, block_size, n_embd) and passes it through the multi-head attention and feedforward layers, followed by layer normalization. The output of the layer is returned.

The Block class is used to create the encoder and decoder layers of the transformer model.

The TransformerModel class defines the entire transformer model architecture by combining multiple Block layers along with the embedding layer and the final linear output layer.

The __init__ method initializes the hyperparameters and creates the embedding layer, as well as a list of Block layers. The forward method then takes the input tokens, passes them through the embedding layer, applies the series of Block layers, and finally passes the output through the final linear layer to generate the predicted output.

Specifically, the __init__ method defines the following components:

    self.encoder: The embedding layer which maps input tokens to the vector space. The dimension of the vector space is defined by the n_embd hyperparameter.

    self.blocks: A list of Block layers. The number of blocks is defined by the n_layer hyperparameter.

    self.decoder: The final linear layer which maps the output of the last block to the output dimension. The output dimension is defined by the number of unique characters in the input data.

The forward method takes the following steps:

    Pass the input tokens through the embedding layer to get the input embeddings.

    Scale the input embeddings by multiplying with the square root of the embedding dimension.

    Pass the scaled embeddings through each block in self.blocks in sequence. Each block outputs a new set of embeddings.

    Pass the output embeddings of the last block through the decoder to obtain the final output.

    Return the final output.

This way, the TransformerModel class defines the entire transformer model architecture that can be trained and used for prediction.

After defining the TransformerModel class, the code creates an instance of this class called model by passing in the hyperparameters defined earlier (n_embd, n_head, n_layer, and dropout) to the class constructor. This initializes all the weights and biases of the model's layers.

The next step is to check if a GPU is available by calling the torch.cuda.is_available() function. If a GPU is available, the code sets the device to "cuda" and moves the model to the GPU using the to() method. This will enable the model to run on the GPU, which can significantly speed up training.

If a GPU is not available, the device is set to "cpu", and the model remains on the CPU.

The training loop in the code is the main part of the program that trains the Transformer model. It runs for a fixed number of iterations specified by the max_iters hyperparameter. During each iteration, the loop performs the following steps:

    Gets a batch of input data and target output data using the get_batch function. The size of the batch is specified by the batch_size hyperparameter.

    Sends the input and target data to the GPU if available.

    Resets the gradients of the model parameters.

    Computes the forward pass of the model on the input data.

    Calculates the loss between the model's output and the target data using the estimate_loss function.

    Computes the gradients of the loss with respect to the model parameters using backpropagation.

    Updates the model parameters using stochastic gradient descent (SGD) with the learning rate specified by the learning_rate hyperparameter.

    Periodically evaluates the model on the validation dataset at intervals specified by the eval_interval hyperparameter.

After the training loop finishes, the final model parameters are saved to a file named 'model.pth'.

After training the model for a certain number of iterations, the code evaluates the model's performance on both the training and validation datasets using the estimate_loss function. This function computes the average loss across all batches of the dataset, where the loss is computed as the negative log-likelihood of the target sequence given the input sequence.

The evaluation is performed every eval_interval iterations, which is a hyperparameter set at the beginning of the script. During each evaluation, the function prints the average loss for both the training and validation datasets.

This evaluation allows the user to monitor the model's progress over time and detect any signs of overfitting or underfitting. If the training loss is significantly lower than the validation loss, it suggests that the model is overfitting to the training data and may not generalize well to new data. In this case, one can consider reducing the model's capacity or introducing regularization techniques such as dropout. If both the training and validation losses are high, it suggests that the model is underfitting and may benefit from increasing its capacity or adjusting the hyperparameters.

