#Hyperparameters:

#Hyperparameters are values that define the structure and behavior of a machine learning model, but are not learned during the training process. 
#They are set by the user prior to training and can greatly impact the performance of the model.

#In this code, several hyperparameters are initialized:


    # batch_size: the number of examples in each batch used during training
    # block_size: the length of input sequence fed into the model
    # max_iters: the maximum number of iterations to train the model
    # eval_interval: the number of iterations between evaluating the model on the validation set
    # learning_rate: the step size used in the optimization algorithm (in this case, stochastic gradient descent)
    # device: the device on which the model will be trained (e.g. "cpu" or "cuda" for a GPU)
    # eval_iters: the number of iterations used to estimate the validation loss
    # n_embd: the dimensionality of the embedding layer in the model
    # n_head: the number of self-attention heads in the multi-head attention layer of the model
    # n_layer: the number of transformer blocks in the model
    # dropout: the dropout probability used in the model for regularization


#By setting these hyperparameters, the code can be customized to the specific requirements of the task and the available resources. 
#Choosing appropriate hyperparameters is crucial for achieving good performance in a machine learning model.

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
