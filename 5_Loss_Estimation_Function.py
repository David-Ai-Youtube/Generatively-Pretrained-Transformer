#Loss Estimation Function: 

#The estimate_loss function takes the model, the training and validation datasets, the loss function, and the device as input. 
#It first initializes two variables, total_loss and total_count, to zero.

#Then, for each batch in the training dataset, it calls the get_batch function to get a batch of inputs and targets, and sends them to the device. 
#It computes the model's forward pass on the inputs, calculates the loss between the predicted outputs and the targets using the specified loss function, 
#and adds this loss to the total_loss variable. It also adds the number of elements in the batch to the total_count variable.

#After processing all the batches in the training dataset, it divides the total_loss by the total_count to get the average loss per element in the dataset. 
#It repeats the same process for the validation dataset, and returns the training and validation loss per element.

#In summary, the estimate_loss function computes the average loss for a given dataset using a specified loss function and the model's predictions on that dataset. 
#This function is useful for evaluating the model's performance during training and monitoring for overfitting or underfitting.

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

