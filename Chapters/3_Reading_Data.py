#Reading Data:

#The code reads input data from a text file named 'input.txt'. The text file is read using Python's built-in function, open(), and the entire text is stored as a string in a variable called text. 
#This text is then used to create a mapping between characters and integers, which is required for feeding the data into the model.

#The mapping between characters and integers is created using two dictionaries: stoi and itos. stoi is a dictionary that maps each unique character in the input text to a unique integer value, 
#and itos is a dictionary that maps each unique integer value back to its corresponding character.

#Once the mapping between characters and integers has been created, the input text is encoded as a list of integers using the encode() function. 
#This list of integers is then split into training and validation sets using a simple slicing operation. 
#The first 90% of the data is used for training, and the remaining 10% is used for validation.

#Overall, this process of reading and preprocessing data is an important step in preparing the data for training the model. 
#By converting the input text into a format that can be processed by the model, it allows the model to learn 
#patterns and relationships in the data that can be used for generating new text.

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

