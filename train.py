import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import PreProcess
p=PreProcess()
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = p.tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [p.stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

class Train:
    def __init__(self):
        self.X_train = []
        self.y_train = []       
        
    # create training data
    def createTrainData(self):
        for (pattern_sentence, tag) in xy:
            # X: bag of words for each pattern_sentence
            self.bag = p.bag_of_words(pattern_sentence, all_words)
            self.X_train.append(self.bag)
            # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
            self.label = tags.index(tag)
            self.y_train.append(self.label)

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

    def trainModel(self,train_loader):
        # Train the model
        for epoch in range(num_epochs):
            for (words, labels) in train_loader:
                self.words = words.to(device)
                self.labels = labels.to(dtype=torch.long).to(device)
                
                # Forward pass
                outputs = model(words)
                # if y would be one-hot, we must apply
                # labels = torch.max(labels, 1)[1]
                loss = criterion(outputs, self.labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if (epoch+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def saveTrainData(self):
        data={
        "model_state":model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags":tags
        }

        FILE="data.pth"
        torch.save(data,FILE)
        print(f'training complete, filesaved to {FILE}')

t=Train()
t.createTrainData()

# Hyper-parameters 
num_epochs = 1000
batch_size = 10
learning_rate = 0.001
input_size = len(t.X_train[0])
hidden_size = 50
output_size = len(tags)
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(t.X_train)
        self.x_data = t.X_train
        self.y_data = t.y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

t.trainModel(train_loader)
t.saveTrainData()

