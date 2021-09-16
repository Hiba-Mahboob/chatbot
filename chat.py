import json
import random
import torch
from yaml.events import SequenceStartEvent
from model import NeuralNet
from nltk_utils import PreProcess, ChatBot
c=ChatBot()
p=PreProcess()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents=json.load(f)

with open('fallbacks.json','r') as f:
    fallbacks=json.load(f)

FILE="data.pth"
data=torch.load(FILE)

input_size=data["input_size"]
hidden_size=data["hidden_size"]
output_size=data["output_size"]
all_words=data["all_words"]
tags=data["tags"]
model_state=data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

class Query:
    def __init__(self):
        self.bot_name="University Chatbot"

    def get_response(self,msg):
        sentence=p.tokenize(msg)
        X=p.bag_of_words(sentence, all_words)
        X=X.reshape(1, X.shape[0])    
        X=torch.from_numpy(X)

        output=model(X)
        _, predicted=torch.max(output, dim=1)
        tag=tags[predicted.item()]

        probs=torch.softmax(output, dim=1)
        prob=probs[0][predicted.item()]

        if prob.item()>0.75:
            for intent in intents["intents"]:
                if tag==intent["tag"]:
                    return random.choice(intent['responses'])
                                   
        
        pattern_id=c.get_random_string(4)
        data={}
        data[pattern_id]=msg
        c.updateKB(data)
        return "I do not understand..., but i am saving it to response in future"
        
    
    def enterQuery(self):
        print("Lets chat! type 'quit' to exit")       


            

q=Query()
q.enterQuery()