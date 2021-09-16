import json
import nltk
nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import random
import string

class ChatBot:

    def updateKB(self,data):
        self.a_dictionary=data
        with open("updateKB.json", "r+") as file:
            jsFile = json.load(file)
            jsFile.update(self.a_dictionary)
            file.seek(0)
            json.dump(jsFile, file)
    
    def get_random_string(self,length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str


class PreProcess:
    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        return stemmer.stem(word.lower())

    def bag_of_words(self, tokenized_sentence, all_words):
        """
        sentence = ["hello","how","are","you"]
        words = ["hi","hello","I","you","bye","thank","cool"]
        bag   = [ 0  ,    1  , 0 ,  1  ,  0  ,    0  ,   0  ]

        """

        tokenized_sentence = [self.stem(w) for w in tokenized_sentence]
        bag = np.zeros(len(all_words), dtype=np.float32)
        for idx, w in enumerate(all_words):
            if w in tokenized_sentence:
                bag[idx] = 1.0

        return bag
