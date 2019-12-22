import itertools
import os
import random
from text_processing import Tokenizer, tokenizer_from_json
import numpy as np
from nltk import word_tokenize
import spacy
from tensorflow.keras.preprocessing import text, sequence


nlp = spacy.load('en', disable=['parser', 'tagger', 'entity'])

def get_tokenizer(tokenizer):
    if tokenizer == "spacy":
        return lambda x: [str(i) for i in nlp(x)]
    else:
        return word_tokenize

class TextProcessor: 
    def __init__(self, num_words=None, tokenizer=None, max_length=None):
        filters = ''
        self.tokenizer = Tokenizer(num_words=num_words, filters=filters,\
                                oov_token="[unk]",
                                custom_tokenizer_function=get_tokenizer(tokenizer))
        self.max_length = max_length

    def fit(self, data):
        """
            Data must be a list of text
        """
        self.tokenizer.fit_on_texts(data)
    
    def dump(self, data_file):
        with open(data_file, "w") as file_: 
            file_.write(self.tokenizer.to_json())
    
    def load(self, data_file):
        with open(data_file, "r") as file_: 
            self.tokenizer = tokenizer_from_json(file_.read())

    def process(self, data, sequences=True, max_length=None):
        """You can define a maximum length"""
        sequences = self.tokenizer.texts_to_sequences(data)
        if max_length or self.max_length:
            sequences = sequence.pad_sequences(sequences, maxlen=max_length or self.max_length)
        return sequences
    