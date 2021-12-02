import os
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
import string
from nltk import word_tokenize
import argparse
import matplotlib.pyplot as plt
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH=100

class Vocab:
    def __init__(self, name, sep=" "):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.sep = sep
    def addSentence(self, sentence):
        for word in sentence.split(self.sep):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence, sep=' '):
    return [lang.word2index[word] for word in sentence.split(sep)]

def tensorFromSentence(lang, sentence, sep=' '):
    indexes = indexesFromSentence(lang,sentence,sep)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair,input_lang,output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0],' ')
    target_tensor = tensorFromSentence(output_lang, pair[1],',')
    return (input_tensor, target_tensor)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def getInputOutputVocabs(path):
    if os.path.isfile(path):
        lines = open(path,encoding="utf-8").read().strip().split('\n')
    else: assert False, "[ERROR] The txt path is invalid."
    pairs = [l.split('|') for l in lines]
    pairs = filterPairs(pairs)
    input_vocab = Vocab('sentence'," ")
    output_vocab = Vocab('level',",")
    for pair in pairs:
        input_vocab.addSentence(pair[0])
        output_vocab.addSentence(pair[1])
    return input_vocab,output_vocab,pairs

class PreprocessCSV:
    def __init__(self,config):
        """Read csv, lemmaziation func, stopword and english word vocabs."""
        self.config = config
        self._loadcsv()
        self._lem = WordNetLemmatizer()
        self._stop = set(stopwords.words('english')+list(string.punctuation))
        self._eng = set(nltk.corpus.words.words())
    @staticmethod
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    def _loadcsv(self):
        if os.path.isfile(self.config.csv):
            self.df = pd.read_csv(self.config.csv)
        else: assert False, "[ERROR] The csv path is invalid."
    @staticmethod
    def filterMethod(paragraph,dictionary):
        """
        Inputs:
        1. paragraph: lowercase string.
        2. dictionary: lemmatized lowercase english vocabularies.
        Description of function: 
        1. Filter non-english words.
        2. Filter stopwords.
        3. Lemmatize string.
        5. Return string with unique vocabularies only.
        """
        lem = WordNetLemmatizer()
        stop = set(stopwords.words('english')+list(string.punctuation))
        lemEngNoStop = " ".join(lem.lemmatize(w,PreprocessCSV.get_wordnet_pos(w)) for w in nltk.wordpunct_tokenize(paragraph) if w in dictionary and w not in stop)
        uniqueLemEngNoStop = " ".join(np.unique(lemEngNoStop.split(" ")))
        return uniqueLemEngNoStop
    def _save(self):
        with open(self.config.pairs_txt,'w') as f:
            for i in range(self.df.shape[0]):
                tar = ",".join([self.df['l1'].iloc[i].lower(),
                                self.df['l2'].iloc[i].lower(),
                                self.df['l3'].iloc[i].lower()])
                dstream = "|".join([self.df['filter_text'].iloc[i],tar])
                f.write(dstream+"\n")
    def process2save(self):
        self.df['filter_text'] = ""
        print('[INFO] Processing rows in csv...')
        for i in tqdm(range(self.df.shape[0])):
            self.df['filter_text'].iloc[i] = PreprocessCSV.filterMethod(self.df['text'].iloc[i].lower(),self._eng)
        print('[INFO] Saving input and output pairs into txt...')
        self._save()
        print('[INFO] Done.')

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--csv',type=str,default="/media/yui/Disk/data/util_task/classification_dataset.csv")
    p.add_argument('--pairs_txt',type=str,default="/media/yui/Disk/data/util_task/pairs.txt")
    args = p.parse_args()
    process = PreprocessCSV(args)
    process.process2save()
