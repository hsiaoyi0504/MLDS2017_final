# Parses training data / testing data for language model & word2vec
# This script will create a folder in data called "Parsed_Training_Data/".
# Under this folder, there are two folders, "word" and "sentence", respectively.
# Each folder has many files parsed from input training data.
# Under word foler, each file contains a long list of words which exist in corresponding file before parsing.
# Each line only has one word.
# Similarly, under sentence folder, each file contains a long list of sentences which exist in file corresponding file before parsing.
# Each line only has one sentence.
# Usage: python parse_data.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import os
import sys
import string
import nltk
import pandas as pd

try:
    nltk.data.find("tokenizers\punkt\english.pickle")
except LookupError:
    nltk.download("punkt")
from nltk.tokenize import word_tokenize, sent_tokenize

def read_file(read_path):
    with open(read_path,"r",errors="ignore") as f:
        lines = f.read().splitlines()
        lines = [ line for line in lines if line != ""]
    return lines

# Remove Gutenberg header
def remove_header(lines):
    indicators_list = ["*END", "ENDTHE"]
    start_line = len(lines)
    for i in range(len(lines)):
        if any(indicator in lines[i] for indicator in indicators_list): 
            start_line = i + 1;
            break
    if start_line == len(lines):
        print("starting indicator not found in file:" + read_path)
    lines = lines[start_line:len(lines)]
    return lines

# Fill in sentences (for testing data)
def fill_sentences(sentences, options):
    filled_sentences = []
    for i in range(len(sentences)):
        for j in range(options.shape[1]):
            sentence = sentences[i].replace("_____", options.ix[i,j])
            filled_sentences.append(sentence)
    return filled_sentences

# To lower case
def to_lowercase(lines):
    lines = [line.lower() for line in lines]
    return lines

# Check if string is number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Remove specfic texts
def process_specific_symbols(lines):
    exception_list = ['!', '-', '?', '.', '\'']
    punc_remove_list = [sym for sym in string.punctuation if sym not in exception_list]
    punc_replace_list = ['!', '?']
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\'s', " \'s")# perhaps there's a better solution
        lines[i] = lines[i].replace('n\'t', " n\'t")
        # Remove punctuations fom punc_remove_list
        for punc in punc_remove_list:
            lines[i] = lines[i].replace(punc, ' ')
        # Replace punctuations fom punc_replace_list
        for punc in punc_replace_list:
            lines[i] = lines[i].replace(punc, '.')
        # Replace numbers with N
        words = lines[i].split()
        words = ['N' if is_number(word) else word for word in words]
        lines[i] = " ".join(words)
    return lines

def filter_short_sentence(lines,threshold=5):
    lines = [ line for line in lines if len(line)>threshold] 
    return lines

# Split into sentences
def split_sentences(lines):
    lines = " ".join(lines)
    sentences = sent_tokenize(lines)
    return sentences

# Finalize each sentence by removing its punctuations
def finalize_sentences(sentences):
    punc_remove_list = ['.']
    for i in range(len(sentences)):
        for punc in punc_remove_list:
            sentences[i] = sentences[i].replace(punc, ' ')
    return sentences

def save_file(items, path):
    with open(path, "w") as f:
        for i in range(len(items)):
            f.write(items[i]+"\n")


if __name__ ==  "__main__":
    # Training data
    data_path = "data/"
    parsed_path = data_path + "Parsed_Training_Data/"
    sub_dirs = [parsed_path+"sentence/"]
    # Check if the output directory exists
    for p in sub_dirs:
        if not os.path.exists(p):
            os.makedirs(p)
    # Get list of training files
    train_path = data_path + "Holmes_Training_Data/"
    file_list = [f for f in os.listdir(train_path)]
    for i in range(len(file_list)):
        file = file_list[i]
        print("Processing file (", i+1, ", ", len(file_list),")...")
        loc_train = train_path + file
        loc_sentence = parsed_path + "sentence/" + file
        lines = read_file(loc_train)
        lines = remove_header(lines)
        lines = to_lowercase(lines)
        lines = process_specific_symbols(lines)
        sentences = split_sentences(lines)
        sentences = finalize_sentences(sentences)
        save_file(sentences,loc_sentence)

    # Testing data
    parsed_path = data_path + "parsed_testing_data.txt"
    test_path = data_path + "testing_data.csv"

    df = pd.read_csv(test_path)
    sentences = df.ix[:,1]
    options = df.ix[:,2:]
    
    sentences = fill_sentences(sentences, options)
    sentences = to_lowercase(sentences)
    sentences = process_specific_symbols(sentences)
    sentences = finalize_sentences(sentences)
    save_file(sentences, parsed_path)
