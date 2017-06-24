from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(0)

import string
import sys
from collections import Counter
import argparse
import six

import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers import GRU, Convolution1D, MaxPooling1D, Flatten, Merge, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--op', type=str, default="train", help='train/test')
parser.add_argument('--restore', type=bool, default=False, help='Restore model or not')
args = parser.parse_args()

train_path = "data/train_data.csv"
test_path = "data/test_data.csv"
output_path = "data/output.csv"
model_path = "model/RNN/rnn.hdf5"
tokenizer_path = "model/RNN/tokenizer.pkl"

#####################
###   parameter   ###
#####################
CONFIG = 2
split_ratio = 0.1
if CONFIG == 7:
    embedding_dim = 300
else:
    embedding_dim = 100
nb_epoch = 1000
batch_size = 128

################
###   Util   ###
################
def read_data(path, training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#########################
###   Main function   ###
#########################
def main():
    ### read training and testing data
    (Y_data, X_data, tag_list) = read_data(train_path, True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = X_data + X_test
    print('Find %d articles.' %(len(all_corpus)))
    
    if args.restore or (args.op == "test"):
        print('Load tokenizer...')
        tokenizer = six.moves.cPickle.load(open(tokenizer_path))
    else:
        ### tokenizer for all data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_corpus)
        print('Saving tokenizer to ', tokenizer_path)
        six.moves.cPickle.dump(tokenizer, open(tokenizer_path, "wb"))
    word_index = tokenizer.word_index

    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    print ('Padding sequences.')
    train_sequences = pad_sequences(train_sequences)
    max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
    ###
    train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
    
    ### get mebedding matrix from glove
    print('Get embedding dict from glove.')
    embedding_dict = get_embedding_dict('data/glove.6B.%dd.txt' % embedding_dim)
    print('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    print('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    if args.restore or (args.op == "test"):
        model = load_model(model_path, custom_objects={'f1_score':f1_score})
    else:
        CONFIG = 2
        ### build model from scratch
        print('Building model.')

        model = Sequential()
        model.add(Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_article_length,
                            trainable=False))
        dropout_prob = 0.3
        if CONFIG == 1:
            model.add(GRU(128, return_sequences=True, activation='tanh', recurrent_dropout=dropout_prob, dropout=dropout_prob))
            model.add(GRU(128, return_sequences=True, activation='tanh', recurrent_dropout=dropout_prob, dropout=dropout_prob))
            model.add(GRU(128, activation='tanh',recurrent_dropout=dropout_prob, dropout=dropout_prob))
        elif CONFIG == 2 or CONFIG == 3:
            model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh', recurrent_dropout=dropout_prob, dropout=dropout_prob)))
            model.add(Bidirectional(LSTM(128, activation='tanh', recurrent_dropout=dropout_prob, dropout=dropout_prob)))
        elif CONFIG == 4:
            model.add(Bidirectional(LSTM(128, activation='tanh',recurrent_dropout=dropout_prob, dropout=dropout_prob)))
        elif CONFIG == 5:
            model.add(Bidirectional(LSTM(256, return_sequences=True, activation='tanh', recurrent_dropout=dropout_prob, dropout=dropout_prob)))
            model.add(Bidirectional(LSTM(256, activation='tanh', recurrent_dropout=dropout_prob, dropout=dropout_prob)))
        else: # CONFIG == 6 or CONFIG == 7
            model.add(Bidirectional(LSTM(512, return_sequences=True, activation='tanh', recurrent_dropout=dropout_prob, dropout=dropout_prob)))
            model.add(Bidirectional(LSTM(512, activation='tanh', recurrent_dropout=dropout_prob, dropout=dropout_prob)))

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout_prob))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_prob))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dropout_prob))
        model.add(Dense(38, activation='sigmoid'))
        model.summary()
    if CONFIG == 1 or CONFIG == 2:
        optimizer = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
    else: # CONFIG == 3 or CONFIG == 4 or CONFIG == 5 or CONFIG == 6 or CONFIG == 7
        optimizer = RMSprop()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[f1_score])
    
    if args.op == "train":
        earlystopping = EarlyStopping(monitor='val_f1_score', patience=20, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(filepath=model_path,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_f1_score',
                                     mode='max')
        hist = model.fit(X_train, Y_train, 
                         validation_data=(X_val, Y_val),
                         epochs=nb_epoch, 
                         batch_size=batch_size,
                         callbacks=[earlystopping,checkpoint])

    elif args.op == "test":
        Y_pred = model.predict(test_sequences)
        thresh = 0.4
        with open(output_path,'w') as output:
            print('\"id\",\"tags\"',file=output)
            Y_pred_thresh = (Y_pred > thresh).astype('int')
            for index,labels in enumerate(Y_pred_thresh):
                labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
                labels_original = ' '.join(labels)
                print('\"%d\",\"%s\"'%(index,labels_original),file=output)

    else:
        print("invalid op")


if __name__=='__main__':
    main()
