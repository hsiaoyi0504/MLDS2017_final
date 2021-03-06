from __future__ import print_function
from itertools import islice
import numpy as np
np.random.seed(1166)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Input, Flatten, Conv1D, MaxPooling1D, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import matthews_corrcoef, f1_score

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 512
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 100
SEARCH_THRESHOLD = True
ALL_TRAIN = True

unique_tags = set()
tags = []
texts = []
with open('./data/train_data.csv') as f:
    for line in islice(f,1,None):
        temp = line.rstrip('\n')
        temp = temp.split('"')
        tgs = temp[1].split(' ')
        for t in tgs:
            unique_tags.add(t)
        temp = temp[2:]
        text = ''
        for t in temp:
            text += t
        text = text.lstrip(',')
        tags.append(tgs)
        texts.append(text)

tags_dict = {}
reverse_tags_dict = {}
i = 0
for t in unique_tags:
    tags_dict[t] = i
    reverse_tags_dict[i] = t
    i += 1
labels = []
for tgs in tags:
    temp = np.zeros(len(tags_dict))
    for t in tgs:
        temp[tags_dict[t]] = 1
    labels.append(temp)
labels = np.asarray(labels)
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

if ALL_TRAIN:
    x_train = data
    y_train = labels
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
else:
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

texts_test = []
ids = []
with open('./data/test_data.csv') as f:
    for line in islice(f,1,None):
        temp = line.rstrip('\n')
        temp = temp.split(',')
        id = temp[0]
        text = ''
        temp = temp[1:]
        for t in temp:
            text += t
        ids.append(id)
        texts_test.append(text)
sequences_test = tokenizer.texts_to_sequences(texts_test)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

embeddings_index = {}
with open('./data/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

filter_lengths = [3,4,5]

t = []
for l in filter_lengths:
    c = Conv1D(128, l, activation='relu')(embedded_sequences)
    c = MaxPooling1D()(c)
    t.append(c)
x = concatenate(t , axis=1)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(len(tags_dict), activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001))

if ALL_TRAIN:
    model.fit(x_train, y_train, epochs=10, batch_size=32)
else:
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

y_train_preds = model.predict(x_train) 

if SEARCH_THRESHOLD:
    threshold = np.arange(0.1,0.9,0.05)
    acc = []
    accuracies = []
    best_threshold = np.zeros(y_train_preds.shape[1])
    for i in range(y_train_preds.shape[1]):
        y_prob = np.array(y_train_preds[:,i])
        for j in threshold:
            y_pred = [1 if prob>=j else 0 for prob in y_prob]
            acc.append( f1_score(y_train[:,i],y_pred) )
            acc = np.array(acc)
            index = np.where(acc==acc.max()) 
            accuracies.append(acc.max()) 
            best_threshold[i] = threshold[index[0][0]]
            acc = []

    y_test = model.predict(x_test)
    y_test = np.array([[1 if y_test[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
else:
    y_test = model.predict(x_test)
    y_test = np.array([[1 if y_test[i,j]>=0.5 else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])

labels_test = []
for y in y_test:
    temp = []
    for i in range(len(y)):
        if y[i] == 1:
            temp.append(reverse_tags_dict[i])
    labels_test.append(temp)
with open('submission.txt', 'w') as f:
    f.write('"id","tags"\n')
    for id, t in zip(ids, labels_test):
        f.write('"{}","{}"\n'.format(id," ".join(t)))
