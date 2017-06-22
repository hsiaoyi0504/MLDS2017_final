from keras.models import load_model, Model, Sequential
from keras.layers import Input,merge, Bidirectional, TimeDistributed
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,Masking
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.activations import *
from keras.layers.convolutional import Conv2D, Conv1D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.recurrent import *
from keras.layers.normalization import *
from keras.layers.merge import *
from keras.layers.embeddings import Embedding
from keras.optimizers import *
from keras.initializers import *


def rnn_model(hidden_size, vocab_size, num_steps, batch_size):
	embedding_layer = Embedding(vocab_size, hidden_size, input_length=num_steps)
	sequence_input = Input(batch_shape=(batch_size,num_steps), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)

	x = LSTM(hidden_size,return_sequences=True,stateful=True,implementation=2)(embedded_sequences)
	x = LSTM(hidden_size,return_sequences=True,stateful=True,implementation=2)(x)
	softmax_layer = TimeDistributed(Dense(vocab_size,activation='softmax'))(x)

	model=Model(sequence_input, softmax_layer)

	return model


def cnn_model(hidden_size, vocab_size, num_steps, filter_sizes):

	mask = Masking(mask_value=1., input_shape=(num_steps,))
	embedding_layer = Embedding(vocab_size, hidden_size, input_length=num_steps)
	sequence_input = Input(shape=(num_steps,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)

	# cnn_outputs = []
	x = Conv1D(512, 3, activation='relu', padding='same')(embedded_sequences)
	for i in range(7):
		x = res_block(x,512,3)
	# concat = concatenate(cnn_outputs, axis=-1)
	softmax_layer = Conv1D(vocab_size, 1, activation='softmax',padding='valid')(x)

	model=Model(sequence_input, softmax_layer)

	return model

def res_block(input_layer,ndf,w):
	e = input_layer
	linear1 = Conv1D(ndf/4, 1, activation='linear', padding='same')(e)
	gate1 = Conv1D(ndf/4, 1, activation='sigmoid', padding='same')(e)
	e = multiply([linear1,gate1])
	linear2 = Conv1D(ndf/4, w, activation='linear', padding='same')(e)
	gate2 = Conv1D(ndf/4, w, activation='sigmoid', padding='same')(e)
	e = multiply([linear2,gate2])
	linear3 = Conv1D(ndf, 1, activation='linear', padding='same')(e)
	gate3 = Conv1D(ndf, 1, activation='sigmoid', padding='same')(e)
	e = multiply([linear3,gate3])
	return add([e,input_layer])























