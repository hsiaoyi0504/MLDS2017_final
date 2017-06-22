from keras.models import load_model, Model, Sequential
from keras.layers import Input,merge, Bidirectional, TimeDistributed
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
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

learning_rate = 1.0
max_grad_norm = 5
keep_prob = 1.0
lr_decay = 0.5

def rnn_model(hidden_size, vocab_size, num_steps):
	embedding_layer = Embedding(vocab_size, hidden_size, input_length=num_steps)
	sequence_input = Input(shape=(num_steps,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)

	x = Bidirectional(LSTM(hidden_size//2,return_sequences=True,implementation=2))(embedded_sequences)
	x = Bidirectional(LSTM(hidden_size//2,return_sequences=True,implementation=2))(x)
	softmax_layer = TimeDistributed(Dense(vocab_size,activation='softmax'))(x)

	model=Model(sequence_input, softmax_layer)
	adam=SGD(lr=learning_rate)
	model.compile(loss='categorical_crossentropy', optimizer=adam, sample_weight_mode='temporal')
	return model


def cnn_model(hidden_size, vocab_size, num_steps, filter_sizes):
	num_cnn_filters = hidden_size//len(filter_sizes)

	embedding_layer = Embedding(vocab_size, hidden_size, input_length=num_steps)
	sequence_input = Input(shape=(num_steps,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)

	cnn_outputs = []
	for i, filter_size in enumerate(filter_sizes):
		x = Conv1D(num_cnn_filters, filter_size, activation='relu', padding='same')(embedded_sequences)
		cnn_outputs.append(x)	
	concat = concatenate(cnn_outputs, axis=-1)
	softmax_layer = Conv1D(vocab_size, 1, activation='softmax',padding='valid')(concat)

	model=Model(sequence_input, softmax_layer)
	adam=Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=adam, sample_weight_mode='temporal')

	return model