from keras.models import load_model, Model, Sequential
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.activations import *
from keras.layers.convolutional import Conv2D, Conv1D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import *
from keras.layers.merge import *
from keras.layers.embeddings import Embedding
from keras.optimizers import *
from keras.initializers import *
from keras.utils import plot_model, to_categorical
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import *
from keras.preprocessing import sequence
import h5py
import reader
from model_keras import *

def generator(data, batch_size, sentence_length, vocab_size):
	data_len = data.shape[0]
	batch_len = data_len // batch_size
	data = np.reshape(data[0 : batch_size * batch_len],	[batch_size, batch_len])
	epoch_size = (batch_len - 1) // sentence_length
	batch_sentences = np.zeros((batch_size, sentence_length))
	batch_targets = np.zeros((batch_size, sentence_length, vocab_size))
	index = 0
	while True:
		batch_sentences[:] = data[:,index:index+sentence_length]
		for i in range(batch_size):
			batch_targets[i] = to_categorical(data[i,index+1:index+1+sentence_length],vocab_size)
		index += sentence_length		
		if index >= batch_len-sentence_length-1:
			index = 0
		yield batch_sentences, batch_targets

def generator_cnn(data, batch_size, sentence_length, vocab_size):
	batch_sentences = np.zeros((batch_size, sentence_length))
	batch_targets = np.zeros((batch_size, sentence_length, vocab_size))
	index=0
	while True:		
		for i in range(batch_size):
			batch_sentences[i] = data[index,:-1]
			batch_targets[i] = to_categorical(data[index,1:],vocab_size)
			index += 1		
			if index == data.shape[0]-1:
				index = 0
		yield batch_sentences, batch_targets


"""Small config."""
mode_cnn = True
init_scale = 0.1
learning_rate = 0.001
max_grad_norm = 0.1
num_layers = 2
# num_steps = 20
hidden_size = 256
max_epoch = 3
max_max_epoch = 8
keep_prob = 1.0
lr_decay = 0.5
batch_size = 20
vocab_size = 20000
data_path = "./data/"
model_path = "./model/"
assert os.path.exists(data_path)
assert os.path.exists(model_path)


train_data, test_data, word_to_id, chose_len = reader.Holmes_raw_data(data_path, vocab_size)
num_steps = chose_len
if mode_cnn:
	train_data = sequence.pad_sequences(train_data, maxlen=num_steps+1, dtype='int32',
    	padding='post', truncating='post', value=1.)
else:
	train_data = np.array(train_data)
test_data = np.array(test_data)

size = hidden_size
filter_sizes = [1,2,3,4,5]
if mode_cnn:
	model = cnn_model(size, vocab_size, num_steps, filter_sizes)
	opt=SGD(lr=learning_rate, momentum=0.99, decay=0.0, nesterov=True, clipnorm=max_grad_norm)
else:	
	model = rnn_model(size, vocab_size, num_steps, batch_size)
	opt=SGD(lr=learning_rate, clipnorm=max_grad_norm)
model.compile(loss='categorical_crossentropy', optimizer=opt, sample_weight_mode='temporal')
model.summary()

checkpointer = ModelCheckpoint(
						filepath=model_path+"RNN_20170612.hdf5",
						monitor="loss",
						mode="min",
						verbose=0,
						save_best_only=True)

loss = []
save_loss_callback = LambdaCallback(
    on_batch_end=lambda batch, logs: loss.append(logs['loss']))

def scheduler(epoch):
	if epoch < max_epoch:
		return learning_rate
	else:
		return learning_rate*(0.5**(epoch-max_epoch+1))
if mode_cnn:
	steps_per_epoch = ((train_data.shape[0] // batch_size) - 1)
	generator = generator_cnn(train_data,batch_size,num_steps,vocab_size)
else:
	steps_per_epoch = ((train_data.shape[0] // batch_size) - 1) // num_steps
	generator = generator(train_data,batch_size,num_steps,vocab_size)

for i in range(max_max_epoch):
	K.set_value(model.optimizer.lr, scheduler(i))
	model.fit_generator(generator,
						steps_per_epoch=steps_per_epoch, 
						epochs=1, 
						verbose=1, 
						callbacks=[checkpointer, save_loss_callback])
	np.save("loss.npy",loss)
	# model.reset_states()

#----------
# Testing
#----------
wordlen_list = reader.get_wordlen_list(data_path, word_to_id)
wordlen_list = wordlen_list[0:5200]

def cal_cost_list(mode_cnn, batch_size, test_data, num_steps, vocab_size, wordlen_list, model):
	cost_list = np.zeros((5200))
	total_sentences = 5200
	batch_len = test_data.shape[0] // total_sentences
	test_data = np.reshape(test_data[0 : total_sentences * batch_len], [total_sentences, batch_len])
	# Stateful == False
	if mode_cnn:
		for i in range(total_sentences):
			test = test_data[i,:num_steps]
			target = to_categorical(test_data[i,1:1+num_steps],vocab_size)
			valid = np.zeros((1,num_steps))
			valid[:,:wordlen_list[i]] = 1.0
			cost_list[i] = model.evaluate(test.reshape((-1,num_steps)),
										target.reshape((-1,num_steps,vocab_size)),
										verbose=0,
										sample_weight=valid)
	# Stateful == True
	else:
		for i in range(total_sentences):
			test_dummy = np.zeros((batch_size, num_steps))
			test_dummy[0,:] = test_data[i,:num_steps]			
			target = to_categorical(test_data[i,1:1+num_steps],vocab_size)
			target_dummy = np.zeros((batch_size, num_steps, vocab_size))
			target_dummy[0,:,:] = target
			valid = np.zeros((batch_size,num_steps))
			valid[0,:wordlen_list[i]] = 1.0
			cost_list[i] = model.evaluate(test_dummy,
										target_dummy,
										verbose=0,
										batch_size=batch_size,
										sample_weight=valid)
			model.reset_states()
	return cost_list
cost_list = cal_cost_list(mode_cnn, batch_size, test_data, num_steps, vocab_size, wordlen_list, model)
cost_list = cost_list.reshape((1040, 5))

ans = np.argmin(cost_list, axis=1)
print("\nWriting prediction to \"" + data_path + "submission.csv\"\n")
with open(data_path + "submission.csv", 'w') as outfile:
	outfile.write("id,answer\n")
	for idx in range(1040):
		outfile.write(str(idx+1)+",")
		if ans[idx]==0:
			outfile.write("a\n")
		elif ans[idx]==1:
			outfile.write("b\n")
		elif ans[idx]==2:
			outfile.write("c\n")
		elif ans[idx]==3:
			outfile.write("d\n")
		else:
			outfile.write("e\n")