# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing Holmes text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import codecs
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", " <eos> ").split()

def _build_vocab(train_path, test_path, vocab_size):
  # build the dict based on testing data
  words = _read_words(test_path)
  counter = collections.Counter(words)
  count_pairs = sorted(counter.items(), key=lambda x:(-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  dictionary = dict(zip(words, range(len(words))))


  # extend the dict based on training data
  words = _read_words(train_path)
  count = collections.Counter(words).most_common(vocab_size)
  dictionary_appendage = dict()
  for word, _ in count:
    if (len(dictionary) + len(dictionary_appendage)) == vocab_size:
      break
    if word not in dictionary:
      dictionary_appendage[word] = len(dictionary) + len(dictionary_appendage)
  dictionary.update(dictionary_appendage)

  return dictionary


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  """
  for i in range(len(data)):
    if data[i] not in word_to_id:
      data[i] = "<unk>"
  """
  return [word_to_id[word] for word in data if word in word_to_id]

def _aggregate_data(data_path=None):
  file_list = [ data_path+"Parsed_Training_Data/sentence/"+f for f in os.listdir(data_path+"Parsed_Training_Data/sentence/")]
  # split data based on your preference
  train_list = file_list

  # Include testing sentences as training data
  train_list.extend([data_path+"parsed_testing_data.txt"])
  def write_list_to_file(file_list, data_path, file_name):
    with open(data_path + file_name, 'w') as outfile:
      for fname in file_list:
        with open(fname) as infile:
          for line in infile:
            outfile.write(line)
            
  write_list_to_file(train_list, data_path, 'holmes.train.txt')

# This function fills testing data into equal-lengthed sentences
# the reason why this is not dealt with during parse_data.py
# is that the length(chose_len) is actually closely connected
# with the testing phase
def fill_file2batch(data_path=None):
    with open(data_path + "parsed_testing_data.txt", 'r') as f:
      lines = f.readlines()
    max_len = 0
    for i in range(len(lines)):
        if len(lines[i].split()) > max_len:
            max_len = len(lines[i].split())
    #print("max length of sentences = ", max_len)
    chose_len = max_len + 1
    #print("choosing len = ", chose_len)

    filled_lines = []
    for i in range(len(lines)):
        words = lines[i].split()
        slots_needed = chose_len - len(words)
        words.extend(["<eos>" for i in range(slots_needed)])
        filled_lines.append(" ".join(words))
    
    with open(data_path + "batch_parsed_testing_data.txt", "w") as f:
        for i in range(len(filled_lines)):
            f.write(filled_lines[i]+"\n")

    return chose_len

def Holmes_raw_data(data_path=None, vocab_size=10000):
  _aggregate_data(data_path)
  train_path = os.path.join(data_path, "holmes.train.txt")
  test_path = os.path.join(data_path, "parsed_testing_data.txt")
  chose_len = fill_file2batch(data_path)
  filled_test_path = os.path.join(data_path, "batch_parsed_testing_data.txt")

  word_to_id = _build_vocab(train_path, test_path, vocab_size)
  train_data = _file_to_word_ids(train_path, word_to_id)
  test_data = _file_to_word_ids(filled_test_path, word_to_id)
  
  return train_data, test_data, word_to_id, chose_len

def Holmes_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw Holmes data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from Holmes_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "HolmesProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y

def get_wordlen_list(data_path, word_to_id):
  file_words = _read_words(data_path + "parsed_testing_data.txt")
  file_lines = ' '.join(file_words)
  lines = file_lines.split("<eos>")

  wordlen_list = []
  for i in range(len(lines)):
    words = lines[i].split()
    word_ids = [word_to_id[word] for word in words if word in word_to_id]
    wordlen_list.append(len(word_ids))
  return wordlen_list
