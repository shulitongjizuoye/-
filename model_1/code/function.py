import os
import numpy as np
import random
from keras.utils import np_utils


def get_data(load_path):
    train_data = np.load(os.path.join(load_path,'train_data.npy'), allow_pickle=True)
    train_data = train_data.tolist()
    train2_data = np.load(os.path.join(load_path,'train2_data.npy'), allow_pickle=True)
    train2_data = train2_data.tolist()
    train_label = np.load(os.path.join(load_path,'train_label.npy'), allow_pickle=True)
    train_label = train_label.tolist()
    test_data = np.load(os.path.join(load_path,'test_data.npy'), allow_pickle=True)
    test_data = test_data.tolist()
    return train_data, train_label,train2_data, test_data

def word2vec(inputs_list, maxSeqLength, wordlist, wordvec):
    vec_matrix = np.zeros((maxSeqLength, 50), dtype='float32')
    num = int(maxSeqLength)
    if len(inputs_list)>num:
        inputs_list = inputs_list[:num]
    inputs_length = len(inputs_list)
    indexCounter = num - int(inputs_length)
    for word in inputs_list:
        try:
            index = wordlist.index(word)
        except ValueError:
            index = 399999
        vec_matrix[indexCounter, :] = wordvec[index]
        indexCounter = indexCounter + 1
    return vec_matrix

def get_shuffle(train_data):
    shuffle_list = [i for i in range(len(train_data))]
    random.shuffle(shuffle_list)
    return shuffle_list

def get_dictionary():
    root = os.getcwd()
    wordlist_path = os.path.join(root,'wordsList.npy')
    wordvec_path = os.path.join(root,'wordVectors.npy')
    wordlist = np.load(wordlist_path)
    wordlist = wordlist.tolist()
    wordlist = [word.decode('UTF-8') for word in wordlist]
    wordvec = np.load(wordvec_path)
    return  wordlist, wordvec

def get_batch(batch_size, shuffle_list, train_data, train_label, wordlist, wordvec, maxSeqLength=35):
    data_length = len(train_data)
    # for i in range(2):
    for i in range(data_length//batch_size):
        data_list = []
        label_list = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        for index in sub_list:
            index = int(index)
            data = word2vec(train_data[index], maxSeqLength, wordlist, wordvec)
            label = train_label[index]
            data_list.append(data)
            label_list.append(label)
        data_list = np.array(data_list)
        data_list = data_list.reshape(batch_size, maxSeqLength, 50, -1)
        label_list = np.array(label_list)
        label_list = np_utils.to_categorical(label_list,num_classes=2)
        yield data_list, label_list
