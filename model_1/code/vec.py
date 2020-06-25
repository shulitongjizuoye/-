from tqdm import tqdm
import os
import numpy as np

def get_data():
    root = os.getcwd()
    load_path = os.path.join(root,'RNN')
    train2_label = np.load(os.path.join(load_path,'train2_label.npy'), allow_pickle=True)
    train2_label = train2_label.tolist()
    length = int(len(train2_label))
    train2_data = np.load(os.path.join(load_path,'train2_data.npy'), allow_pickle=True)
    train2_data = train2_data.tolist()
    train2_data = train2_data[:length]
    return train2_data ,length, load_path

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

def get_dictionary():
    root = os.getcwd()
    wordlist_path = os.path.join(root,'wordsList.npy')
    wordvec_path = os.path.join(root,'wordVectors.npy')
    wordlist = np.load(wordlist_path)
    wordlist = wordlist.tolist()
    wordlist = [word.decode('UTF-8') for word in wordlist]
    wordvec = np.load(wordvec_path)
    return  wordlist, wordvec


train2_data, total_nums,load_path = get_data()
wordlist, wordvec =  get_dictionary()
maxSeqLength = 35

data_list = []
label_list = []
for i in tqdm(range(total_nums)):
    data = word2vec(train2_data[i], maxSeqLength, wordlist, wordvec)
    data_list.append(data)
data_list = np.array(data_list)
data_list = data_list.reshape(total_nums,maxSeqLength, 50, -1)
np.save(os.path.join(load_path,'train2_vec'), data_list)