import os
import numpy as np
from keras.models import load_model
from tqdm import tqdm


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_data(load_path, batch_size):
    test_data = np.load(os.path.join(load_path,'test_data.npy'), allow_pickle=True)
    test_data = test_data.tolist()
    length = len(test_data)//batch_size
    length_2 = int(length*batch_size)
    test_data = test_data[:length_2]
    return test_data, length


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


def get_inputs_batch(batch_size, i, input_data, wordlist, wordvec, maxSeqLength=35):
    data_list = []
    begin = i * batch_size
    end = begin + batch_size
    for index in range(begin, end):
        index = int(index)
        data = word2vec(input_data[index], maxSeqLength, wordlist, wordvec)
        data_list.append(data)
    data_list = np.array(data_list)
    data_list = data_list.reshape(batch_size, maxSeqLength, 50, -1)
    return data_list

batch_size = 20
root = os.getcwd()
load_path = os.path.join(root,'RNN')
test_data, length = get_data(load_path, batch_size)
wordlist, wordvec =  get_dictionary()
model = load_model('GRU2.h5')
label_vec = np.array([0, 1])
label_list = []
for i in tqdm(range(length)):
    inputs = get_inputs_batch(batch_size, i, test_data, wordlist, wordvec)
    preds = model.predict_on_batch(inputs)
    score = preds.dot(label_vec)
    score = score.tolist()
    for achievement in score:
        if achievement>0.45:
            label = 1
        else:
            label = 0
        label_list.append(label)
label_array = np.array(label_list, dtype = int)
label_length  = len(label_array)

id = np.arange(label_length)
np.save(os.path.join(load_path,'test_label'), label_array)
np.save(os.path.join(load_path,'id'), id)


