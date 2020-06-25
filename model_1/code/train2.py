import os
import numpy as np
import random
from keras.utils import np_utils
from keras.models import load_model
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_data():
    root = os.getcwd()
    load_path = os.path.join(root,'RNN')
    train2_label = np.load(os.path.join(load_path,'train2_label.npy'), allow_pickle=True)
    train2_data = np.load(os.path.join(load_path,'train2_vec.npy'), allow_pickle=True)
    train2_label = train2_label.tolist()
    length = int(len(train2_label))
    shuffle_list = [i for i in range(length)]
    random.shuffle(shuffle_list)
    return train2_data ,train2_label, shuffle_list, length

def get_batch(batch_size, shuffle_list, inputs_data, inputs_label, batch_num, maxSeqLength=35):
    #for i in range (5):
    for i in range(batch_num):
        data_list = np.zeros((batch_size, maxSeqLength, 50, 1))
        label_list = []
        begin = i * batch_size
        end = begin + batch_size
        sub_list = shuffle_list[begin:end]
        count_num = 0
        for index in sub_list:
            index = int(index)
            data_list[count_num,:,:,:] = inputs_data[index]
            label = inputs_label[index]
            label_list.append(label)
            count_num = count_num+1
        label_list = np.array(label_list)
        label_list = np_utils.to_categorical(label_list,num_classes=2)
        yield data_list, label_list

train2_data, train2_label, shuffle_list, total_nums = get_data()
am = load_model('GRU.h5')
batch_size = 25
#batch_num = 5
batch_num = total_nums // batch_size
epochs =10
for k in range(epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    batch = get_batch(batch_size, shuffle_list, train2_data, train2_label,batch_num)
    am.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)
am.save('GRU2.h5')