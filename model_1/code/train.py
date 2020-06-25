from GRU_model import gru_model
from function import get_data, word2vec, get_dictionary, get_shuffle, get_batch
import os

total_nums = 200000
batch_size = 20
batch_num = total_nums // batch_size
epochs = 25
am = gru_model()
root = os.getcwd()
load_path = os.path.join(root,'RNN')
train_data, train_label, train2_data, test_data = get_data(load_path)
wordlist, wordvec =  get_dictionary()
shuffle_list = get_shuffle(train_data)

for k in range(epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    batch = get_batch(batch_size, shuffle_list, train_data, train_label, wordlist, wordvec)
    am.model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)

#am.model.save('GRU.h5')