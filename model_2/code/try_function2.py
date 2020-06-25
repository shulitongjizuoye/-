from keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd
import pickle
import numpy as np


def get_data():
    label = os.path.join(os.getcwd(),'RNN','train2_label.npy')
    y_train = np.load(label)
    path = os.path.join(os.getcwd(), 'model', 'tokenizer_lstm.pickle')
    tok = pickle.load(open(path, 'rb'))
    train = os.path.join(os.getcwd(), 'RNN', 'training_nolabel.txt')
    maxlen = 50
    df_no_label = pd.read_csv(train, header=None, delimiter='\n')
    df_no_label.columns = ['raw']
    df_no_label['x'] = df_no_label['raw'].apply(lambda x: x)
    x_train = df_no_label['x'].values
    x = tok.texts_to_sequences(x_train)
    x = pad_sequences(x, maxlen=maxlen, padding='pre')
    return x, y_train
