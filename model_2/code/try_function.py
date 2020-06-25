from keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd
import pickle


def get_data():
    path = os.path.join(os.getcwd(),'model','tokenizer_lstm.pickle')
    tok = pickle.load(open(path, 'rb'))
    train = os.path.join(os.getcwd(),'RNN','training_label.txt')
    maxlen = 50
    df_train = pd.read_csv(train,header=None, delimiter='\n')
    df_train.columns = ['raw']
    df_train['x'] = df_train['raw'].apply(lambda x: x.split(' +++$+++ ')[1])
    df_train['y'] = df_train['raw'].apply(lambda x: x.split(' +++$+++ ')[0])
    x_train = df_train['x'].values
    y_train = df_train['y'].values
    x = tok.texts_to_sequences(x_train)
    x = pad_sequences(x, maxlen=maxlen, padding='pre')
    return x, y_train
