import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd
import pickle



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_data():
    path = os.path.join(os.getcwd(),'model','tokenizer_lstm.pickle')
    tok = pickle.load(open(path, 'rb'))
    test = os.path.join(os.getcwd(),'RNN','testing_data.txt')
    df_test = pd.read_csv(test,header=None, delimiter='\n')
    df_test.columns = ['raw']
    df_test['x'] = df_test['raw'].apply(lambda x: x.split(',')[1])

    maxlen = 50
    x_test = df_test['x'].values[1:]
    x = tok.texts_to_sequences(x_test)
    x = pad_sequences(x, maxlen=maxlen, padding='pre')
    return x


batch_size = 50
test_data = get_data()
print (test_data.shape)
model = load_model('GRU_2.h5')
y = model.predict_classes(test_data)
np.save(os.path.join(os.getcwd(),'RNN','test_label'), y)

