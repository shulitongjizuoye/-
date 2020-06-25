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
    train = os.path.join(os.getcwd(),'RNN','training_nolabel.txt')
    maxlen = 50
    df_no_label = pd.read_csv(train,header=None, delimiter='\n')
    df_no_label.columns = ['raw']
    df_no_label['x'] = df_no_label['raw'].apply(lambda x: x)
    x_train = df_no_label['x'].values
    x = tok.texts_to_sequences(x_train)
    x = pad_sequences(x, maxlen=maxlen, padding='pre')
    return x


batch_size = 50
train2_data = get_data()
print (train2_data.shape)
model = load_model('GRU_1.h5')
y = model.predict_classes(train2_data)
np.save(os.path.join(os.getcwd(),'RNN','train2_label'), y)



