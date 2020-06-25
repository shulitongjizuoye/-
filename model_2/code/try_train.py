from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import os
from keras.optimizers import Adam
from keras.layers import Dense, Dropout,Bidirectional
from keras.layers import GRU
from try_function import get_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_data, train_label = get_data()
num_words = 20000

model = Sequential()
model.add(Embedding(num_words, 128))
model.add(Dropout(0.2))
model.add(Dense(128,  activation='relu'))
model.add(Bidirectional(GRU(128,  kernel_initializer='he_normal', return_sequences=True)))
model.add(Bidirectional(GRU(128, dropout=0.5, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
opt = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

batch_size=500
epochs=5

model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('GRU_1.h5')