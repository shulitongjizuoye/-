
from keras.models import load_model
import os
from try_function2 import get_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = load_model('GRU_1.h5')
train_data, train_label = get_data()
num_words = 20000
batch_size=500
epochs=5
model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('GRU_2.h5')