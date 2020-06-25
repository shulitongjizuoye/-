from keras.layers import Input, Reshape, Dense, Dropout
from keras.layers import GRU
from keras.layers import add
from keras.optimizers import Adam
from keras.models import Model
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def bi_gru(units, x):
    x = Dropout(0.2)(x)
    y1 = GRU(units, kernel_initializer='he_normal', return_sequences=True)(x)
    y2 = GRU(units, kernel_initializer='he_normal', return_sequences=True, go_backwards=True)(x)
    y = add([y1, y2])
    return y

def dense(units, x, activation='relu'):
    x = Dropout(0.2)(x)
    y =Dense(units, activation=activation, kernel_initializer='he_normal')(x)
    return y

class gru_model():
    def __init__(self):
        super(gru_model, self).__init__()
        self.label_size = 2
        self._model_init()
        self.opt_init()

    def _model_init(self):
        self.inputs = Input(shape=(35, 50,1))#输入
        x = Reshape((-1, 50))(self.inputs)#reshape成N*50
        x = dense(128, x)
        x = bi_gru(128, x)
        x = bi_gru(128, x)
        x = dense(128, x)
        self.outputs = GRU(self.label_size, activation='softmax', kernel_initializer='he_normal')(x)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)


    def opt_init(self):
        opt = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# am = gru_model()
# am.model.summary()