from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.callbacks import Callback
import numpy as np

### input data: Spectrogram size (5511, 101)
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

# Load preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")

# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


class myCallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.60):
      print("\nReached 60% accuracy, stop training!")
      self.model.stop_training = True

callback = myCallback()

def mainModel(input_shape):
    X_input = Input(shape=input_shape)

    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = LSTM(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    X = LSTM(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    X = TimeDistributed(Dense(units=1, activation='sigmoid'))(X)

    model = Model(inputs=X_input, outputs=X)

    return model


model = mainModel(input_shape=(Tx, n_freq))
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X, Y, batch_size=5, epochs=30, callbacks=[callback])
model.save_weights('./models/model.h5')
loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)
