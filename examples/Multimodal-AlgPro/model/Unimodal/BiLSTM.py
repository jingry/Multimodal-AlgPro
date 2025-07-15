from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional
from keras import optimizers



lstm_output_size = 256
hidden_dims = 256



print('Building model...')
model = Sequential()
model.add(Reshape((1000,20),input_shape=(20000,)))
model.add(Bidirectional(LSTM(lstm_output_size,return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


