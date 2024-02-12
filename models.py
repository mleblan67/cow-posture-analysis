from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D, LSTM
from tensorflow.keras.layers import MaxPooling1D


'''
Pure CNN model from project
Runs two convolutions on the data
'''
def CNN(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu',
              input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    # model.add(Conv1D(filters=128, kernel_size=4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model

'''
CNN-LSTM model
Runs two convolutions on the data then
passes the filtered data into an LSTM for time sensitive classification
'''
def CNN_LSTM(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu',
              input_shape=(n_timesteps, n_features)))
    # model.add(Dropout(0.2))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))

    model.add(LSTM(4, dropout=0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    
    return model
