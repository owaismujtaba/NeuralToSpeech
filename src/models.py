import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def model_Seq(input_shape, output_shape):
    model = Sequential()

     # GRU Layer(s)
    #model.add(GRU(128, input_shape=(input_shape[1], 1), return_sequences=True, activation='relu'))
    #model.add(GRU(256, return_sequences=True, activation='relu'))
    #model.add(GRU(512, return_sequences=False, activation='relu'))  
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
