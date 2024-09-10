import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Input, Conv1D, MaxPooling1D, concatenate, Dense, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)



def inception_module(input_tensor, filters):

    # 1x1 convolution
    conv_1x1 = Conv1D(filters, 1, padding='same', activation='relu')(input_tensor)

    # 3x3 convolution
    conv_3x3 = Conv1D(filters, 3, padding='same', activation='relu')(input_tensor)

    # 5x5 convolution
    conv_5x5 = Conv1D(filters, 5, padding='same', activation='relu')(input_tensor)

    # MaxPooling followed by 1x1 convolution
    max_pool = MaxPooling1D(3, strides=1, padding='same')(input_tensor)
    max_pool = Conv1D(filters, 1, padding='same', activation='relu')(max_pool)

    # Concatenate all the layers
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)

    return output

def inceptDecoder(input_shape, output_shape):
    input_layer = Input(shape=input_shape)

    # Inception Module 1
    x = inception_module(input_layer, 64)

    # GRU Layer(s)
    x = GRU(128, return_sequences=True, activation='relu')(x)
    x = GRU(256, return_sequences=True, activation='relu')(x)
    x = GRU(512, return_sequences=False, activation='relu')(x)

    # Reshape the output to add a time dimension of 1 (necessary for Conv1D)
    x = Reshape((1, 512))(x)

    # Inception Module 2
    x = inception_module(x, 128)

    # Flatten the output from Inception modules and GRU layers
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    output_layer = Dense(output_shape, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model




class NeuroInceptDecoder:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        

    def inception_module(self, input_tensor, filters):
        conv_1x1 = Conv1D(filters, 1, padding='same', activation='relu')(input_tensor)
        conv_3x3 = Conv1D(filters, 3, padding='same', activation='relu')(input_tensor)
        conv_5x5 = Conv1D(filters, 5, padding='same', activation='relu')(input_tensor)

        max_pool = MaxPooling1D(3, strides=1, padding='same')(input_tensor)
        max_pool = Conv1D(filters, 1, padding='same', activation='relu')(max_pool)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)

        return output

    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        # Inception Module 1
        x = self.inception_module(input_layer, 64)

        # GRU Module
        x = GRU(128, return_sequences=True, activation='relu')(x)
        x = GRU(256, return_sequences=True, activation='relu')(x)
        x = GRU(512, return_sequences=False, activation='relu')(x)

        x = Reshape((1, 512))(x)

        # Inception Module 2
        x = self.inception_module(x, 128)

        x = Flatten()(x)

        # Fully Connected Layers
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)

        output_layer = Dense(self.output_shape, activation='linear')(x)
        model = Model(inputs=input_layer, outputs=output_layer)

        return model




def FCN(input_shape, output_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)  # Flatten layer before the dense blocks
    
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(output_shape, activation='linear')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

def CNN(input_shape, output_shape):
    input_shape = (input_shape[0], input_shape[1], 1)
    inputs = tf.keras.Input(shape=input_shape)  # Adjust input shape as needed

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(32, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.25)(x)
    x = Dropout(0.3)(x)

    # Flatten and Dense layer
    x = Flatten()(x)
    x = Dense(output_shape)(x)

    model = Model(inputs=inputs, outputs=x)

    return model
