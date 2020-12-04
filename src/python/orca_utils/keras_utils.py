import logging

import tensorflow as tf

from tensorflow.keras import layers

tf.get_logger().setLevel('INFO')

def configure_model(model_info, lstm_type='', optimizer = tf.compat.v1.train.AdamOptimizer(0.001)):

    '''

    :param input_size:
    :param n_classes:
    :param layers:
    :param lstm_type:
    :param optimizer:
    :param CD: concatenated depth
    :return:
    '''

    model = tf.keras.Sequential()
    model.add(layers.Masking(mask_value=1., input_shape=(None, model_info.feat_size)))

    for l, layer in enumerate(model_info.layers):
        if l == 0:
            if lstm_type == 'b':
                logging.info('Using bidirectional LSTM')
                model.add(layers.Bidirectional(layers.LSTM(layer, input_shape=(None, model_info.feat_size), dropout=0.1, return_sequences=True, recurrent_dropout=0.1)))
            else:
                model.add(layers.LSTM(layer, input_shape=(None, model_info.feat_size), dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
        else:
            model.add(layers.TimeDistributed(layers.Dense(layer,activation='relu')))
            model.add(layers.Dropout(0.1))

        model.add(layers.TimeDistributed(layers.Dense(model_info.n_classes,activation='softmax')))

    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model

def clear_memory():

    tf.keras.backend.clear_session()

def debug_device():

    tf.debugging.set_log_device_placement(True)