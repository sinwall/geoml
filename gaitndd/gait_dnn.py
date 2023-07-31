# # uncomment if in colab environment
# from google.colab import drive
# import os
# drive.mount('/content/drive')
# os.chdir('/content/drive/MyDrive/laboratory/geoml')
# !git pull

# !pip install wfdb

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

import os, sys
sys.path.append(os.path.abspath('./'))
from gaitndd.loader import load_data
from tools import Embedder, SineFilter, calculate_weighting_vectors


def get_model(n_classes, ts_shape=(300, 2), ts_only=True, xw_size=(128, )):
    l2_coef = 1e-2
    inputs_ts = keras.Input(ts_shape)
    inputs_xw = keras.Input(xw_size)
    layers_ts = keras.Sequential([
        keras.layers.LSTM(
            units=32,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2_coef)
        ),
        keras.layers.LSTM(
            units=32,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2_coef)
        ),
        keras.layers.Conv1D(
            filters=64,
            kernel_size=8,
            strides=2,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(l2_coef)
        ),
        keras.layers.MaxPool1D(
            pool_size=2,
            strides=2
        ),
        keras.layers.Conv1D(
            filters=128,
            kernel_size=4,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(l2_coef)
        ),
        # keras.layers.MaxPool1D(
        #     pool_size=4,
        #     strides=4
        # ),
        # keras.layers.Conv1D(
        #     filters=16,
        #     kernel_size=16,
        #     padding='same',
        #     activation='relu'
        # ),
        # keras.layers.MaxPool1D(
        #     pool_size=4,
        #     strides=4
        # ),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.BatchNormalization()
    ])
    final_layer = keras.layers.Dense(
            units=n_classes,
            activation='softmax'
        )
    outputs_ts = layers_ts(inputs_ts)
    if ts_only:
        outputs = outputs_ts
    else:
        layers_xw = keras.Sequential([
            keras.layers.Dense(
                units=8,
                activation='relu',
            kernel_regularizer=keras.regularizers.L2(l2_coef)
            ),
            keras.layers.BatchNormalization(),
        ])
        outputs_xw = layers_xw(inputs_xw)
        outputs = tf.concat([outputs_ts, outputs_xw], axis=-1)
    outputs = final_layer(outputs)

    model = keras.Model(
        inputs=[inputs_ts, inputs_xw],
        outputs=outputs
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    seg_len = 600
    ol_rate = 0.5

    segs, seg_labels, seg_indivs, mask_non_outlier, tabular_data, features_ts = load_data(seg_len=seg_len, ol_rate=ol_rate)
    

    embedder = Embedder(lag=24, reduce=0, dim_raw=2, channel_last=True)
    x = embedder.transform(segs)
    w_filename = './output/w_gaitndd_20230721.npy'
    if not os.path.isdir('./output'):
        os.mkdir('./output')
    if os.path.isfile(w_filename):
        w = np.load(w_filename)
    else:
        w = calculate_weighting_vectors(10*x)
        np.save(w_filename, w)

    file_log = open('./output/log_20230719.csv', 'w')
    file_log.write('random_state,ts_only_true,ts_only_false\n')

    for random_state in range(42, 42+100):
        sine_filter = SineFilter(dim=x.shape[-1], n_filters=128, scale=1e1, random_state=random_state)
        sine_0d = sine_filter.apply(x, w, batch_size=256)

        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_state)
        tf.config.experimental.enable_op_determinism()

        # subcls = ['als', 'control']
        # subcls = ['control', 'hunt']
        subcls = ['control', 'park']
        mask_subcls = np.isin(seg_labels, subcls)

        X = segs[:, ::4]
        y_true = seg_labels.copy()
        # y_true = seg_indivs.copy()
        y_pred = y_true.copy()
        y_pred_another = y_pred.copy()

        for test_indiv in np.unique(seg_indivs[np.isin(seg_labels, subcls)]):
            print(test_indiv, end=' ')
            mask_train = (seg_indivs != test_indiv) & mask_subcls & mask_non_outlier
            mask_test = (seg_indivs == test_indiv) & mask_subcls & mask_non_outlier
            y = 0*(y_true == subcls[0])
            for i in range(1, len(subcls)):
                y += i*(y_true == subcls[i])

            for ts_only in (True, False):
                model = get_model(len(subcls), ts_shape=X.shape[1:], xw_size=sine_0d.shape[-1], ts_only=ts_only)
                model.fit(
                    (X[mask_train], sine_0d[mask_train]), y[mask_train],
                    epochs=50,
                    batch_size=64,
                    verbose=0
                )
                predicted = np.vectorize(subcls.__getitem__)(np.argmax(model.predict((X[mask_test], sine_0d[mask_test]), verbose=0), axis=-1))
                if ts_only:
                    y_pred[mask_test] = predicted
                else:
                    y_pred_another[mask_test] = predicted
            acc = 1e2*accuracy_score(y_true[mask_test], y_pred[mask_test])
            acc_another = 1e2*accuracy_score(y_true[mask_test], y_pred_another[mask_test])
            print(acc, acc_another)
        print( 1e2*accuracy_score(y_true, y_pred) )
        print( 1e2*accuracy_score(y_true, y_pred_another) )
        file_log.write(f'{random_state},{1e2*accuracy_score(y_true, y_pred)},{1e2*accuracy_score(y_true, y_pred_another)}\n')
        file_log.flush(); os.fsync(file_log.fileno())
        
    file_log.close()

if __name__ == '__main__':
    main()