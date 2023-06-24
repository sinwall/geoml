# # uncomment if in colab environment
# from google.colab import drive
# import os
# drive.mount('/content/drive')
# os.chdir('/content/drive/MyDrive/laboratory/geoml')
# !git pull


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans Mono'
from scipy.interpolate import interp1d
import itertools

import os, sys
sys.path.append(os.path.abspath('./'))

from tools import Embedder, SineFilter, calculate_weighting_vectors

from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import tensorflow as tf
from tensorflow import keras
import random


class Preprocessor():
    def __init__(self, gap_max=1.0, seg_dur=5.0, ol_rate=0.5, resamp_freq=100, burn=0., ma=None):
        self.gap_max = gap_max
        self.seg_dur = seg_dur
        self.ol_rate = ol_rate
        self.resamp_freq = resamp_freq
        self.burn = burn
        self.ma = ma

    def clean(self, df):
        # df = pd.read_csv(
        #     f'../input/WISDM_ar_v1.1/WISDM_ar_v1.1_raw_modified.txt', 
        #     names=['user', 'activity', 'timestamp', 'ax', 'ay', 'az'],
        #     header=None)
        df = df.copy()
        df['timestamp'] *= 1e-9
        df = df[df['timestamp'] != 0]
        df = df[~df['timestamp'].duplicated()]
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def generate_components(self, df):
        gap_max = self.gap_max
        seg_dur = self.seg_dur
        ol_rate = self.ol_rate
        resamp_freq = self.resamp_freq

        users = sorted(df['user'].unique())
        activities = sorted(df['activity'].unique())

        for user, activity in itertools.product(users, activities):
            mask_ua = (df['user'] == user) & (df['activity'] == activity)
            if not mask_ua.any():
                continue
            cpnt_nums = np.cumsum( (df.loc[mask_ua, 'timestamp'].diff() > gap_max) | (df.loc[mask_ua, 'timestamp'].diff() < 0) )
            for num in range(cpnt_nums.min(), cpnt_nums.max()+1):
                cpnt = df[mask_ua][cpnt_nums == num]
                cpnt = cpnt[cpnt['timestamp'] >= (cpnt['timestamp'].min() + self.burn)]
                cpnt['timestamp'] -= cpnt['timestamp'].min()

                if len(cpnt) < 4:
                    continue
                # trusting timestamps
                f = interp1d(cpnt['timestamp'], cpnt[['ax', 'ay', 'az']], axis=0, kind='linear')
                t = np.arange(0, cpnt['timestamp'].max(), 1/resamp_freq)
                # not trusting timestamps
                # f = interp1d(np.arange(cpnt.shape[0])/20, cpnt[['ax', 'ay', 'az']], axis=0, kind='linear')
                # t = np.arange(5*cpnt.shape[0]-5)/100
                signal = f(t)

                if self.ma is not None:
                    signal = np.stack([np.convolve(el, np.ones((self.ma, ))/self.ma, mode='same') for el in signal.T], axis=-1)
                yield signal, activity, user

    def transform(self, df):
        gap_max = self.gap_max
        seg_dur = self.seg_dur
        ol_rate = self.ol_rate
        resamp_freq = self.resamp_freq

        df = self.clean(df)

        segments = []
        seg_usrs = []
        seg_acts = []

        for cpnt, act, usr in self.generate_components(df):
            for begin in np.arange(0, cpnt.shape[0]-int(seg_dur*resamp_freq), int(seg_dur*(1-ol_rate)*resamp_freq)):
                segments.append(cpnt[begin:begin+int(seg_dur*resamp_freq)])
                seg_acts.append(act)
                seg_usrs.append(usr)

        return np.array(segments), np.array(seg_acts), np.array(seg_usrs)


def load_dataset(gap_max=1.0, seg_dur=5.0, ol_rate=0.5, resamp_freq=100, burn=0., ma=None):
    df = pd.read_csv(
        f'../input/WISDM_ar_v1.1/WISDM_ar_v1.1_raw_modified.txt', 
        names=['user', 'activity', 'timestamp', 'ax', 'ay', 'az'],
        header=None)
    return Preprocessor(
        gap_max=gap_max, seg_dur=seg_dur, ol_rate=ol_rate, resamp_freq=resamp_freq, burn=burn, ma=ma
    ).transform(df)


def get_model(ts_only=True):
    l2_coef = 1e-2
    ts_part = keras.Sequential([
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
            kernel_size=5,
            strides=2,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(l2_coef)
        ),
        keras.layers.MaxPooling1D(
            pool_size=2,
            strides=2,
        ),
        keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(l2_coef)
        ),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.BatchNormalization(),
    ])

    final_layer = keras.layers.Dense(
            units=6,
            activation='softmax',
            kernel_regularizer=keras.regularizers.L2(l2_coef)
        )

    if ts_only:
        model = keras.Sequential([ts_part, final_layer])
    else:
        input_ts = keras.Input(shape=(120, 3))
        input_f = keras.Input(shape=(32, ))

        f_part = keras.Sequential([
            keras.layers.Dense(
                units=16,
                activation='relu',
                kernel_regularizer=keras.regularizers.L2(l2_coef)
            ),
            keras.layers.BatchNormalization(),
        ])

        output_ts = ts_part(input_ts)
        output_f = f_part(input_f)
        outputs = final_layer(tf.concat([output_ts, output_f], axis=-1))

        model = keras.Model(inputs=[input_ts, input_f], outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    df = pd.read_csv(
        f'../input/WISDM_ar_v1.1/WISDM_ar_v1.1_raw_modified.txt', 
        names=['user', 'activity', 'timestamp', 'ax', 'ay', 'az'],
        header=None)
    users = sorted(df['user'].unique())
    activities = sorted(df['activity'].unique())


    segs_dnn, seg_acts_dnn, seg_usrs_dnn = load_dataset(seg_dur=6, ol_rate=0.5, resamp_freq=20, ma=None)
    seg_acts_encoded_dnn = OrdinalEncoder(categories=[activities]).fit_transform(seg_acts_dnn[..., np.newaxis])[:, 0]
    segs, _, _ = load_dataset(seg_dur=6, ol_rate=0.5, resamp_freq=100, ma=10)

    x = Embedder(dim_raw=2, lag=8, reduce=0, channel_last=True).transform(segs)
    if not os.isfile('./output/w_20230624.npy'):
        w = calculate_weighting_vectors(x)[..., np.newaxis]
        np.save('./output/w_20230624.npy', w, allow_pickle=False)
    else:
        w = np.load('./output/w_20230624.npy', allow_pickle=False)

    sine_filter = SineFilter(dim=x.shape[-1], n_filters=32, scale=1e-1, random_state=42)
    sine_0d = sine_filter.apply(x, w)  #new filter......
    sine_0d = PCA(whiten=True, random_state=42).fit_transform(sine_0d)


    y_true = seg_acts_encoded_dnn
    y_pred = seg_acts_encoded_dnn.copy()

    rng = np.random.default_rng(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    n_fold = 100
    file_w = open('./output/accs_20230624.csv', 'w')
    for fold_no in range(n_fold):
        usrs_tst = rng.choice(len(users), 12, replace=False)
        usrs_val = np.sort(usrs_tst[:6])
        usrs_tst = np.sort(usrs_tst[6:])
        mask_val_dnn = np.isin(seg_usrs_dnn, usrs_val)
        mask_test_dnn = np.isin(seg_usrs_dnn, usrs_tst)
        mask_train_dnn = ~(mask_val_dnn | mask_test_dnn)

        scrs = []
        for ts_only in (True, False):
            # if ts_only is False: continue
            model = get_model(ts_only=ts_only)
            callbacks = keras.callbacks.EarlyStopping(
                patience=30, restore_best_weights=True
            )
            model.fit(
                (segs_dnn[mask_train_dnn] if ts_only else [segs_dnn[mask_train_dnn], sine_0d[mask_train_dnn]]), 
                seg_acts_encoded_dnn[mask_train_dnn],
                batch_size=192,
                epochs=200,
                validation_data=(
                    segs_dnn[mask_val_dnn] if ts_only else [segs_dnn[mask_val_dnn], sine_0d[mask_val_dnn]], 
                    seg_acts_encoded_dnn[mask_val_dnn]
                ),
                callbacks=callbacks,
                verbose=0
            )
            y_true_fold = seg_acts_encoded_dnn[mask_test_dnn]
            y_pred_fold = np.argmax(model.predict(
                segs_dnn[mask_test_dnn] if ts_only else [segs_dnn[mask_test_dnn], sine_0d[mask_test_dnn]],
                verbose=0
            ), axis=-1)

            scrs.append(1e2*accuracy_score(y_true_fold, y_pred_fold))
        print(f'without: {scrs[0]:.2f}, with: {scrs[1]:.2f}, delta: {scrs[1]-scrs[0]:.2f}')
        file_w.write(f'{scrs[0]},{scrs[1]}\n')
        # y_pred[mask_test_dnn] = y_pred_fold
    file_w.close()
    # print(confusion_matrix(y_true, y_pred))
    # print(accuracy_score(y_true, y_pred))
    # print(f1_score(y_true, y_pred, average=None))

if __name__ == '__main__':
    main()