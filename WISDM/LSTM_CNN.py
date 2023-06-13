
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans Mono'
from scipy.interpolate import interp1d
import itertools

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


def main():
    df = pd.read_csv(
        f'../input/WISDM_ar_v1.1/WISDM_ar_v1.1_raw_modified.txt', 
        names=['user', 'activity', 'timestamp', 'ax', 'ay', 'az'],
        header=None)
    users = sorted(df['user'].unique())
    activities = sorted(df['activity'].unique())

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    l2_coef = 1e-1
    model = keras.Sequential([
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
        keras.layers.Dense(
            units=6,
            activation='softmax',
            kernel_regularizer=keras.regularizers.L2(l2_coef)
        )
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    segs_dnn, seg_acts_dnn, seg_usrs_dnn = load_dataset(seg_dur=6, ol_rate=0.5, resamp_freq=20, ma=None)
    seg_acts_encoded_dnn = OrdinalEncoder(categories=[activities]).fit_transform(seg_acts_dnn[..., np.newaxis])[:, 0]

    y_true = seg_acts_encoded_dnn
    y_pred = seg_acts_encoded_dnn.copy()

    rng = np.random.default_rng(42)
    n_fold = 5
    fold_assign = rng.permuted(np.linspace(0, n_fold, endpoint=False)//1)[seg_usrs_dnn]
    for fold_no in range(n_fold):
        mask_train_dnn = fold_assign == fold_no
        mask_test_dnn = ~mask_train_dnn

        model.fit(
            segs_dnn[mask_train_dnn], seg_acts_encoded_dnn[mask_train_dnn],
            epochs=200,
            validation_data=(segs_dnn[mask_test_dnn], seg_acts_encoded_dnn[mask_test_dnn])
        )
        y_true_fold = seg_acts_encoded_dnn[mask_test_dnn]
        y_pred_fold = np.argmax(model.predict(segs_dnn[mask_test_dnn]), axis=-1)

        print(1e2*accuracy_score(y_true_fold, y_pred_fold))
        y_pred[mask_test_dnn] = y_pred_fold
    print(confusion_matrix(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    print(f1_score(y_true, y_pred, average=None))

if __name__ == '__main__':
    main()