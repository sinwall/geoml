import numpy as np
import pandas as pd
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
# import tsfresh


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


def load_data(gap_max=1.0, seg_dur=5.0, ol_rate=0.5, resamp_freq=100, burn=0., ma=None):
    df = pd.read_csv(
        f'../input/WISDM_ar_v1.1/WISDM_ar_v1.1_raw_modified.txt', 
        names=['user', 'activity', 'timestamp', 'ax', 'ay', 'az'],
        header=None)
    
    segs, seg_acts, seg_usrs = Preprocessor(
        gap_max=gap_max, seg_dur=seg_dur, ol_rate=ol_rate, resamp_freq=resamp_freq, burn=burn, ma=ma
    ).transform(df)
    return segs, seg_acts, seg_usrs