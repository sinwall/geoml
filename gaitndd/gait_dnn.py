import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans Mono'

import os
import wfdb
from scipy.interpolate import interp1d
from sklearn.impute import SimpleImputer


import tensorflow as tf
from tensorflow import keras
import random
from sklearn.metrics import accuracy_score

def get_model(n_classes):
    model = keras.Sequential([
        keras.layers.LSTM(
            units=32,
            return_sequences=True
        ),
        keras.layers.LSTM(
            units=16,
            return_sequences=True
        ),
        keras.layers.Conv1D(
            filters=16,
            kernel_size=16,
            activation='relu'
        ),
        keras.layers.MaxPool1D(
            pool_size=4,
            strides=4
        ),
        keras.layers.Conv1D(
            filters=16,
            kernel_size=16,
            activation='relu'
        ),
        keras.layers.MaxPool1D(
            pool_size=4,
            strides=4
        ),
        keras.layers.Conv1D(
            filters=16,
            kernel_size=16,
            activation='relu'
        ),
        keras.layers.MaxPool1D(
            pool_size=4,
            strides=4
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=n_classes,
            activation='softmax'
        )
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    n_indivs = 64
    default_freq = 300
    # signal from wfdb, human crafted features
    signals_pre = np.empty((n_indivs, 90000, 2))
    labels = []
    ts_files = []
    input_path = f'E:/database/gait-in-neurodegenerative-disease-database-1.0.0'
    for file_name in sorted(os.listdir(f'{input_path}')):
        if not file_name.endswith('hea'):
            continue
        label = ''.join(filter(lambda x: x.isalpha(), file_name[:-4]))
        file_name = f'{input_path}/{file_name[:-4]}'
        sig, info = wfdb.rdsamp(file_name)

        t = np.arange(len(sig))
        for j in range(2):
            mask_na = np.isnan(sig[..., j])
            f = interp1d(t[~mask_na], sig[~mask_na, j], fill_value='extrapolate')
            sig[mask_na, j] = f(t[mask_na])
        signals_pre[len(labels)] = sig; labels.append( label )
        assert info['fs'] == default_freq
        assert info['sig_len'] == 90000

        ts_file = pd.read_csv(
            file_name + '.ts',
            delimiter='\t',
            names=[
                'Elapsed Time (sec)',
                'Left Stride Interval (sec)', 'Right Stride Interval (sec)', 'Left Swing Interval (sec)', 'Right Swing Interval (sec)',
                'Left Swing Interval (% of stride)', 'Right Swing Interval (% of stride)', 
                'Left Stance Interval (sec)', 'Right Stance Interval (sec)', 
                'Left Stance Interval (% of stride)', 'Right Stance Interval (% of stride)',
                'Double Support Interval (sec)', 'Double Support Interval (% of stride)'
            ]
        )
        ts_files.append(ts_file)

    labels = np.array(labels)

    # basic clinical data
    df_tabular = pd.read_csv(
        f'{input_path}/subject-description.txt', 
        delimiter='\t',
        na_values='MISSING'
    )
    df_tabular['gender'] = (df_tabular['gender'] == 'm').astype(int)
    tabular_data_preimpute = df_tabular.iloc[:, 2:].values

    tabular_data = SimpleImputer().fit_transform(tabular_data_preimpute)

    from scipy.signal import butter, sosfilt

    signals = signals_pre
    # signal filtering and scaling
    # signals = sosfilt(butter(2, 15, btype='low', output='sos', fs=default_freq), signals, axis=-2)
    # signal_means, signal_stds = np.mean(signals, axis=-2), np.std(signals, axis=-2)
    # signals = (signals - signal_means[..., np.newaxis, :]) / (signal_stds[..., np.newaxis, :])

    seg_len = 600
    ol_rate = 0.8
    n_segs_per_indiv = len(np.arange(0, signals.shape[1]-seg_len+1, int(seg_len*(1-ol_rate))))

    # signal partitioning
    segs = []
    seg_labels = []
    seg_indivs = []
    mask_nonstart = []
    for i in range(n_indivs):
        signal = signals[i]
        for j in range(0, signals.shape[1]-seg_len+1, int(seg_len*(1-ol_rate))):
            segs.append(signal[j:j+seg_len])
            seg_labels.append( labels[i] )
            seg_indivs.append(i)
            mask_nonstart.append(j >= default_freq*20) # 20s in the beginning discarded
    segs = np.array(segs); seg_labels = np.array(seg_labels); seg_indivs = np.array(seg_indivs)
    mask_nonstart = np.array(mask_nonstart)

    # ts file partitioning # currently invalid
    features_ts = np.zeros((n_indivs, n_segs_per_indiv, ts_files[0].shape[1]-1))
    mask_turnback = np.full((n_indivs, n_segs_per_indiv), False)
    for i in range(n_indivs):
        ts_file = ts_files[i]
        stride_l = ts_file['Left Stride Interval (sec)']; slm = stride_l.median(); sls = stride_l.std()
        stride_r = ts_file['Right Stride Interval (sec)']; srm = stride_r.median(); srs = stride_r.std()
        for _, row in ts_file.iterrows():
            t = row['Elapsed Time (sec)']; dt = row['Left Stride Interval (sec)']
            begin = ((t-dt)*default_freq -seg_len )/ (seg_len*(1-ol_rate)); assert begin >= 0
            end = t*default_freq / (seg_len*(1-ol_rate))
            js = np.arange(int(begin)+1, min(int(end)+1, n_segs_per_indiv))
            # ratio = np.minimum(js+1, end) - np.maximum(js, begin)
            ratio = 1 - np.maximum((begin-js)*(1-ol_rate)+1, 0) - np.maximum((js-end)*(1-ol_rate)+1, 0)
            features_ts[i, js] += ratio[..., np.newaxis]*row.iloc[1:].values
            # for j in range(int(begin), int(end)+1):
            #     ratio = min((j+1), end) - max(j, begin)
            #     features_ts[i*n_segs_per_indiv + j] += ratio*row.iloc[1:]

        # expand left bdry value
        # begin = (ts_file.iloc[0, 0]-ts_file.iloc[0, 1])*default_freq / seg_len
        end = (ts_file.iloc[0, 0]-ts_file.iloc[0, 1])*default_freq / (seg_len*(1-ol_rate))
        # ratio = np.minimum(np.arange(begin, 0, -1), 1)
        ratio = np.minimum(( end - np.arange(int(end)+1))*(1-ol_rate), 1)
        features_ts[i, :len(ratio)] += ratio[..., np.newaxis]*ts_file.iloc[0, 1:].values

        # expand right bdry value
        begin = (ts_file.iloc[-1, 0]* default_freq -seg_len)/ (seg_len*(1-ol_rate))
        # end = ts_file.iloc[-1, 0]* default_freq / seg_len
        # ratio = np.minimum(np.arange(int(end), n_segs_per_indiv)-end+1, 1)
        ratio = np.minimum((np.arange(int(begin)+1, n_segs_per_indiv) - begin)*(1-ol_rate), 1)
        if len(ratio) != 0:
            features_ts[i, -len(ratio):] += ratio[..., np.newaxis]*ts_file.iloc[-1, 1:].values

        mask_turnback[i, (np.abs(features_ts[i, :, 0]-slm)>3*sls)|(np.abs(features_ts[i, :, 1]-srm)>3*srs)] = True

    features_ts = features_ts.reshape(-1, features_ts.shape[-1])
    mask_turnback = mask_turnback.reshape(-1)

    mask_non_outlier = mask_nonstart & (~mask_turnback)

    # from scipy.interpolate import interp1d

    # features_ts = np.empty((segs.shape[0], 12))
    # for i in range(signals.shape[0]):
    #     ts_file = ts_files[i]
    #     x, y = ts_file['Elapsed Time (sec)']*300, ts_file.iloc[:, 1:].values
    #     f = interp1d(
    #         x, y,
    #         axis=0, bounds_error=False, fill_value=(y[0], y[-1]),
    #     )
    #     features_ts[seg_indivs==i] = f(np.arange(0, signals.shape[1], seg_len) + 0.5*seg_len)

    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # subcls = ['als', 'control']
    subcls = ['control', 'hunt']
    mask_subcls = np.isin(seg_labels, subcls)

    X = segs
    y_true = seg_labels.copy()
    # y_true = seg_indivs.copy()
    y_pred = y_true.copy()

    for test_indiv in np.unique(seg_indivs[np.isin(seg_labels, subcls)]):
        print(test_indiv, end=' ')
        mask_train = (seg_indivs != test_indiv) & mask_subcls & mask_non_outlier
        mask_test = (seg_indivs == test_indiv) & mask_subcls & mask_non_outlier
        y = 0*(y_true == subcls[0])
        for i in range(1, len(subcls)):
            y += i*(y_true == subcls[i])


        model = get_model(len(subcls))
        model.fit(
            X[mask_train], y[mask_train],
            epochs=10,
            batch_size=64,
            verbose=0
        )
        y_pred[mask_test] = np.vectorize(subcls.__getitem__)(np.argmax(model.predict(X[mask_test], verbose=0), axis=-1))
        acc = 1e2*accuracy_score(y_true[mask_test], y_pred[mask_test])
        print(acc)
    1e2*accuracy_score(y_true, y_pred)