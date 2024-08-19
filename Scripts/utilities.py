import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import scipy.stats as stats
import seaborn as sns
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GRU
from tcn import TCN, tcn_full_summary
import time
import pickle


def critical_points(df):
    """
    :param df: Pandas dataframe for one measurement with only required columns (time, temperature, DSCalpha, Impedance)
    :return: Indices of critical points, CP2, CP3, CP4 in df
    """

    # CP2 is where impedance value is minimum
    cp2 = df[df['impedance1.78kHz/Ohm'] == df['impedance1.78kHz/Ohm'].min()].index.tolist()[0]

    # CP4 is where DSCalpha value becomes more than 0.95
    cp4 = df[df['DSCalpha'] >= 0.95].index.tolist()[0]

    # To find point of inflection we find second derivative
    # As the inflection point is between CP2 and CP4, we calculate derivatives only between these points
    df[['first_der', 'second_der', 'curve_dir_change']] = np.nan
    dy = np.diff(df['impedance1.78kHz/Ohm'].iloc[cp2:cp4 + 1])
    dx = np.diff(df['time_/min'].iloc[cp2:cp4 + 1])
    df['first_der'].iloc[cp2 + 1:cp4 + 1] = dy / dx
    d2y = np.diff(df['first_der'].iloc[cp2 + 1:cp4 + 1])
    d2x = 0.5 * (df['time_/min'].iloc[cp2:cp4] + df['time_/min'].iloc[cp2 + 1:cp4 + 1]).dropna()
    df['second_der'].iloc[cp2 + 2:cp4 + 1] = d2y / d2x

    # Point of inflection is where second derivative changes sign,
    # hence multiplying each second derivative value with previous to find where multiplication is negative
    df['curve_dir_change'].iloc[cp2 + 3:cp4 + 1] = np.multiply(np.array(df['second_der'].iloc[cp2 + 2:cp4]),
                                                               np.array(df['second_der'].iloc[cp2 + 3:cp4 + 1]))
    cp3 = df[df['curve_dir_change'] <= 0].index.tolist()[0]

    return cp2, cp3, cp4


def find_delta_t(df, cp2, cp4):
    """
    :param df: Pandas dataframe for one measurement with only required columns (time, temperature, DSCalpha, Impedance)
    :param cp2: Index of critical point CP2
    :param cp4: Index of critical point CP3
    :return: Time difference betwwen CP2 and CP4
    """
    # Difference between time at CP4 and CP2
    delta_t = df['time_/min'].iloc[cp4] - df['time_/min'].iloc[cp2]
    return delta_t


def find_train_data(df, cp2, cp3):
    """
    :param df: Pandas dataframe for one measurement with only required columns (time, temperature, DSCalpha, Impedance)
    :param cp2: Index of critical point CP2
    :param cp3: Index of critical point CP3
    :return: Pandas dataframe consisting of all rows between CP2 and CP3
    """
    xcp23 = df.iloc[cp2:cp3+1, :4]
    return xcp23


def impedance_curve(df, filename, cp2, cp3, cp4):
    """

    :param df: Pandas dataframe for one measurement with only required columns (time, temperature, DSCalpha, Impedance)
    :param filename: Name of file containing data for that measurement
    :param cp2: Index of critical point CP2
    :param cp3: Index of critical point CP3
    :param cp4: Index of critical point CP4
    :return: Impedance Curve
    """
    plt.scatter(df['time_/min'], df['impedance1.78kHz/Ohm'], s = 5)

    # Highlighting specific data points
    plt.scatter(df['time_/min'].iloc[cp2], df['impedance1.78kHz/Ohm'].iloc[cp2], c='brown', marker='o')
    plt.annotate('CP2', (df['time_/min'].iloc[cp2], df['impedance1.78kHz/Ohm'].iloc[cp2]), xytext=(-60, 10),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'), c='brown')
    plt.scatter(df['time_/min'].iloc[cp3], df['impedance1.78kHz/Ohm'].iloc[cp3], c='green', marker='o')
    plt.annotate('CP3', (df['time_/min'].iloc[cp3], df['impedance1.78kHz/Ohm'].iloc[cp3]), xytext=(50, 10),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'), c='green')
    plt.scatter(df['time_/min'].iloc[cp4], df['impedance1.78kHz/Ohm'].iloc[cp4], c='orange', marker='o')
    plt.annotate('CP4', (df['time_/min'].iloc[cp4], df['impedance1.78kHz/Ohm'].iloc[cp4]), xytext=(50, -30),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'), c='orange')

    plt.xlabel("Time (min)", fontsize=13, labelpad=10)
    plt.xticks(fontsize=10)
    plt.ylabel("Impedance", fontsize=13, labelpad=10)
    plt.yticks(fontsize=10)
    plt.title(f"Impedance Curve - {filename}", fontsize=16, pad=15)
    plt.show()


def data_transformation(xcp23, mean_time_cp2_cp3):
    """
    Standardizes, applies PCA and scales the data to prepare for training
    :param xcp23: Pandas dataframe for one measurement with only required columns (time, temperature, DSCalpha, Impedance)
    :param mean_time_cp2_cp3: Mean time between CP2 and CP3 for that specific measurement
    :return: pandas dataframe with the data ready to be used for training
    """
    # Standardising data
    xcp23 = xcp23.apply(stats.zscore)

    # Applying PCA transform to reduce dimensionality
    pca = PCA(n_components=2)
    pca.fit(xcp23)
    data_pca = pca.transform(xcp23)
    data_pca = pd.DataFrame(data_pca, columns=["pca1", "pca2"])

    # Scaling data around mean time between CP2 and CP3
    data_pca = data_pca.apply(lambda x: mean_time_cp2_cp3 * x)
    xcp23 = np.array(data_pca)

    return xcp23


def create_lstm_model(input_shape):
    """
    :param input_shape: Shape of input data to the model
    :return: LSTM based model
    """
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.0)(inputs)
    lstm_out = LSTM(64)(masked_inputs)  # Single output from LSTM
    outputs = Dense(1)(lstm_out)

    model = Model(inputs, outputs)
    return model

def create_gru_model(input_shape):
    """
    :param input_shape: Shape of input data to the model
    :return: GRU based model
    """
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.0)(inputs)
    gru_out = GRU(64)(masked_inputs)  # Single output from GRU
    outputs = Dense(1)(gru_out)  # Single output

    model = Model(inputs, outputs)
    return model


def create_cnn_model(input_shape):
    """
    :param input_shape: Shape of input data to the model
    :return: CNN based model
    """
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.0)(inputs)
    conv_out = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(masked_inputs)
    pooled_out = MaxPooling1D()(conv_out)
    conv_out = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(pooled_out)
    pooled_out = GlobalMaxPooling1D()(conv_out)
    outputs = Dense(1)(pooled_out)

    model = Model(inputs, outputs)
    return model


def create_tcn_model(input_shape):
    """
    :param input_shape: Shape of input data to the model
    :return: TCN based model
    """
    inputs = Input(shape=input_shape)
    masked_inputs = Masking(mask_value=0.0)(inputs)
    tcn_out = TCN(nb_filters=64, dilations=[1, 2, 4, 8, 16])(masked_inputs)
    outputs = Dense(1)(tcn_out)

    model = Model(inputs, outputs)
    return model
