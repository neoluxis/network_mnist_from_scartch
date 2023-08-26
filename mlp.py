import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os, json

train_data = pd.read_csv("./datasets/train.csv")

train_data = np.array(train_data)
m, n = train_data.shape
np.random.shuffle(train_data)  # 洗牌，把训练数据打乱

data_verify = train_data[0:5000].T  # 验证集 5000
labels_verify = data_verify[0]
images_verify = data_verify[1:n]
images_verify = images_verify / 255.0

data_train = train_data[5000:m].T
labels_train = data_train[0]
images_train = data_train[1:n]
images_train = images_train / 255.0
_, m_train = images_train.shape


def init_params(h_layers=1, neuron_num=[10]):
    W_i = np.random.rand(neuron_num[0], 784) - 0.5
    B_i = np.random.rand(neuron_num[0], 1) - 0.5
    Ws_h = []
    Bs_h = []
    for i in range(h_layers):
        Ws_h.append(np.random.rand(neuron_num[i + 1], neuron_num[i]) - 0.5)
        Bs_h.append(np.random.rand(neuron_num[i + 1], 1) - 0.5)
    W_o = np.random.rand(10, neuron_num[-1]) - 0.5
    B_o = np.random.rand(10, 1) - 0.5
    return W_i, B_i, W_o, B_o, Ws_h, Bs_h


def ReLU(Z):
    return np.maximum(Z, 0)


def ReLU_deriv(Z):
    return np.where(Z > 0, 1, 0)


def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def Sigmoid_deriv(Z):
    return Sigmoid(Z) * (1 - Sigmoid(Z))


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


class Activunc:
    def __init__(self, activ, deriv) -> None:
        self.activ = activ
        self.deriv = deriv


activ_sigmoid = Activunc(Sigmoid, Sigmoid_deriv)
activ_relu = Activunc(ReLU, ReLU_deriv)


def fore_prop(W_i, B_i, Ws_h, Bs_h, W_o, B_o, Image, activ=activ_relu):
    Z_i = W_i.dot(Image) + B_i
    A_i = activ.activ(Z_i)
    Zs_h = []
    As_h = []
    Zs_h.append(Ws_h[0].dot(A_i) + Bs_h[0])
    As_h.append(activ.activ(Zs_h[0]))
    for i in range(1, len(Ws_h)):
        Zs_h.append(Ws_h[i].dot(As_h[i - 1]) + Bs_h[i])
        As_h.append(activ.activ(Zs_h[i]))
    Z_o = W_o.dot(As_h[-1]) + B_o
    A_o = softmax(Z_o)
    return Z_i, A_i, Zs_h, As_h, Z_o, A_o


def one_hot(labels):
    one_hot_labels = np.zeros((10, len(labels)))
    for i in range(len(labels)):
        one_hot_labels[labels[i], i] = 1
    return one_hot_labels


def back_prop(Z_i, A_i, Zs_h, As_h, Z_o, A_o, W_i, Ws_h, W_o, Image, labels, activ=activ_relu):
    one_hot_labels = one_hot(labels)
    dZ_o = A_o - one_hot_labels
    dW_o = 1 / m * dZ_o.dot(As_h[-1].T)
    dB_o = 1 / m * np.sum(dZ_o)
    dZs_h = [] * len(Zs_h)
    dWs_h = [] * len(Ws_h)
    dBs_h = [] * len(Zs_h) #``
    dZs_h[-1] = W_o.T.dot(dZ_o) * activ.deriv(Zs_h[-1])
    dWs_h[-1] = 1 / m * dZs_h[-1].dot(As_h[-2].T)
    dBs_h[-1] = 1 / m * np.sum(dZs_h[-1])
    for i in range(len(Zs_h) - 2, -1, -1):
        dZs_h[i] = Ws_h[i + 1].T.dot(dZs_h[i + 1]) * activ.deriv(Zs_h[i])
        dWs_h[i] = 1 / m * dZs_h[i].dot(As_h[i - 1].T)
        dBs_h[i] = 1 / m * np.sum(dZs_h[i])
    dZ_i = Ws_h[0].T.dot(dZs_h[0]) * activ.deriv(Z_i)
    dW_i = 1 / m * dZ_i.dot(Image.T)
    dB_i = 1 / m * np.sum(dZ_i)
    return dW_i, dB_i, dWs_h, dBs_h, dW_o, dB_o

def update_params(W_i, B_i, Ws_h, Bs_h, W_o, B_o, dW_i, dB_i, dWs_h, dBs_h, dW_o, dB_o, alpha):
    W_i = W_i - alpha * dW_i
    B_i = B_i - alpha * dB_i
    for i in range(len(Ws_h)):
        Ws_h[i] = Ws_h[i] - alpha * dWs_h[i]
        Bs_h[i] = Bs_h[i] - alpha * dBs_h[i]
    W_o = W_o - alpha * dW_o
    B_o = B_o - alpha * dB_o
    return W_i, B_i, Ws_h, Bs_h, W_o, B_o

def get_pred(A_o):
    return np.argmax(A_o, axis=0)

def get_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size

def save_params(W_i, B_i, Ws_h, Bs_h, W_o, B_o):
    pass

def load_params(path):
    pass

def train(Images, labels, alpha, iterations, path=None, activ=activ_relu):
    W_i, B_i, Ws_h, Bs_h, W_o, B_o = init_params()
    
    for i in range(iterations):
        Z_i, A_i, Zs_h, As_h, Z_o, A_o = fore_prop(W_i, B_i, Ws_h, Bs_h, W_o, B_o, Images, activ)
        dW_i, dB_i, dWs_h, dBs_h, dW_o, dB_o = back_prop(Z_i, A_i, Zs_h, As_h, Z_o, A_o, W_i, Ws_h, W_o, Images, labels, activ)
        W_i, B_i, Ws_h, Bs_h, W_o, B_o = update_params(W_i, B_i, Ws_h, Bs_h, W_o, B_o, dW_i, dB_i, dWs_h, dBs_h, dW_o, dB_o, alpha)

        if i % 10 == 0:
            print("Iteration: ", i)
            _, A_i, _, _, _, A_o = fore_prop(W_i, B_i, Ws_h, Bs_h, W_o, B_o, Images, activ)
            predictions = get_pred(A_o)
            print(get_accuracy(predictions, labels))