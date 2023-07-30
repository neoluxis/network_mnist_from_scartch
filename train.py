import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import json
import os

train_data = pd.read_csv("./datasets/train.csv")

train_data = np.array(train_data)
m, n = train_data.shape
# print(m, n)
np.random.shuffle(train_data)  # 洗牌，把训练数据打乱 ? 为什么要打乱, 不打乱是否可行？

data_dev = train_data[0:1000].T  # 1000个样本作为验证集
# print(data_dev[0].shape)
te_labels = data_dev[0]  # 第一列是标签
te_images = data_dev[1:n]  # Test images
# print(te_images.shape, te_labels.shape)
te_images = te_images / 255.0  # 归一化 0-255 -> 0-1

tr_images = train_data[1000:m].T  # 剩下的作为训练集
tr_labels = tr_images[0]  # 第一列是标签
tr_images = tr_images[1:n]  # Train images
tr_images = tr_images / 255.0  # 归一化 0-255 -> 0-1

_, m_train = tr_images.shape  # 训练集样本数
# print(_, m_train)

def initWB(layers=1, neurons=10, final=10):
    '''
    初始化参数
    先随机生成权重和偏置
    
    Args:
    layers: 隐藏层数(Input->Hidden 1->Output, 这是1层)
    neurons: 每隐藏层的神经元数, 为了简单起见，每层的神经元数都一样
    final: 输出层的神经元数
    
    Returns:
    vWs_h: 隐藏层权重, 一个列表，每个元素是一个矩阵
    vBs_h: 隐藏层偏置, 一个列表，每个元素是一个矩阵
    vW_o: 输出层权重, 一个矩阵
    vB_o: 输出层偏置, 一个矩阵
    '''
    vWs_h, vBs_h = [], []
    for i in range(layers):
        # In hiden layers
        vW_h = np.random.rand(neurons, 784) - 0.5 # 784 -> 10 第一层(1st Hidden Layer)有10个神经元，一个784行10列的矩阵
        vB_h = np.random.rand(neurons, 1) - 0.5 # 一个神经元对应一个偏置， 一个10行1列的矩阵
        vWs_h.append(vW_h)
        vBs_h.append(vB_h)
    vW_o = np.random.rand(final, neurons) - 0.5 # 10 -> 10 第二层(Output Layer)有10个神经元，一个10行10列的矩阵
    vB_o = np.random.rand(final, 1) - 0.5 # 一个神经元对应一个偏置， 一个10行1列的矩阵
    return vWs_h, vBs_h, vW_o, vB_o

# Then define 2 activation functions, ReLU, Sigmoid
def ReLU(x):
    return np.maximum(x, 0)
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the softmax function, which is used in the output layer
def softmax(x):
    return np.exp(x) / sum(np.exp(x))

# Define the derivative of 2 activation functions
def ReLU_deriv(x):
    return x > 0
def Sigmoid_deriv(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

# 定义前向传播
def forward_prop(vWs_h, vBs_h, vW_o, vB_o, X, activation):
    '''
    Args:
    vWs_h: 所有Hidden Layer的权重(weight)
    vBs_h: 所有Hidden Layer的偏置(bias)
    vW_o: Output Layer的权重(weight)
    vB_o: Output Layer的偏置(bias)
    X: 输入数据
    activation: 激活函数
    '''
    Zs_h, As_h = [], []
    for i in range(len(vWs_h)):
        Z_h = vWs_h[i].dot(X) + vBs_h[i]
        A_h = activation(Z_h)
        Zs_h.append(Z_h)
        As_h.append(A_h)
    Z_o = vW_o.dot(As_h[-1]) + vB_o
    A_o = softmax(Z_o)
    return Zs_h, As_h, Z_o, A_o

# I dont know what this function is doing
def one_hot_encode(labels):
    """
    将标签转换为独热编码

    Args:
    labels: 标签数组

    Returns:
    one_hot_labels: 独热编码数组
    """
    num_labels = labels.size  # 获取标签数量, e.g. 0-9, 10个标签
    num_classes = labels.max() + 1  # 获取类别数量 e.g. 0-9, 10个类别
    one_hot_labels = np.zeros((num_classes, num_labels))  # 初始化独热编码数组, 10行10列
    for i in range(num_labels):
        label = labels[i]  # 获取当前标签
        one_hot_labels[label, i] = 1  # 将对应位置的值设为1, 第label行, 第i列
    return one_hot_labels
# test one_hot_encode
# test_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(one_hot_encode(test_arr))

# 定义反向传播
def back_prop(Zs_h, As_h, Z_o, A_o, vWs_h, vW_o, X, labels, activation_deriv):
    '''
    反向传播
    从输出层开始，逐层计算误差，然后计算梯度，返回梯度
    
    Args:
    Zs_h: 所有Hidden Layer的Z
    As_h: 所有Hidden Layer的A
    Z_o: Output Layer的Z
    A_o: Output Layer的A
    vWs_h: 所有Hidden Layer的权重(weight); 
    vW_o: Output Layer的权重(weight)
    X: 输入数据
    labels: 标签的列表
    activation_deriv: 激活函数的导数
    
    Returns:
    dWs_h: 所有Hidden Layer权重的梯度
    dBs_h: 所有Hidden Layer偏置的梯度
    dW_o: Output Layer权重的梯度
    dB_o: Output Layer偏置的梯度
    '''
    oh_labels = one_hot_encode(labels)  # 将标签转换为独热编码
    dZ_o = A_o - oh_labels  # 输出层的误差
    dW_o = 1 / m_train * dZ_o.dot(As_h[-1].T)  # 输出层权重的梯度
    dB_o = 1 / m_train * np.sum(dZ_o)  # 输出层偏置的梯度
    
    dZs_h = [None] * len(Zs_h)  # 隐藏层的误差
    dWs_h = [None] * len(vWs_h)  # 隐藏层权重的梯度
    dBs_h = [None] * len(vWs_h)  # 隐藏层偏置的梯度
    
    dZs_h[-1] = dW_o.T.dot(dZ_o) * ReLU_deriv(Zs_h[-1])  # 最后一层隐藏层的误差
    dWs_h[-1] = 1 / m_train * dZs_h[-1].dot(X.T)  # 最后一层隐藏层权重的梯度
    dBs_h[-1] = 1 / m_train * np.sum(dZs_h[-1])  # 最后一层隐藏层偏置的梯度
    for i in range(-2, -len(vWs_h), -1):
        dZs_h[i] = vWs_h[i + 1].T.dot(dZs_h[i + 1]) * activation_deriv(Zs_h[i])
        dWs_h[i] = 1 / m_train * dZs_h[i].dot(As_h[i - 1].T)
        dBs_h[i] = 1 / m_train * np.sum(dZs_h[i])
        pass
    dZs_h[0] = vWs_h[0].T.dot(dZs_h[0]) * activation_deriv(Zs_h[0]) # 第一层隐藏层的误差
    dWs_h[0] = 1 / m_train * dZs_h[0].dot(X.T) # 第一层隐藏层权重的梯度
    dBs_h[0] = 1 / m_train * np.sum(dZs_h[0])  # 第一层隐藏层偏置的梯度
    return dWs_h, dBs_h, dW_o, dB_o

# 定义更新参数
def update_params(vWs_h, vBs_h, vW_o, vB_o, dWs_h, dBs_h, dW_o, dB_o, alpha):
    '''
    更新参数
    包括所有Hidden Layer的权重和偏置，以及Output Layer的权重和偏置
    
    Args:
    vWs_h: 所有Hidden Layer的权重(weight)
    vBs_h: 所有Hidden Layer的偏置(bias)
    vW_o: Output Layer的权重(weight)
    vB_o: Output Layer的偏置(bias)
    dWs_h: 所有Hidden Layer权重的梯度
    dBs_h: 所有Hidden Layer偏置的梯度
    dW_o: Output Layer权重的梯度
    dB_o: Output Layer偏置的梯度
    alpha: 学习率
    
    Returns:
    vWs_h: 所有Hidden Layer的权重(weight)
    vBs_h: 所有Hidden Layer的偏置(bias)
    vW_o: Output Layer的权重(weight)
    vB_o: Output Layer的偏置(bias)
    '''
    for i in range(len(vWs_h)):
        vWs_h[i] = vWs_h[i] - alpha * dWs_h[i]
        vBs_h[i] = vBs_h[i] - alpha * dBs_h[i]
        pass
    vW_o = vW_o - alpha * dW_o
    vB_o = vB_o - alpha * dB_o
    return vWs_h, vBs_h, vW_o, vB_o

def get_predictions(Activation_o):
    '''
    获取预测结果
    
    Args:
    Activation_o: Output Layer的激活值
    
    Returns:
    predictions: 预测结果
    '''
    return np.argmax(Activation_o, 0)

def get_accuracy(predictions, labels):
    '''
    获取准确率
    
    Args:
    predictions: 预测结果
    labels: 标签
    
    Returns:
    accuracy: 准确率
    '''
    print(predictions, labels)
    return np.sum(predictions == labels) / labels.size

# 保存模型
def save_model(Ws_h, Bs_h, W_o, B_o):
    '''
    保存模型
    分别保存所有Hidden Layer的权重和偏置，以及Output Layer的权重和偏置
    分别保存为 .npz 文件 和 .json 文件
    
    Args:
    Ws_h: 所有Hidden Layer的权重(weight)
    Bs_h: 所有Hidden Layer的偏置(bias)
    W_o: Output Layer的权重(weight)
    B_o: Output Layer的偏置(bias)
    
    Returns:
    
    '''
    np.savez("model.npz", Ws_h=Ws_h, Bs_h=Bs_h, W_o=W_o, B_o=B_o)
    params = {"Ws_h": Ws_h, "Bs_h": Bs_h, "W_o": W_o, "B_o": B_o}
    with open("model.json", "w") as json_file:
        json.dump(params, json_file)
    pass

# 定义梯度下降
def gradient_descend(tr_images, tr_labels, alpha, iterations, layers=1, neurons=10, final=10):
    '''
    梯度下降
    这是训练的主要过程，包括前向传播、反向传播、更新参数
    
    Args:
    
    Returns:
    
    '''
    Ws_h, Bs_h, W_o, B_o = initWB(layers, neurons, final)  # 初始化参数
    for i in range(iterations):
        Zs_h, As_h, Z_o, A_o = forward_prop(Ws_h, Bs_h, W_o, B_o, tr_images, ReLU) # 前向传播
        dWs_h, dBs_h, dW_o, dB_o = back_prop(Zs_h, As_h, Z_o, A_o, Ws_h, W_o, tr_images, tr_labels, ReLU_deriv) # 反向传播 
        Ws_h, Bs_h, W_o, B_o = update_params(Ws_h, Bs_h, W_o, B_o, dWs_h, dBs_h, dW_o, dB_o, alpha) # 更新参数
        if i % 10 == 0:
            print("Iteration: ", i)
            _, _, _, A_o = forward_prop(Ws_h, Bs_h, W_o, B_o, tr_images, ReLU)
            predictions = get_predictions(A_o)
            print(get_accuracy(predictions, tr_labels))
            pass
        pass
    save_model(Ws_h, Bs_h, W_o, B_o) # 保存模型
    return Ws_h, Bs_h, W_o, B_o

# 定义预测
def make_pred(image, Ws_h, Bs_h, W_o, B_o):
    '''
    预测
    
    Args:
    image: 输入数据
    Ws_h: 所有Hidden Layer的权重(weight)
    Bs_h: 所有Hidden Layer的偏置(bias)
    W_o: Output Layer的权重(weight)
    B_o: Output Layer的偏置(bias)
    
    Returns:
    predictions: 预测结果
    '''
    _, _, _, A_o = forward_prop(Ws_h, Bs_h, W_o, B_o, image, ReLU)
    predictions = get_predictions(A_o)
    return predictions

# 定义加载模型 从 .npz 文件中加载模型
def read_npz(path):
    '''
    从 .npz 文件中加载模型
    
    Args:
    path: .npz 文件的路径
    
    Returns:
    Ws_h: 所有Hidden Layer的权重(weight)
    Bs_h: 所有Hidden Layer的偏置(bias)
    W_o: Output Layer的权重(weight)
    B_o: Output Layer的偏置(bias)
    '''
    npzfile = np.load(path)
    Ws_h = npzfile["Ws_h"]
    Bs_h = npzfile["Bs_h"]
    W_o = npzfile["W_o"]
    B_o = npzfile["B_o"]
    return Ws_h, Bs_h, W_o, B_o

# 定义加载模型 从 .json 文件中加载模型
def read_json(path):
    '''
    从 .json 文件中加载模型
    
    Args:
    path: .json 文件的路径
    
    Returns:
    Ws_h: 所有Hidden Layer的权重(weight)
    Bs_h: 所有Hidden Layer的偏置(bias)
    W_o: Output Layer的权重(weight)
    B_o: Output Layer的偏置(bias)
    '''
    # TODO: 从 .json 文件中加载模型

# 定义测试模型
def test_pred(index, Ws_h, Bs_h, W_o, B_o):
    c_image = te_images[:, index].reshape(784, 1) # 获取测试集中的第index个样本
    c_label = te_labels[index] # 获取测试集中的第index个样本的标签
    prediction = make_pred(c_image, Ws_h, Bs_h, W_o, B_o) # 预测
    print("Prediction: ", prediction)
    print("Label: ", c_label)
    c_image = c_image.reshape((28, 28)) * 255 # 将图像数据转换为28*28的矩阵
    plt.gray()
    plt.imshow(c_image, interpolation="nearest") # 显示图像
    plt.show()
    return

Ws_h, Bs_h, W_o, B_o = gradient_descend(tr_images, tr_labels, 0.10, 500, 1, 10, 10)