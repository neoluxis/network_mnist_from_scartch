import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os, json

# 读取训练数据
train_data = pd.read_csv("./datasets/train.csv")

train_data = np.array(train_data)
m, n = train_data.shape
np.random.shuffle(train_data)  # 洗牌，把训练数据打乱

data_dev = train_data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.0

data_train = train_data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape

# print(Y_train)


def init_params():
    W1 = np.random.rand(16, 784) - 0.5
    b1 = np.random.rand(16, 1) - 0.5
    W2 = np.random.rand(16, 16) - 0.5
    b2 = np.random.rand(16, 1) - 0.5
    W3 = np.random.rand(10, 16) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3
    

# 3 个激活函数, ReLU, Sigmoid, softmax
def ReLU(Z):
    return np.maximum(Z, 0)


def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def ReLU_deriv(Z):
    return Z > 0


def Sigmoid_deriv(Z):
    return Sigmoid(Z) * (1 - Sigmoid(Z))


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    # print(W2.T.dot(dZ2).shape, ReLU_deriv(Z1).shape)
    # raise Exception
    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A3):
    return np.argmax(A3, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def save_params(W1, b1, W2, b2, W3, b3):
    np.savez_compressed("params.npz", W1, b1, W2, b2, W3, b3)
    params = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "W3": W3.tolist(),
        "b3": b3.tolist(),
    }
    with open("params.readable.json", "w") as file:
        json.dump(params, file, indent=4)
    with open("params.min.json", "w") as file:
        json.dump(params, file)


def load_params(path):
    if not os.path.exists(path):
        raise Exception("No such file %s exists" % path)
    # if json file
    if path.endswith(".json"):
        with open(path) as file:
            params = json.load(file)
            W1 = np.array(params["W1"])
            b1 = np.array(params["b1"])
            W2 = np.array(params["W2"])
            b2 = np.array(params["b2"])
            W3 = np.array(params["W3"])
            b3 = np.array(params["b3"])
            return W1, b1, W2, b2, W3, b3
    # if npz file
    if path.endswith(".npz"):
        params = np.load(path)
        W1 = params["arr_0"]
        b1 = params["arr_1"]
        W2 = params["arr_2"]
        b2 = params["arr_3"]
        W3 = params["arr_4"]
        b3 = params["arr_5"]
        return W1, b1, W2, b2, W3, b3


def gradient_descent(X, Y, alpha, iterations, path=None):
    W1, b1, W2, b2, W3, b3 = init_params()
    if path:
        try:
            W1, b1, W2, b2, W3, b3 = load_params(path)
        except Exception as e:
            print(e)
            print("Loading default params")
            W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(
            Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y
        )
        W1, b1, W2, b2, W3, b3 = update_params(
            W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha
        )
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    save_params(W1, b1, W2, b2, W3, b3)
    return W1, b1, W2, b2, W3, b3


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions


def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    # plt.show()
    return prediction[0] == label


if __name__ == "__main__":
    # Train
    # W1, b1, W2, b2, W3, b3 = gradient_descent(
    #     X_train, Y_train, alpha=0.10, iterations=4000, path="params.npz" # 继续训练
    # )
    # W1, b1, W2, b2, W3, b3 = gradient_descent(
    #     X_train, Y_train, alpha=0.10, iterations=4000 # 重新训练
    # )
    # corr = 0
    # samples = 100
    # for i in range(samples):
    #     if test_prediction(i, W1, b1, W2, b2, W3, b3):
    #         corr+=1
    #         pass
    # dev_predictions = make_predictions(X_dev, W1, b1, W2, b2, W3, b3)
    # # get_accuracy(dev_predictions, Y_dev)
    # print("Accuracy on dev set: ", corr/samples)

    # Test
    # W1, b1, W2, b2, W3, b3 = load_params("params.npz")
    # data_verify = train_data.T
    # Y_verify = data_verify[0]
    # X_verify = data_verify[1:n]
    # X_verify = X_verify / 255.0
    # _, m_verify = X_verify.shape
    # corr = 0
    # samples = 20000
    # for i in range(samples):
    #     # print(i)
    #     if test_prediction(i, W1, b1, W2, b2, W3, b3):
    #         corr += 1
    # print(f"Accuracy on verify set: {corr}/{samples}={corr / samples} ")

    # Predict
    import csv
    pred_data = pd.read_csv("./datasets/test.csv")
    pred_data = np.array(pred_data)
    m_pred, n_pred = pred_data.shape
    X_pred = pred_data.T
    X_pred = X_pred / 255.0
    W1, b1, W2, b2, W3, b3 = load_params("results/layer2/16/params.npz")
    predictions = make_predictions(X_pred, W1, b1, W2, b2, W3, b3)
    with open("predictions.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["ImageId", "Label"])
        for i in range(m_pred):
            writer.writerow([i + 1, predictions[i]])
    pass
