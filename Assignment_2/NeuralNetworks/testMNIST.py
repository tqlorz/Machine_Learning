from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NN import *
from nn_train import nn_train
from nn_test import nn_test
import numpy as np
import os
import matplotlib.image as mpimg
import pickle
from nn_forward import nn_forward
from nn_predict import nn_predict
from nn_backward import nn_backward
from nn_applygradient import nn_applygradient
from function import sigmoid, softmax


def save_variable(v, filename):
    f = open(filename, "wb")
    pickle.dump(v, f)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, "rb")
    r = pickle.load(f)
    f.close()
    return r


# Training
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "storedMNIST.pkl")
dir_path = os.path.join(base_path, "MNIST", "train")
file_ls = os.listdir(dir_path)
data = np.zeros((60000, 784), dtype=float)
label = np.zeros((60000, 10), dtype=float)
flag = 0
for dir in file_ls:
    files = os.listdir(os.path.join(dir_path, dir))
    for file in files:
        filename = os.path.join(dir_path, dir, file)
        img = mpimg.imread(filename)
        data[flag, :] = np.reshape(img, -1) / 255
        label[flag, int(dir)] = 1.0
        flag += 1

ratioTraining = 0.95
xTraining, xValidation, yTraining, yValidation = train_test_split(
    data, label, test_size=1 - ratioTraining, random_state=0
)  # 随机分配数据集


if os.path.exists(model_path):
    nn = load_variable(model_path)
else:
    nn = NN(
        layer=[784, 400, 169, 49, 10],
        batch_normalization=1,
        active_function="relu",
        batch_size=50,
        learning_rate=0.001,
        optimization_method="RMSPropNesterov",
        objective_function="Cross Entropy",
    )

epoch = 0
maxAccuracy = 0
totalAccuracy = []
totalCost = []
maxEpoch = 100
while epoch < maxEpoch:
    epoch += 1
    nn = nn_train(nn, xTraining, yTraining)
    totalCost.append(sum(nn.cost.values()) / len(nn.cost.values()))
    wrongs, predictedLabel, accuracy, _ = nn_test(nn, xValidation, yValidation)
    totalAccuracy.append(accuracy)
    if accuracy > maxAccuracy:
        maxAccuracy = accuracy
        storedNN = nn
        save_variable(nn, model_path)
    cost = totalCost[epoch - 1]
    print("Epoch:", epoch)
    print("Accuracy:", accuracy)
    print("Cost:", cost)

# Testing
dir_path = os.path.join(base_path, "MNIST", "test")
file_ls = os.listdir(dir_path)
xTesting = np.zeros((10000, 784), dtype=float)
yTesting = np.zeros((10000, 10), dtype=float)
flag = 0
for dir in file_ls:
    files = os.listdir(os.path.join(dir_path, dir))
    for file in files:
        filename = os.path.join(dir_path, dir, file)
        img = mpimg.imread(filename)
        xTesting[flag, :] = np.reshape(img, -1) / 255
        yTesting[flag, int(dir)] = 1.0
        flag += 1
if os.path.exists(model_path):
    storedNN = load_variable(model_path)
    wrongs, predictedLabel, accuracy, _ = nn_test(storedNN, xTesting, yTesting)
    print("Accuracy on Test set:", accuracy)
    confusionMatrix = np.zeros((10, 10), dtype=int)
    for i in range(len(predictedLabel)):
        trueLabel = np.argmax(yTesting[i, :])
        confusionMatrix[trueLabel, predictedLabel[i]] += 1
    print("The Confusion Matrix is:\n", confusionMatrix)
