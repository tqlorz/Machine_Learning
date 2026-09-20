import numpy as np
from nn_predict import nn_predict

def nn_test(nn,test_x,test_y):
    nn = nn_predict(nn,test_x)
    y_output = nn.a[nn.depth-1]
    y_output = y_output.T
    yPred = np.argmax(y_output, axis=1) #按行找出最大元素所在下标
    y = np.argmax(test_y, axis=1)
    wrongs = (yPred != y) #求预测与期望不相等的个数
    accuracy = 1-sum(wrongs)/test_y.shape[0]
    return wrongs, yPred, accuracy, y_output