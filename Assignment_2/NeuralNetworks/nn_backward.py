import numpy as np

def nn_backward(nn, batch_y) :
    batch_y = batch_y.T
    m = nn.a[0].shape[1]
    f = nn.output_function
    if f == 'sigmoid' :
        nn.delta[nn.depth-1] = -(batch_y - nn.a[nn.depth-1]) * nn.a[nn.depth-1] * (1 - nn.a[nn.depth-1])
    if f == 'tanh' :
        nn.delta[nn.depth-1] = -(batch_y - nn.a[nn.depth-1]) * (1 - nn.a[nn.depth-1]**2)
    if f == 'softmax' :
        y = np.dot(nn.W[nn.depth - 2], nn.a[nn.depth - 2]) + np.tile(nn.b[nn.depth - 2], (1, m))
        nn.delta[nn.depth-1] = nn.a[nn.depth-1] - batch_y

    if nn.batch_normalization :
        x = np.dot(nn.W[nn.depth - 2], nn.a[nn.depth -2]) + np.tile(nn.b[nn.depth - 2], (1, m))
        x = (x - np.tile(nn.E[nn.depth -2], (1, m))) / np.tile(nn.S[nn.depth -2] + 0.0001*np.ones(nn.S[nn.depth - 2].shape), (1, m))
        temp = nn.delta[nn.depth-1] * x
        nn.Gamma_grad[nn.depth - 2] = sum(np.mean(temp, axis = 1))
        nn.Beta_grad[nn.depth - 2] = sum(np.mean(nn.delta[nn.depth-1], axis = 1))
        nn.delta[nn.depth - 1] = nn.Gamma[nn.depth - 2]*nn.delta[nn.depth-1] / np.tile((nn.S[nn.depth - 2] + 0.0001), (1, m))

    nn.W_grad[nn.depth - 2] = np.dot(nn.delta[nn.depth-1], nn.a[nn.depth - 2].T) / m + nn.weight_decay*nn.W[nn.depth - 2]
    nn.b_grad[nn.depth - 2] = np.array([np.sum(nn.delta[nn.depth-1], axis=1) / m]).T
    #因为np.sum()返回维度为(n,)，会让之后的加法操作错误，所以要转换为(n,1)维度矩阵，下面的也是一样

    f = nn.active_function
    for i in range(1, nn.depth - 1):
        k = nn.depth - i - 1
        if  f == 'sigmoid':
            nn.delta[k] = np.dot(nn.W[k].T, nn.delta[k + 1]) * nn.a[k] * (1 - nn.a[k])
        elif f == 'tanh':
            nn.delta[k] = np.dot(nn.W[k].T, nn.delta[k + 1]) * (1 - nn.a[k] ** 2)
        elif f == 'relu':
            nn.delta[k] = np.dot(nn.W[k].T,nn.delta[k + 1])* (nn.a[k] > 0)
        if nn.batch_normalization:
            x = np.dot(nn.W[k - 1], nn.a[k - 1]) + np.tile(nn.b[k - 1], (1, m))
            x = (x - np.tile(nn.E[k - 1], (1, m))) / np.tile(nn.S[k - 1] + 0.0001 * np.ones(nn.S[k - 1].shape), (1, m))
            temp = nn.delta[k] * x
            nn.Gamma_grad[k - 1] = sum(np.mean(temp, axis=1))
            nn.Beta_grad[k - 1] = sum(np.mean(nn.delta[k], axis=1))
            nn.delta[k] = (nn.Gamma[k - 1] * nn.delta[k]) / np.tile((nn.S[k - 1] + 0.0001), (1, m))
        nn.W_grad[k - 1] = np.dot(nn.delta[k], nn.a[k - 1].T) / m + nn.weight_decay * nn.W[k - 1]
        nn.b_grad[k - 1] = np.array([np.sum(nn.delta[k], axis=1) / m]).T
    return nn