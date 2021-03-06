# coding: utf-8
from common.np import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    # 3차원이라면 x(N,T,D)가 들어온 것
    elif x.ndim == 3:
        N, _, _ = x.shape
        for n in range(N):
            x[n] = x[n] - x[n].max(axis=1, keepdims=True)
            x[n] = np.exp(x[n])
            x[n] = x[n]/x[n].sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    
    #0에 가까울수록 마이너스 무한대로 가기 때문에 이를 막기 위해 1e-7라는 작은 수를 더한다.
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
