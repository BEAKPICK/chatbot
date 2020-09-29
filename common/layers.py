# coding: utf-8
from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error
from common.util import pos_Encoding
from common.util import masking
from common.util import normalization
from common.functions import relu

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

class QueryKeyMatMul:
    # x,query/x,key/query,key 이렇게 matmul이 3개 있는 형태를 한꺼번에 처리
    # query, key -> array
    def __init__(self, qW, kW):
        self.params = [qW, kW]
        # query, key -> (D,H)
        self.grads = [np.zeros_like(qW), np.zeros_like(kW)]
        self.x = None

    def forward(self, x, key=None):
        # x->N,(T,D), qW, kW->(D,H)
        N, T, D = x.shape
        # query, key의 weight
        qW, kW, = self.params
        H = qW.shape[1]

        # batch에 해당되는 맨앞 숫자는 괄호 밖으로 표현
        # q_out, k_out -> N,(T,H)
        q_out, k_out = np.empty((N, T, H), dtype='f'), np.empty((N, T, H), dtype='f')

        for n in range(N):
            q_out[n] = np.dot(x[n,:,:], qW)
            if key is not None:
                k_out[n] = np.dot(key[n, :, :], kW)
            else:
                k_out[n] = np.dot(x[n,:,:], kW)

        # q_out, k_out have to be saved for backpropagation
        # mQ -> N,(T,H)
        # mK -> N,(T,H)
        self.mQ = q_out
        self.mK = k_out

        # x -> (N,T,D)
        self.x = x

        # return N,(T,T)
        return np.matmul(q_out, np.transpose(k_out, (0,2,1)))

    def backward(self, dout):
        # dout -> N,(T,T)
        N, T, _ = dout.shape
        self.x = self.x.reshape((N*T,-1))
        # self.x->(N*T,D)
        qW, kW, = self.params

        # dmQ, dmK -> N,(T,H)
        dmQ = np.matmul(dout, self.mK)
        dmK = np.matmul(dout, self.mQ)

        # dmQ, dmK -> (N*T,H)
        dmQ = dmQ.reshape((N*T,-1))
        dmK = dmK.reshape((N*T,-1))

        # dqW, dkW -> (D,H)
        dqW = np.dot(self.x.T, dmQ)
        dkW = np.dot(self.x.T, dmK)

        # dmQ -> (N*T,H), qW.T -> (H,D)
        dx1 = np.dot(dmQ, qW.T)
        # dmk -> (N*T,H), kW.T -> (H,D)
        dx2 = np.dot(dmK, kW.T)

        # dx1, dx2-> N,(T,D)
        dx1 = dx1.reshape((N,T,-1))
        dx2 = dx2.reshape((N,T,-1))

        # what we want to update is qW and kW
        self.grads[0][...] = dqW / N
        self.grads[1][...] = dkW / N

        return dx1, dx2

class ValueMatMul:
    def __init__(self, vW):
        self.params = [vW]
        self.grads = [np.zeros_like(vW)]
        self.x = None

    def forward(self, x):
        # x->N,(T,D), vW->(D,H)
        N,T,D = x.shape
        vW = self.params[0]
        H = vW.shape[1]
        self.x = x

        # x->N,(T,D), vW->(D,H)
        self.x = self.x.reshape(N*T,-1)
        v_out = np.dot(self.x, vW)
        v_out = v_out.reshape((N,T,-1))
        self.x = self.x.reshape((N,T,D))

        return v_out

    def backward(self, dout):
        # dout->N,(T,H), vW->(D,H)
        vW = self.params[0]
        N,T,H = dout.shape
        D = vW.shape[0]

        # self.x->(N*T,D) / dout->(N*T,H)
        self.x = self.x.reshape((N*T,-1))
        dout = dout.reshape((N*T,-1))

        # self.x.T->(D, N*T) / dvW->(D,H)
        dvW = np.dot(self.x.T, dout)
        self.grads[0][...] = dvW / N

        # vW.T->(H,D)
        dx_out = np.dot(dout, vW.T)
        # dx_out->N,(T,D)
        dx_out = dx_out.reshape((N,T,-1))

        return dx_out

class QueryKeyValueMatMul:

    def __init__(self, query, key, value):
        self.QKMatMul = QueryKeyMatMul(query, key)
        self.VMatMul = ValueMatMul(value)
        self.QKSoftmax = Softmax()
        self.params = self.QKMatMul.params + self.VMatMul.params
        self.grads = self.QKMatMul.grads + self.VMatMul.grads

    def forward(self, x, kv=None, mask=False):
        # x->N,(T,D)
        # qkx->N,(T,T), vx->N,(T,H)

        # vx->N,(T,H), extract value matmul
        if kv is not None:
            vx = self.VMatMul.forward(kv)
        else:
            vx = self.VMatMul.forward(x)

        # extract query, key matmul
        if kv is not None:
            # kv->N,(T,D)
            qkx = self.QKMatMul.forward(x, key=kv)
        else:
            qkx = self.QKMatMul.forward(x)
        # scaling
        _, T, H = vx.shape
        qkx = qkx/np.sqrt(H)

        # masking
        qkx = masking(qkx, mask=mask)
        # softmax
        qkx = self.QKSoftmax.forward(qkx)

        self.qkx = qkx
        self.vx = vx

        # return N,(T,H)
        return np.matmul(qkx, vx)

    def backward(self, dout):
        # dout->N,(T,H)
        # dqkx->N,(T,T)
        # dvx->N,(T,H)
        N,T,_ = dout.shape

        dqkx = np.matmul(dout, np.transpose(self.vx, (0,2,1)))
        dqkx = self.QKSoftmax.backward(dqkx)
        _,T,H = dout.shape
        dqkx = dqkx/np.sqrt(H)
        dvx = np.matmul(np.transpose(self.qkx,(0,2,1)), dout)

        dx1, dx2 = self.QKMatMul.backward(dqkx)
        dx3 = self.VMatMul.backward(dvx)

        return dx1, dx2, dx3

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

# Feed Forward Neual Network
class FFNN:
    def __init__(self, W1, b1, W2, b2):
        # W1->(D,df) df for any
        # W2->(df,D)
        self.params = [W1, b1, W2, b2]
        self.grads = [np.zeros_like(W1), np.zeros_like(b1),
                      np.zeros_like(W2), np.zeros_like(b2)]
        self.x = None

    def forward(self, x):
        # x->N,(T,D)
        N,T,D = x.shape
        self.x1 = x
        W1, b1, W2, b2 = self.params
        # out1->(N*T,Df)
        out1 = np.dot(x.reshape(N*T,D), W1) + b1
        out1 = relu(out1)
        # x2->(N*T,Df)
        self.x2 = out1.reshape((N,T,-1))
        # out2->(N*T, D)
        out2 = np.dot(out1, W2) + b2
        out2 = out2.reshape((N,T,-1))
        return out2

    def backward(self, dout):
        W1, b1, W2, b2 = self.params
        # dout->N,(T,D) / W1->(D,Df) / W2->(Df,D)
        N,T,D = self.x1.shape
        dout = dout.reshape(N*T,-1)
        # dout->(N*T,D) / W2.T->(D,Df) / dx2->(N*T,Df)
        dx2 = np.dot(dout, W2.T)
        # dx2->(N*T,Df)
        dx2 = relu(dx2)
        self.x2 = self.x2.reshape(N*T,-1)
        # self.x2->(N*T,Df) / dout->(N*T,D) / dW2->(Df,D)
        dW2 = np.dot(self.x2.T, dout)
        # db2->(D,)
        db2 = np.sum(dout, axis=0)
        # dW2->(D,D)
        dx1 = np.dot(dx2, W1.T)
        # dx1->(N*T,D)
        dx1 = dx1.reshape((N,T,-1))
        self.x1 = self.x1.reshape(N*T,-1)
        # self.x1->(N*T,D) / dW1->(D,Df)
        dW1 = np.dot(self.x1.T, dx2)
        # db1->(Df,)
        db1 = np.sum(dx2, axis=0)

        self.grads[0][...] = dW1 / N
        self.grads[1][...] = db1 / N
        self.grads[2][...] = dW2 / N
        self.grads[3][...] = db2 / N

        return dx1

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=-1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=-1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        #dout은 현재 기울기 값을 나타낸다. 초기값은 보다시피 1로 설정되어있다.
        dx = (self.y - self.t) * dout / batch_size
        return dx


class Dropout:
    '''
    http://arxiv.org/abs/1207.0580
    '''
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Embedding:
    def __init__(self, W, padding_num=0):
        self.padding_num = padding_num
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        # padding_num의 벡터는 고정된 벡터이므로 역전파를 시키지 않는다.
        np.add.at(dW, self.idx, dout)
        dW[self.padding_num] = 0
        return None


class AddNorm:
    def __init__(self, W, b):
        pass
        # W, b->(N,)
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]

    def forward(self, x1, x2):
        eps = 1e-5
        # aW, ab->(D,)
        aW = self.params[0]
        ab = self.params[1]

        # x1 and x2 must be in same shape
        # x_out->N,(T,D)
        x_out = x1+x2
        N,_,_ = x_out.shape
        self.x = x_out
        self.stds, self.means = [], []
        for x in x_out:
            self.stds.append(np.sqrt(x.var()+eps))
            self.means.append(np.mean(x))

        # self.stds = np.array(self.stds)
        self.means = np.array(self.means)
        # self.ivar = 1./self.stds
        self.norm = normalization(x_out)

        for n in range(N):
            x_out[n] = (self.norm[n] * aW[n]) + ab[n]
        # x_out->N,(T,D)
        # x_out = np.matmul(x_out, aW)
        # x_out->N,(T,D)
        return x_out


    def backward(self, dout):
        # https://kimcando94.tistory.com/110
        # http://cthorey.github.io./backpropagation/ 를 참고하여 작성됨

        # dout->N,(T,D)
        N, T, D = dout.shape
        # aW, ab->(D,)
        aW = self.params[0]
        db = np.sum(dout, axis=(1,2))

        dW = np.empty(*aW.shape, dtype='f')
        for n in range(N):
            dW[n] = np.sum(dout[n]*self.norm[n])
        # dW = dW/N

        self.grads[0][...] = dW
        self.grads[1][...] = db

        # dxhat = np.empty(N)
        # for n in range(N):
        #     dxhat[n] = dout[n] * aW[n]
        # divar = np.empty(N)
        # for n in range(N):
        #     divar[n] = dxhat[n]*self.means[n]

        # dxmu1 = dxhat * self.ivar
        # dsqrtvar = -1. / (self.stds**2) * divar
        # dvar = 0.5 * 1. / self.stds * dsqrtvar
        # dsq = 1. /N * np.ones((N,T,D)) * dvar

        # dout = dout.reshape((N*T,-1))
        # dout->(N*T,D)
        # aW, = self.params
        # self.x->(N*T,D)
        # self.x = self.x.reshape((N*T,-1))
        # aW->(D,D) / daW->(D,D)
        # daW = np.dot(self.x.T, dout)
        # dxhat->(N*T,D)
        # dxhat = np.dot(dout, aW.T)

        # dxhat = dxhat.reshape((N,T,-1))
        # self.x = self.x.reshape((N,T,-1))
        # dout = dout.reshape((N,T,-1))

        # self.grads[0][...] = daW / N

        for n in range(N):
            # dout[n] = dout[n] / self.stds[n]
            # dout[n] = (1./N) * (1/self.stds[n]) * (N*dout[n] - np.nansum(dout[n], axis=0)
            #                                       - self.x[n]*np.nansum(dout[n]*self.x[n], axis=0))
            dout[n] = (1. / N) * aW[n] * (1 / self.stds[n]) * (N * dout[n] - np.nansum(dout[n], axis=0)
                                                - (self.x[n] - self.means[n]) * ((1. / self.stds[n]) ** 2)
                                                * np.nansum(dout[n] * (self.x[n] - self.means[n]), axis=0))
        return dout