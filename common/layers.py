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

    def forward(self, x):
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
            k_out[n] = np.dot(x[n,:,:], kW)

        # q_out, k_out have to be saved for backpropagation
        # mQ -> N,(T,H)
        # mK -> N,(H,T)
        self.mQ = q_out
        self.mK = np.transpose(k_out, (0,2,1))

        # x -> (N,T,D)
        self.x = x

        # return N,(T,T)
        return np.matmul(q_out, np.transpose(k_out, (0,2,1)))

    def backward(self, dout):
        # dout -> N,(T,T)
        N, _, _ = dout.shape
        qW, kW, = self.params

        # dmQ, dmK -> N,(T,H)
        dmQ = np.matmul(dout, np.transpose(self.mK, (0,2,1)))
        dmK = np.matmul(dout, self.mQ)

        # dqW, dkW -> N,(D,H)
        dqW = np.nansum(np.matmul(np.transpose(self.x, (0,2,1)), dmQ), axis=0)
        dkW = np.nansum(np.matmul(np.transpose(self.x, (0,2,1)), dmK), axis=0)

        # dmQ -> N,(T,H), qW.T -> (H,D)
        dx1 = np.matmul(dmQ, qW.T)
        # dmk -> N,(T,H), kW.T -> (H,D)
        dx2 = np.matmul(dmK, kW.T)

        # what we want to update is qW and kW
        self.grads[0][...] = dqW
        self.grads[1][...] = dkW

        # dx1, dx2->N,(T,D)
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
        v_out = np.empty((N,T,H), dtype='f')
        for n in range(N):
            v_out[n] = np.dot(x[n,:,:], vW)

        return v_out

    def backward(self, dout):
        # dout->N,(T,H), vW->(D,H)
        vW = self.params[0]
        N,T,H = dout.shape
        D = vW.shape[0]

        # self.x.T->N,(D,T) / dvW->(D,H)
        dvW = np.nansum(np.matmul(np.transpose(self.x, (0,2,1)), dout), axis=0)
        self.grads[0][...] = dvW

        dx_out = np.empty((N,T,D), dtype='f')

        # vW.T->(H,D)
        for n in range(N):
            dx_out[n] = np.dot(dout[n], vW.T)

        return dx_out

class QueryKeyValueMatMul:

    def __init__(self, query, key, value, mask=False):
        self.QKMatMul = QueryKeyMatMul(query, key)
        self.VMatMul = ValueMatMul(value)
        self.QKSoftmax = Softmax()
        self.params = self.QKMatMul.params + self.VMatMul.params
        self.grads = self.QKMatMul.grads + self.VMatMul.grads
        self.mask = mask

    def forward(self, x, qk=None):
        # x->N,(T,D)
        # qkx->N,(T,T), vx->N,(T,H)
        if qk is not None:
            # qk->N,(T,D)
            qkx = self.QKMatMul.forward(qk)
        else:
            qkx = self.QKMatMul.forward(x)
        # scaling
        _, T, _ = qkx.shape
        qkx = qkx/np.sqrt(T)
        # masking
        qkx = masking(qkx, mask=self.mask)
        # softmax
        self.QKSoftmax.forward(qkx)
        # vx->N,(T,H)
        vx = self.VMatMul.forward(x)

        self.qkx = qkx
        self.vx = vx

        # return N,(T,H)
        return np.matmul(qkx, vx)

    def backward(self, dout):
        # dout->N,(T,H)
        # dqkx->N,(T,T)
        # dvx->N,(T,H)
        dqkx = np.matmul(dout, np.transpose(self.vx, (0,2,1)))
        _,T,_ = dqkx.shape
        dqkx = dqkx/np.sqrt(T)
        dvx = np.matmul(np.transpose(self.qkx,(0,2,1)), dout)

        dx1, dx2 = self.QKMatMul.backward(dqkx)
        dx3 = self.VMatMul.backward(dvx)

        return (dx1+dx2+dx3)/3

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
        self.x1 = x
        W1, b1, W2, b2 = self.params
        out1 = np.dot(x, W1) + b1
        out1 = relu(out1)
        # x2->N,(T,D)
        self.x2 = out1
        out2 = np.dot(out1, W2) + b2
        return out2

    def backward(self, dout):
        # dout->N,(T,D) / W1->(D,Df) / W2->(Df,D)
        W1, b1, W2, b2 = self.params
        N, T, _ = dout.shape
        dx2 = np.dot(dout, W2.T)
        # dx2->N,(T,Df)
        dx2 = relu(dx2)
        dW2 = np.matmul(np.transpose(self.x2, (0,2,1)), dout)
        dW2 = np.nansum(dW2, axis=0)
        dx1 = np.dot(dx2, W1.T)
        dW1 = np.matmul(np.transpose(self.x1, (0,2,1)), dx2)
        dW1 = np.nansum(dW1, axis=0)

        db2 = np.nansum(np.sum(dout, axis=0), axis=0)
        db1 = np.nansum(np.sum(dx2, axis=0), axis=0)

        self.grads[0][...] = dW1
        self.grads[1][...] = db1
        self.grads[2][...] = dW2
        self.grads[3][...] = db2

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
            self.t = self.t.argmax(axis=1)

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
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]#행을 뽑는다..?
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None


class AddNorm:
    def __init__(self, W):
        # W->(D,D)
        self.params = [W]
        self.grads = [np.zeros_like(W)]

    def forward(self, x1, x2):
        eps = 1e-8
        # aW->(D,D)
        aW, = self.params
        # x1 and x2 must be in same shape
        # x_out->N,(T,D)
        x_out = x1+x2
        self.x = x_out
        self.stds = []
        for x in x_out:
            self.stds.append(np.sqrt(x.var()+eps))
        x_out = normalization(x_out)
        # x_out->N,(T,D)
        x_out = np.matmul(x_out, aW)
        # x_out->N,(T,D)
        return x_out


    def backward(self, dout):
        # dout->N,(T,D)
        N,_,_ = dout.shape
        aW, = self.params
        # aW->(D,D) / daW->N,(D,D)
        daW = np.matmul(np.transpose(self.x,(0,2,1)), dout)
        # dxhat->N,(T,D)
        dxhat = np.matmul(dout, aW.T)
        self.grads[0][...] = np.nansum(daW, axis=0)

        for n in range(N):
            # dout[n] = dout[n] / self.stds[n]
            dout[n] = (1./N) * (1/self.stds[n]) * (N*dxhat[n] - np.nansum(dxhat, axis=0)
                                                   - self.x[n]*np.nansum(dxhat[n]*self.x[n], axis=0))
        return dout