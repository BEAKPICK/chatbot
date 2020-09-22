import numpy as np

from common.time_layers import TimeSoftmaxWithLoss
from common.time_layers import PositionalEmbedding

from common.layers import QueryKeyValueMatMul
from common.layers import MatMul
from common.layers import AddNorm
from common.layers import FFNN

from common.base_model import BaseModel

class ScaledDotAttention:

    def __init__(self, wordvec_size, head_size, mask=False):
        # S for vocab size(omit), D for wor'D'vector size, H for 'H'ead size
        D, H = wordvec_size, head_size
        rn = np.random.randn

        query_W = (rn(D, H) / np.sqrt(D)).astype('f')
        key_W = (rn(D, H) / np.sqrt(D)).astype('f')
        value_W = (rn(D, H) / np.sqrt(D)).astype('f')

        self.querykeyvalue = QueryKeyValueMatMul(query_W, key_W, value_W, mask=mask)

        self.params = self.querykeyvalue.params
        self.grads = self.querykeyvalue.grads

        return None
    # qk는 encoder로 부터 넘어온 값이 있을 때(qk parameter for decoder only)
    def forward(self, xs, qk=None):
        # xs->N,(T,D), qkvs->N,(T,H)
        qkvs = self.querykeyvalue.forward(xs, qk)
        return qkvs

    def backward(self, dz):
        # dz->N,(T,H), dout->N,(T,D)
        dout = self.querykeyvalue.backward(dz)
        return dout

class MultiHeadAttention:

    def __init__(self, wordvec_size, head_size, num_heads, mask=False):
        D, H = wordvec_size, head_size
        rn = np.random.randn

        # create multiple ScaledDotAttention layer
        # save params, grads
        self.layers = []
        self.params, self.grads = [], []
        for _ in range(num_heads):
            layer = ScaledDotAttention(wordvec_size=D,
                                       head_size=H,
                                       mask=mask)
            self.layers.append(layer)
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

        # 편의를 위해 저장
        self.num_heads = num_heads
        self.head_size = head_size

        # declare w0 (weight for multi head concat)
        # W0를 params에서만 빼다 쓰면 저장된 layer의 params의 개수를 알고 있어야하기 때문에
        # 편의를 위해 따로 저장
        self.W0 = (rn(head_size*num_heads, D)/np.sqrt(head_size*num_heads)).astype('f')
        self.params.append(self.W0)
        self.grads.append(np.zeros_like(self.W0))

    # decoder의 경우 qkW에 weight이 들어있다.
    def forward(self, xs, qk=None):
        # xs->N,(T,D)

        # get forwards from multiple scaleddotattention
        fwds = []
        for layer in self.layers:
            # ScaledDotAttentionEncoder.forward()->N,(T,H)
            fwds.append(layer.forward(xs, qk))

        # h_concat->N,(T,num_heads*H)
        h_concat = np.concatenate([i for i in fwds], axis=2)
        self.x = h_concat

        # w0->(num_heads*H, D)
        N,T,D = xs.shape
        h_linear = np.empty((N,T,D), dtype='f')

        for n in range(N):
            h_linear[n] = np.dot(h_concat[n], self.W0)

        # h_linear->(N,T,D)
        return h_linear

    def backward(self, dout):
        # dout->N,(T,D), self.x->N,(T,num_heads*H)
        N,_,_ = dout.shape
        # dW0->N,(num_heads*H, D)
        dW0 = np.matmul(np.transpose(self.x, (0,2,1)), dout)
        # W0의 grads는 가장 마지막에 있다.
        self.grads[-1][...] = np.nansum(dW0, axis=0)

        # dout->N,(T,D) / w0->(num_heads*H, D)
        dx = np.matmul(dout, self.W0.T)
        # dx->N,(T,num_heads*H)

        # dx를 head_size씩 num_heads만큼 잘라 각 ScaledDotAttention에 넘겨준다.
        cursor = 0
        ix = np.zeros_like(dout)
        for layer in self.layers:
            ix += layer.backward(dx[:,:,cursor:cursor+self.head_size])
            cursor+=self.head_size
        # self.embed.backward(ix/len(self.layers)) ->참고후 삭제

        return ix/len(self.layers)

class TransformerEncoder:
    def __init__(self, wordvec_size, head_size, num_heads, d_ffnn=64):
        D, H = wordvec_size, head_size
        rn = np.random.randn
        self.multiheadattention = MultiHeadAttention(wordvec_size=wordvec_size,
                                                     head_size=head_size,
                                                     num_heads=num_heads)

        self.addnorm1 = AddNorm((rn(D,D)/np.sqrt(D)).astype('f'))

        ffnn_W1 = (rn(D,d_ffnn)/np.sqrt(D)).astype('f')
        ffnn_W2 = (rn(d_ffnn, D) / np.sqrt(d_ffnn)).astype('f')
        ffnn_b1 = np.zeros(d_ffnn).astype('f')
        ffnn_b2 = np.zeros(D).astype('f')
        self.ffnn = FFNN(ffnn_W1, ffnn_b1, ffnn_W2, ffnn_b2)

        self.addnorm2 = AddNorm((rn(D,D)/np.sqrt(D)).astype('f'))

        self.params = self.multiheadattention.params + self.ffnn.params \
                      + self.addnorm1.params + self.addnorm2.params
        self.grads = self.multiheadattention.grads + self.ffnn.grads \
                     + self.addnorm1.grads + self.addnorm2.grads

    def forward(self, xs):
        # xs->N,(T,D)
        mx = self.multiheadattention.forward(xs)
        # mx->N,(T,D)
        an1x = self.addnorm1.forward(mx, xs)
        # an1x->N,(T,D)
        fx = self.ffnn.forward(an1x)
        # fx->N,(T,D)
        an2x = self.addnorm2.forward(fx, an1x)

        return an2x

    def backward(self, dout):
        # dout->N,(T,D)
        an2dx = self.addnorm2.backward(dout)
        # an2dx->N,(T,D)
        fdx = self.ffnn.backward(an2dx)
        # fdx->N,(T,D)
        an1dx = self.addnorm1.backward(fdx)
        # an1dx->N,(T,D)
        mdx = self.multiheadattention.backward(an1dx)
        # mdx->N,(T,D)
        return mdx

    def generate(self, xs):
        return self.forward(xs)

class TransformerDecoder:
    def __init__(self, wordvec_size, head_size, num_heads, d_ffnn=64):
        D, H = wordvec_size, head_size
        rn = np.random.randn

        self.maskedmultiheadattention = MultiHeadAttention(wordvec_size=D,
                                                           head_size=H,
                                                           num_heads=num_heads,
                                                           mask=True)
        self.addnorm1 = AddNorm((rn(D, D) / np.sqrt(D)).astype('f'))
        # decoder의 multiheadattention는 input을 encoder로 부터 받는다.
        self.multiheadattention = MultiHeadAttention(wordvec_size=D,
                                                     head_size=H,
                                                     num_heads=num_heads)
        self.addnorm2 = AddNorm((rn(D, D) / np.sqrt(D)).astype('f'))
        ffnn_W1 = (rn(D, d_ffnn) / np.sqrt(D)).astype('f')
        ffnn_W2 = (rn(d_ffnn, D) / np.sqrt(d_ffnn)).astype('f')
        ffnn_b1 = np.zeros(d_ffnn).astype('f')
        ffnn_b2 = np.zeros(D).astype('f')
        self.ffnn = FFNN(ffnn_W1, ffnn_b1, ffnn_W2, ffnn_b2)

        self.addnorm3 = AddNorm((rn(D, D) / np.sqrt(D)).astype('f'))

        self.params = self.maskedmultiheadattention.params + self.ffnn.params \
                      + self.addnorm1.params + self.addnorm2.params + self.addnorm3.params
        self.grads = self.maskedmultiheadattention.grads + self.ffnn.grads \
                     + self.addnorm1.grads + self.addnorm2.grads + self.addnorm3.grads

        return None
    def forward(self, xs, qk):
        # mmx->N,(T,D)
        mmx = self.maskedmultiheadattention.forward(xs)
        # an1x->N,(T,D)
        an1x = self.addnorm1.forward(xs, mmx)
        # mx->N,(T,D)
        mx = self.multiheadattention.forward(an1x, qk)
        # an2x->N,(T,D)
        an2x = self.addnorm2.forward(mx, an1x)
        # fx->N,(T,D)
        fx = self.ffnn.forward(an2x)
        # an3x->N,(T,D)
        an3x = self.addnorm3.forward(fx, an2x)
        return an3x

    def backward(self, dout):
        # dout->N,(T,D)
        dout = self.addnorm3.backward(dout)
        dout = self.ffnn.backward(dout)
        dout = self.addnorm2.backward(dout)
        # ddout->N,(T,D)
        ddout = self.multiheadattention.backward(dout)
        dout = self.addnorm1.backward(ddout)
        dout = self.maskedmultiheadattention.backward(dout)

        return ddout, dout

    def generate(self, xs):
        # xs->1,(T,D)
        # xs = xs[np.newaxis, :]
        # mmx->1,(T,D)
        mmx = self.maskedmultiheadattention.forward(xs)
        # an1x->1,(T,D)
        an1x = self.addnorm1.forward(xs, mmx)
        # fx->1,(T,D)
        fx = self.ffnn.forward(an1x)
        # an3x->1,(T,D)
        an3x = self.addnorm3.forward(fx, an1x)
        return an3x


class Transformer(BaseModel):
    def __init__(self, vocab_size, wordvec_size, head_size, num_heads, num_encoders=3, num_decoders=3):
        S, D, H = vocab_size, wordvec_size, head_size
        rn = np.random.randn

        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.params, self.grads =[],[]

        # Double embed (encoder, decoder)
        embed_W1 = (rn(S, D) / 100).astype('f')
        self.e_embed = PositionalEmbedding(embed_W1)
        self.params+=self.e_embed.params
        self.grads+=self.e_embed.grads

        self.encoders, self.decoders = [], []
        for _ in range(num_encoders):
            te = TransformerEncoder(wordvec_size=D,
                                   head_size=H,
                                   num_heads=num_heads)
            self.encoders.append(te)
            self.params+=te.params
            self.grads+=te.grads

        for _ in range(num_decoders):
            td = TransformerDecoder(wordvec_size=D,
                                    head_size=H,
                                    num_heads=num_heads)
            self.decoders.append(td)
            self.params+=td.params
            self.grads+=td.grads

        # 편의를 위해 linear 변수에 따로 weight 저장
        self.linear = MatMul((rn(D, S)/np.sqrt(D)).astype('f'))
        self.params+=self.linear.params
        self.grads+=self.linear.grads
        
        # TimeSoftmaxWithLoss도 params와 grads가 있으나 사용되지 않기때문에 생략
        self.softmax = TimeSoftmaxWithLoss(ignore_label=-1)

    def forward(self, xs, ts):
        # xs->(N,T) / eout, dout, ts->N,(T,D)
        eout = self.e_embed.forward(xs)
        dout = self.e_embed.forward(ts)
        N, T, D = eout.shape

        for encoder in self.encoders:
            eout = encoder.forward(eout)
        for decoder in self.decoders:
            ts = decoder.forward(dout, eout)

        ts = ts.reshape(N*T,D)
        # score->(N*T,S)
        score = self.linear.forward(ts)
        _,S = score.shape
        # 순서 주의 score는 linear된 2차원 행렬, xs는 임베딩되기전 2차원 행렬
        # loss->(N*T,1)
        score = score.reshape(N,T,S)
        loss = self.softmax.forward(score, xs)
        return loss

    def backward(self, dout=1):
        # dout->N,(T,S)
        dout = self.softmax.backward(dout)
        N,T,S = dout.shape
        dout = dout.reshape(N*T,S)
        # dout->(N*T,S) / self.linear.W->(D,S)
        dout = self.linear.backward(dout)
        # dout->(N*T,D)
        _,D = dout.shape
        dout = dout.reshape(N,T,D)

        # ddout->N,(T,D)
        for i in range(self.num_decoders-1,0,-1):
            _, dout = self.decoders[i].backward(dout)
        ddout, dout = self.decoders[0].backward(dout)

        # dout->N,(T,D)
        for i in range(self.num_encoders-1, -1, -1):
            ddout = self.encoders[i].backward(ddout)

        self.e_embed.backward(ddout)

    def generate(self, xs, type='GPT'):
        sampled = []
        # 'GPT'는 transformer의 decoder만 이용
        if type == 'GPT':
            # xs->(T,), out->(T,D)
            out = self.e_embed.forward(xs)
            # out->(1,T,D)
            # out = out[np.newaxis,:]
            for i in range(self.num_decoders):
                out = self.decoders[i].generate(out)
            # out->(1,T,D)
            N, T, D = out.shape
            out = out.reshape(N * T, D)
            # score->(1,T,S)
            score = self.linear.forward(out)

            sampled = np.argmax(score, axis=-1).flatten()

        # 'BERT'는 transformer의 encoder만 이용
        # 하지만 아직 masking 처리가 되어있지 않은 구조고
        # positional embedding 이외에 segment embedding이 추가되어야함
        # 따라서 현재 이 코드에서 BERT는 사용하는 의미가 없으며 GPT를 이용해야함
        elif type == 'BERT':
            # xs->(T,), out->(T,D)
            out = self.e_embed.forward(xs)
            # out->(1,T,D)
            out = out[np.newaxis,:]
            for i in range(self.num_encoders):
                out = self.encoders[i].generate(out)

            # decoder의 linear를 그대로 이용하기로 하자
            N, T, D = out.shape
            out = out.reshape(N * T, D)
            # score->(1,T,S)
            score = self.linear.forward(out)

            sampled = np.argmax(score, axis=-1).flatten()
        else:
            print('invalid generate type')

        return sampled