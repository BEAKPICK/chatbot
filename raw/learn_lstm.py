# coding: utf-8
import sys
sys.path.append('..')
from common import config
# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
# ==============================================
# config.GPU = True
# ==============================================
from common.optimizer import SGD, Adam
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu
from dataset import ptb
from model.rnnlm import BetterRnnlm
from preprocess import preprocessing

# 하이퍼파라미터 설정
batch_size = 50
wordvec_size = 300
hidden_size = 300
time_size = 10
lr = 0.01
max_epoch = 5
max_grad = 5.0
dropout = 0.5

# 학습 데이터 읽기
preprocessing.load_data('../dataset/ChatbotData.csv', scaled_size=False)
corpus_val, corpus_train = preprocessing.divide_test_train(preprocessing.corpus, test_rate=0.1)
corpus_test, corpus_train = preprocessing.divide_test_train(corpus_train, test_rate=0.1)

if config.GPU:
    corpus = to_gpu(corpus_train)
    corpus_val = to_gpu(corpus_val)
    corpus_test = to_gpu(corpus_test)

vocab_size = len(preprocessing.word_to_id)

xs = sum(corpus_train,[])[:-1]
ts = sum(corpus_train,[])[1:]
corpus_val = sum(corpus_val,[])
corpus_test = sum(corpus_test,[])

model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
# optimizer = SGD(lr)
optimizer = Adam(lr=lr)
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float('inf')
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,
                time_size=time_size, max_grad=max_grad)

    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print('검증 퍼플렉서티: ', ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()
    print('-' * 50)

model.save_params('myLstm.pkl')

# 테스트 데이터로 평가
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('테스트 퍼플렉서티: ', ppl_test)