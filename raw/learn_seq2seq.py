# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import os
import matplotlib.pyplot as plt

from preprocess import preprocessing
from preprocess import integration
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq

from model.attention_seq2seq import AttentionSeq2seq
from model.seq2seq import Seq2seq
from model.peeky_seq2seq import PeekySeq2seq

from common import config
config.GPU = True

# 데이터 읽기
# x_train, t_train = preprocessing.load_data(file_name='./dataset/myChatbotData.csv', need_soseos=True, save_file_name='./pkl/qadf2.pkl')

# 시간절약
# x_train, t_train = preprocessing.load_preprocess('../pkl/qadf.pkl')
x_train, t_train, model = integration.sum_att_models('./pkl/myAttentionSeq2seq.pkl', './pkl/qadf.pkl', './pkl/myAttentionSeq2seq2.pkl', './pkl/qadf2.pkl')

#test 나누기
x_test, x_train = preprocessing.divide_test_train(x_train, test_rate=0.1)
t_test, t_train = preprocessing.divide_test_train(t_train, test_rate=0.1)

# 하이퍼파라미터 설정
# default wordvec_size = 300
# default hidden_size = 300
# default batch_size = 300
vocab_size = len(preprocessing.id_to_word)
wordvec_size = 300
hidden_size = 300
batch_size = 128
max_epoch = 20
max_grad = 5.0

# model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)

if os.path.isfile("./pkl/myAttentionSeq2seq3.pkl"):
    model.load_params("./pkl/myAttentionSeq2seq3.pkl")

# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
#
#
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
#
#

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    model.save_params('./pkl/myAttentionSeq2seq3.pkl')

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    preprocessing.id_to_word, verbose, is_reverse=False)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('정확도 %.3f%%' % (acc * 100))



# 그래프 그리기
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('에폭')
plt.ylabel('정확도')
plt.ylim(-0.05, 1.05)
plt.show()
