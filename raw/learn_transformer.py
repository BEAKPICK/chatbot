import numpy as np
import os
import matplotlib.pyplot as plt

from common.optimizer import  Adam
from common.optimizer import  SGD
from common.optimizer import  RMSprop
from common.trainer import Trainer
from common.util import eval_transformer

from preprocess import preprocessing
from model.transformer import Transformer

# time_size 설정
time_size = 35
padding_num = 0

if os.path.isfile("../pkl/myTransformer_preprocess.pkl"):
    x_train, t_train = preprocessing.load_preprocess('../pkl/myTransformer_preprocess.pkl')
else:
    x_train, t_train = preprocessing.load_data(file_name="../dataset/ChatbotData.csv",
                                               need_soseos=True,
                                               save_file_name='../pkl/myTransformer_preprocess.pkl',
                                               time_size=time_size,
                                               padding_num=padding_num)

# 파라미터 설정
vocab_size = len(preprocessing.id_to_word)
wordvec_size = 100 # d_model
head_size = 64
batch_size = 512
max_epoch = 20

x_test, x_train = preprocessing.divide_test_train(x_train, test_rate=0.2)
t_test, t_train = preprocessing.divide_test_train(t_train, test_rate=0.2)

# t_train에 <sos>, <eos> 모두 포함되어 있음.
# t_train = removeeos(t_train, padding_num=padding_num)

#실험: <sos>도 없애보자
# t_train = removesos(t_train, padding_num=padding_num)

model = Transformer(vocab_size, wordvec_size, head_size, batch_size=batch_size, num_heads=8, num_encoders=2, num_decoders=2)

if os.path.isfile("../pkl/myTransformer_params.pkl"):
    model.load_params("../pkl/myTransformer_params.pkl")

# optimizer = Adam()
# optimizer = SGD(lr=0.001)
optimizer = RMSprop(lr=0.01)
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, eval_interval=1)
    model.save_params('../pkl/myTransformer_params.pkl')

    correct_num = 0
    for i in range(len(x_test)//20):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_transformer(model, question, correct,
                                        preprocessing.id_to_word, preprocessing.word_to_id['<eos>'],
                                        verbose, is_reverse=False, padding_num=padding_num)

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