import sys
sys.path.append('..')
import numpy as np
from preprocess import preprocessing
import matplotlib.pyplot as plt
from model.attention_seq2seq import AttentionSeq2seq
from model.seq2seq import Seq2seq
from model.peeky_seq2seq import PeekySeq2seq
# plt.rc('font', family='NanumGothic')

# 데이터 읽기
x_train, t_train = preprocessing.load_data(file_name='../dataset/ChatbotData.csv')

#test 나누기
x_test, x_train = preprocessing.divide_test_train(x_train, test_rate=0.1)
t_test, t_train = preprocessing.divide_test_train(t_train, test_rate=0.1)

vocab_size = len(preprocessing.word_to_id)
wordvec_size = 16
hidden_size = 256

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
model.load_params('./pkl/myPeekySeq2seq.pkl')

start_words = '12시 땡'
start_ids = np.array([preprocessing.word_to_id[w] for w in start_words.split(' ')])
t, = start_ids.shape
start_ids.shape = (1,t)

print(' '.join([preprocessing.id_to_word[i] for i in model.generate(start_ids, start_id=preprocessing.word_to_id['<sos>'], sample_size=5)]))