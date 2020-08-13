import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocessing, integration
from model.attention_seq2seq import AttentionSeq2seq
from model.seq2seq import Seq2seq
from model.peeky_seq2seq import PeekySeq2seq
# plt.rc('font', family='NanumGothic')

# 데이터 읽기
# x_train, t_train = preprocessing.load_data(file_name='../dataset/ChatbotData.csv')
x_train, t_train, model = integration.sum_att_models('../pkl/myAttentionSeq2seq.pkl', '../pkl/qadf.pkl', '../pkl/myAttentionSeq2seq2.pkl', '../pkl/qadf2.pkl')
#test 나누기
# x_test, x_train = preprocessing.divide_test_train(x_train, test_rate=0.1)
# t_test, t_train = preprocessing.divide_test_train(t_train, test_rate=0.1)

vocab_size = len(preprocessing.word_to_id)
wordvec_size = 300
hidden_size = 300

# model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
model.load_params('../pkl/myAttentionSeq2seq3.pkl')

print("'종료'를 입력하여 채팅을 종료")
while True:
    start_words = input()
    if start_words == '종료':
        break
    okt = preprocessing.Okt()
    morphs = okt.morphs(start_words)
    filtered = [i for i in morphs if i in preprocessing.word_to_id.keys()]
    enc = [preprocessing.word_to_id[w] for w in filtered]
    enc.insert(0, preprocessing.word_to_id['<sos>'])
    start_ids = np.array(enc)
    t, = start_ids.shape
    start_ids.shape = (1,t)

    print(' '.join([preprocessing.id_to_word[i] for i in model.generate(start_ids, start_id=preprocessing.word_to_id['<sos>'], sample_size=t)]))
