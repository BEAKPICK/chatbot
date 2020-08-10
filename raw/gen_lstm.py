import sys
sys.path.append('..')
from common.np import *
from model.rnnlmgen import BetterRnnlmGen
from preprocess import preprocessing

preprocessing.load_data('../dataset/ChatbotData.csv', scaled_size=False)
vocab_size = len(preprocessing.word_to_id)
corpus_size = len(preprocessing.corpus)

wordvec_size = 300
hidden_size = 300

model = BetterRnnlmGen(vocab_size=vocab_size, wordvec_size=wordvec_size, hidden_size=hidden_size)
model.load_params('./pkl/myLstm.pkl')

start_words = '오늘 너무 힘들어'
start_ids = [preprocessing.word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], sample_size=7)
# word_ids = start_ids[:-1] + word_ids
txt = ' '.join([preprocessing.id_to_word[i] for i in word_ids])
print(txt)