#preprocessing for chatbot dataset
import pandas as pd
import os
import pickle
import random
import numpy as np
from konlpy.tag import *
from collections import Counter
import concurrent.futures

word_to_id = {}
id_to_word = {}
corpus = []

#my stop_words list
stop_words = ['.', '_', '?', '<sos>', '<eos>']

#add external stop_words list
def get_external_stopwords():
    global stop_words
    with open('../dataset/korean_stopwords.txt', 'r', encoding='utf-8') as f:
        b = f.read().splitlines()
        stop_words += b

def divide_test_train(data, test_rate=0.2):
    pivot = int(test_rate*len(data))+1
    test = data[:pivot]
    train = data[pivot:]
    return test, train

# bind id with whole words in corpus
def bind_id_all(qdata, adata):

    tdata = qdata+adata
    max_len = max(len(item) for item in tdata)

    words = [word for word in sum(tdata,[]) if len(word) > 0]
    words = list(set(words)) #remove duplication
    counts = Counter(words)
    sorted_keys = sorted(counts.items(), key=lambda x:x[1], reverse=True)

    word_to_id[''] = -1
    id_to_word[-1] = ''

    idx=1
    for i,d in sorted_keys:
        word_to_id[i] = idx
        id_to_word[idx] = i
        idx+=1

    return word_to_id, id_to_word, max_len

# parameter target_data[[]]
# padding 0 for post only
def convert_data(target_data, size=10, scaled_size=True, padding='post', padding_num = 0):
    result = []
    tmp = []

    for dd in target_data:
        tmp.clear()
        if padding == 'front':
            if scaled_size:
                left = size - len(dd)
                for i in range(left):
                    tmp.append(padding_num)

        for d in dd:
            tmp.append(word_to_id[d])

        if padding == 'post':
            if scaled_size:
                left = size - len(dd)
                for i in range(left):
                    tmp.append(padding_num)

        if len(tmp) > size:
            tmp = tmp[:size]

        result.append(tmp.copy())

    return result

def encode(input_str, size):
    okt = Okt()
    m = okt.morphs(input_str)
    m = [x for x in m if x in word_to_id.keys()]
    return convert_data([m], size=size)


# load data for chatbot
# scaled_size option is for seq2seq
# sos = start of sentence, eos = end of sentence
def load_data(file_name='../dataset/ChatbotData.csv', seed=1995, need_soseos=False, need_corpus=False,
              scaled_size=True, padding_num=-1, save=True, save_file_name='./pkl/qadf.pkl', time_size=35):
    random.seed(seed)
    # get_external_stopwords()

    df = pd.read_csv(file_name)
    if os.path.isfile('../dataset/myChatbotData.csv'):
        df2 = pd.read_csv('../dataset/myChatbotData.csv')
        df = pd.concat([df, df2])

    print("file loaded")

    okt = Okt()
    qdf, adf = [], []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in executor.map(okt.morphs, df['Q']):
            result = [f for f in result if f not in stop_words]
            # result = [w for w,t in okt.pos(' '.join(result)) if t!='Josa']
            qdf.append(result)
        for result in executor.map(okt.morphs, df['A']):
            result = [f for f in result if f not in stop_words]
            # result = [w for w, t in okt.pos(' '.join(result)) if t !='Josa']
            if need_soseos:
                result.insert(0, '<sos>')
                result.append('<eos>')
            adf.append(result)

    print("file parsed")

    #원래 encoder, decoder 각각 max_len을 따로 정하지만 여기서는 일상대화를 다루므로 같이 진행하였다.
    word_to_id, id_to_word, max_len = bind_id_all(qdf, adf)

    if max_len < time_size:
        max_len = time_size

    print("id binded")

    enc_q = convert_data(qdf, size=max_len, scaled_size=scaled_size, padding='post', padding_num=padding_num)
    enc_a = convert_data(adf, size=max_len, scaled_size=scaled_size, padding='post', padding_num=padding_num)

    print("words converted to ids")

    if need_corpus:
        for q,a in zip(enc_q, enc_a):
            corpus.append(q)
            corpus.append(a)

        print("corpus ready")

    tmp = [[x,y] for x, y in zip(enc_q, enc_a)]
    random.shuffle(tmp)
    enc_q = [n[0] for n in tmp]
    enc_a = [n[1] for n in tmp]

    print("words shuffled")

    with open(save_file_name, 'wb') as f:
        pickle.dump(np.array(enc_q), f)
        pickle.dump(np.array(enc_a), f)
        pickle.dump(word_to_id, f)
        pickle.dump(id_to_word, f)

    return np.array(enc_q), np.array(enc_a)

def load_preprocess(file_name):
    global word_to_id
    global id_to_word
    with open(file_name, 'rb') as f:
        enc_q = pickle.load(f)
        enc_a = pickle.load(f)
        word_to_id = pickle.load(f)
        id_to_word = pickle.load(f)

        return enc_q, enc_a



#example
if __name__=='__main__':
    enc_q, enc_a = load_data()
    print(enc_q)
    print(enc_a)