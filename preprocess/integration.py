import pandas as pd
import concurrent.futures

from preprocess import preprocessing
from model.attention_seq2seq import AttentionSeq2seq
from model.seq2seq import Seq2seq
from model.peeky_seq2seq import PeekySeq2seq

from common.np import *

def sum_att_models(model1_pkl, model1preprocess_pkl, model2_pkl, model2preprocess_pkl,
                wordvec_size=300, hidden_size=300, model='attention'):

    # activate embed and word_to_id from model1
    enc_q1, enc_a1 = preprocessing.load_preprocess(model1preprocess_pkl)
    loaded_word_to_id1 = preprocessing.word_to_id.copy()

    tmp_model1 = None
    if model == 'attention':
        tmp_model1 = AttentionSeq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'peeky':
        tmp_model1 = PeekySeq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'seq2seq':
        tmp_model1 = Seq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)
    tmp_model1.load_params(model1_pkl)

    # activate embed and word_to_id from model2
    enc_q2, enc_a2 = preprocessing.load_preprocess(model2preprocess_pkl)
    loaded_word_to_id2 = preprocessing.word_to_id


    tmp_model2 = None
    if model == 'attention':
        tmp_model2 = AttentionSeq2seq(vocab_size=len(loaded_word_to_id2), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'peeky':
        tmp_model2 = PeekySeq2seq(vocab_size=len(loaded_word_to_id2), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'seq2seq':
        tmp_model2 = Seq2seq(vocab_size=len(loaded_word_to_id2), wordvec_size=wordvec_size, hidden_size=hidden_size)

    tmp_model2.load_params(model2_pkl)

    startidx,_ = tmp_model1.encoder.embed.W.shape

    for i in loaded_word_to_id2.keys():
        # integrate word vector
        if i in list(loaded_word_to_id1.keys()):
            tmp_model1.encoder.embed.W[loaded_word_to_id1[i]] = \
                (tmp_model1.encoder.embed.W[loaded_word_to_id1[i]] + tmp_model2.encoder.embed.W[loaded_word_to_id2[i]]) / 2
            tmp_model1.decoder.embed.W[loaded_word_to_id1[i]] = \
                (tmp_model1.decoder.embed.W[loaded_word_to_id1[i]] + tmp_model2.decoder.embed.W[loaded_word_to_id2[i]]) / 2
        # add new word vector
        else:
            tmp_model1.encoder.embed.W = np.vstack((tmp_model1.encoder.embed.W, [tmp_model2.encoder.embed.W[loaded_word_to_id2[i]]]))
            tmp_model1.decoder.embed.W = np.vstack((tmp_model1.decoder.embed.W, [tmp_model2.decoder.embed.W[loaded_word_to_id2[i]]]))
            loaded_word_to_id1[i] = startidx
            enc_q2 = update_encode(enc_q2, loaded_word_to_id2[i], startidx)
            enc_a2 = update_encode(enc_a2, loaded_word_to_id2[i], startidx)
            startidx+=1

    #update word_to_id, id_to_word
    preprocessing.word_to_id = loaded_word_to_id1
    preprocessing.id_to_word = {x: y for y, x in preprocessing.word_to_id.items()}

    # update enc_q, enc_a
    enc_q1, enc_q2 = resize_padding(enc_q1, enc_q2)
    enc_a1, enc_a2 = resize_padding(enc_a1, enc_a2)
    enc_q1 = np.vstack((enc_q1, enc_q2))
    enc_a1 = np.vstack((enc_a1, enc_a2))

    # define new model with integrated embedding W
    new_model = None
    if model == 'attention':
        new_model = AttentionSeq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'peeky':
        new_model = PeekySeq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'seq2seq':
        new_model = Seq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)

    new_model.encoder.embed.W = tmp_model1.encoder.embed.W
    new_model.decoder.embed.W = tmp_model1.decoder.embed.W

    return enc_q1, enc_a1, new_model

def update_encode(data, num1, num2):
    size = len(data)
    for i in range(size):
        for n in range(len(data[i])):
            if data[i][n] == num1:
                data[i][n] = num2
    return data

def resize_padding(data1, data2, padding_num=-1):
    if data1[0].shape[0] > data2[0].shape[0]:
        size = data1[0].shape[0] - data2[0].shape[0]
        result = []
        for i in data2:
            tmp = list(i)
            for _ in range(size):
                tmp.append(padding_num)
            result.append(tmp)
        data2 = np.array(result)
        return data1, data2
    elif data1[0].shape[0] < data2[0].shape[0]:
        size = data2[0].shape[0] - data1[0].shape[0]
        result = []
        for i in data1:
            tmp = list(i)
            for _ in range(size):
                tmp.append(padding_num)
            result.append(tmp)
        data1 = np.array(result)
        return data1, data2
    else:
        return data1, data2