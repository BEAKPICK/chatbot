from preprocess import preprocessing
from model.attention_seq2seq import AttentionSeq2seq
from model.seq2seq import Seq2seq
from model.peeky_seq2seq import PeekySeq2seq

from common.np import *

def sum_enc_dec(model1_pkl, model1preprocess_pkl, model2_pkl, model2preprocess_pkl, wordvec_size=300, hidden_size=300, model='attention'):

    # activate embed and word_to_id from model1
    preprocessing.load_preprocess(model1preprocess_pkl)
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
    preprocessing.load_preprocess(model2preprocess_pkl)
    loaded_word_to_id2 = preprocessing.word_to_id

    tmp_model2 = None
    if model == 'attention':
        tmp_model2 = AttentionSeq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'peeky':
        tmp_model2 = PeekySeq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'seq2seq':
        tmp_model2 = Seq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)

    tmp_model2.load_params(model2_pkl)

    startidx, = tmp_model1.encoder.embed.W.shape

    for i in loaded_word_to_id2.keys():
        if i in list(loaded_word_to_id1.keys()):
            tmp_model1.encoder.embed.W[loaded_word_to_id1[i]] = \
                (tmp_model1.encoder.embed.W[loaded_word_to_id1[i]] + tmp_model2.encoder.embed.W[loaded_word_to_id2[i]]) / 2
            tmp_model1.decoder.embed.W[loaded_word_to_id1[i]] = \
                (tmp_model1.decoder.embed.W[loaded_word_to_id1[i]] + tmp_model2.decoder.embed.W[loaded_word_to_id2[i]]) / 2
        else:
            np.vstack((tmp_model1.encoder.embed.W, [tmp_model2.encoder.embed.W[loaded_word_to_id2[i]]]))
            np.vstack((tmp_model1.decoder.embed.W, [tmp_model2.decoder.embed.W[loaded_word_to_id2[i]]]))
            loaded_word_to_id1[i] = startidx
            startidx+=1

    new_model = None
    if model == 'attention':
        new_model = AttentionSeq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'peeky':
        new_model = PeekySeq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)
    elif model == 'seq2seq':
        new_model = Seq2seq(vocab_size=len(loaded_word_to_id1), wordvec_size=wordvec_size, hidden_size=hidden_size)

    new_model.encoder.embed.W = tmp_model1.encoder.embed.W
    new_model.decoder.embed.W = tmp_model1.decoder.embed.W

    return new_model