from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from preprocess import preprocessing
from attention import AttentionLayer

# seems cupy does not support GPU operation for some reason,
# would rather use tensorflow api until cupy problem solved

# if you need gpu check
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

class att_seq2seq_tf():

    def decode_sequence(self, input_seq):
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1,1))
        target_seq[0,0] = preprocessing.word_to_id['<sos>']

        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq]+[e_out, e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = preprocessing.id_to_word[sampled_token_index]

            if(sampled_token != '<eos>'):
                decoded_sentence += sampled_token + ' '

            if (sampled_token == '<eos>' or len(decoded_sentence.split())>=self.max_len-1):
                stop_condition = True

            # update sequence
            target_seq = np.zeros((1,1))
            target_seq[0,0] = sampled_token_index

            # update state
            e_h, e_c = h, c

        return decoded_sentence

    # isload_model parameter must be False always
    def learn(self, hidden_size=300, embedding_dim=300, epoch=10, batch_size=128, isload_model=False):
        # load data
        if os.path.isfile("./qadf_tf.pkl"):
            self.x_train, self.t_train = preprocessing.load_preprocess('./qadf_tf.pkl')
        else:
            self.x_train, self.t_train = preprocessing.load_data(file_name='../dataset/ChatbotData.csv', need_soseos=True, padding_num=0,
                                                       save_file_name='qadf_tf.pkl')

        # divide data
        self.x_test, self.x_train = preprocessing.divide_test_train(self.x_train, test_rate=0.1)
        self.t_test, self.t_train = preprocessing.divide_test_train(self.t_train, test_rate=0.1)

        self.max_len = self.x_train.shape[1]

        model = None

        if os.path.isdir('./my_model') and isload_model:
            model = load_model('./my_model', compile=False)
            print('model loaded...')

        else:
            print('creating model...')
            # encoder
            embedding_dim = embedding_dim
            hidden_size = hidden_size

            encoder_inputs = Input(shape = (self.x_train.shape[1],))

            # encoder embedding
            encoder_embed = Embedding(len(preprocessing.word_to_id), embedding_dim)(encoder_inputs)

            # encoder LSTM
            encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True,
                                 dropout= 0.4)
            encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_embed)

            encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True,
                                 dropout= 0.4)
            encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

            encoder_lstm3 = LSTM(hidden_size, return_sequences=True, return_state=True,
                                 dropout= 0.4)
            encoder_outputs, state_h, state_c = encoder_lstm2(encoder_output2)

            # decoder
            decoder_inputs = Input(shape=(None,))

            # decoder embedding
            decoder_embed_layer = Embedding(len(preprocessing.word_to_id),embedding_dim)
            decoder_embed = decoder_embed_layer(decoder_inputs)

            # decoder LSTM
            decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True,
                                dropout=0.4)
            decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=[state_h, state_c])

            # softmax
            # decoder_softmax = Dense(len(preprocessing.word_to_id), activation='softmax')
            # decoder_softmax_outputs = decoder_softmax(decoder_outputs)

            # add attention
            attention_layer = AttentionLayer(name='attention_layer')
            attention_outputs, attention_states = attention_layer([encoder_outputs, decoder_outputs])

            # connect attention results and hidden states of decoder
            decoder_concat_inputs = Concatenate(axis= -1, name='concat_layer')([decoder_outputs, attention_outputs])

            #softmax
            decoder_softmax = Dense(len(preprocessing.word_to_id), activation='softmax')
            decoder_softmax_outputs = decoder_softmax(decoder_concat_inputs)

            # define model
            model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
            save_model(model, './my_model')
            model.summary()

        if os.path.isfile('mycheckpoint.index'):
            model.load_weights('mycheckpoint')
            print("checkpoint loaded...")

        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
        # fit
        if epoch > 0:
            # set condition for early stopping
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=0.1)

            history = model.fit([self.x_train, self.t_train[:, :-1]], self.t_train.reshape(self.t_train.shape[0], self.t_train.shape[1], 1)[:, 1:],
                            epochs = epoch, callbacks=[], batch_size=128,
                            validation_data=([self.x_test, self.t_test[:, :-1]], self.t_test.reshape(self.t_test.shape[0], self.t_test.shape[1], 1)[:, 1:]))
            model.save_weights('mycheckpoint')

            plt.plot(history.history['loss'], label='train_loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.legend()
            plt.show()

        # make test model
        self.encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
        decoder_state_input_h = Input(shape=(hidden_size,))
        decoder_state_input_c = Input(shape=(hidden_size,))
        decoder_embed2 = decoder_embed_layer(decoder_inputs)
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_embed2, initial_state=[decoder_state_input_h, decoder_state_input_c])

        # attention function
        decoder_hidden_state_input = Input(shape=((self.x_train.shape[1], hidden_size)))
        attention_out_inf, attention_states_inf = attention_layer([decoder_hidden_state_input, decoder_outputs2])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attention_out_inf])

        # decoder output layer
        decoder_outputs2 = decoder_softmax(decoder_inf_concat)

        # decoder model
        self.decoder_model = Model([decoder_inputs]+[decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
                              [decoder_outputs2] + [state_h2, state_c2])

        self.encoder_model.save('test_encoder')
        self.decoder_model.save('test_decoder')

    def seq2text(self, input_seq):
        result = ''
        for i in list(input_seq):
            if i == 0:
                continue
            result += preprocessing.id_to_word[i]
            result += ' '
        return result

    def ask_question(self, input_str):
        m = preprocessing.encode(input_str, size=self.max_len)
        print('M : ', self.decode_sequence(m))

# customized callback for tensorflow
# class predict_examples(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         rn = random.randint(0, len()-5)
#         t_pred = self.model.predict(self.self.x_test[rn:rn+5])
#         print(t_pred)
#         for x,t,p in zip(self.self.x_test[rn:rn+5], self.t_test[rn:rn+5], t_pred):
#             print('---------------')
#             print('question: {}\n'
#                   'answer: {}\n'
#                   'my answer: {}\n'.format(' '.join([preprocessing.id_to_word[n] for n in x]),
#                                            ' '.join([preprocessing.id_to_word[n] for n in t]),
#                                            ' '.join([preprocessing.id_to_word[n] for n in p])))

if __name__ == '__main__':
    astf = att_seq2seq_tf()
    astf.learn(epoch=1)
    print_num = 1
    for i in range(print_num):
        print('------------------------------------------------')
        print('Q ', astf.seq2text(np.ravel(astf.x_test[i:i+1])))
        print('A ', astf.seq2text(np.ravel(astf.t_test[i:i+1])[1:]))
        print('M ', astf.decode_sequence(astf.x_test[i:i+1, :]))
    print('------------------------------------------------')
    print("채팅창에 '종료'를 입력하여 대화를 종료할 수 있습니다.")
    me = ''
    while True:
        me = input('Q : ')
        if me == '종료':
            break
        astf.ask_question(me)