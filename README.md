# chatbot
for hangul daily conversation

### working env
python 3.7.7

tensorflow 2.3.0

c++ 2015-2019 redistribute 14.27.29016

CUDA 10.1

cuDNN 7.6.5


### Note

* create directory named 'dataset' and download dataset link below in 'dataset' directory
    ##### dataset https://github.com/songys/Chatbot_data
* train attention seq2seq model with data in mytensorflow/attention_seq2seq_tf.py
* start mytensorflow/qt.py to test trained chatbot with gui
* while chatting, if you want to have a dialog fixed, you can select the text, write the best answer you think and press save.
 you can create your own dataset named 'myChatbotData.csv' in dataset directory.
* FYI, in 'raw' folder, they are no-tensorflow version. (no GPU work)
* feel free to play with it and make the better chatbot with your own dataset.