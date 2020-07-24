from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
import nltk
from nltk import word_tokenize
import collections
import pickle
import math
import time
import argparse
import tensorflow as tf
from tensorflow.contrib import rnn
import re
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
import kenlm
model = kenlm.Model('vietnamese_wiki_3-gram.binary')


app = Flask(__name__)
CORS(app)

# nltk.download('punkt')

# Set paths
#train_english_path = "data/train-en-vi/train.en"
#train_vietnamese_path = "data/train-en-vi/train.vi"
word2int_english_path = "vocab_english/word2int.pickle"
int2word_english_path = "vocab_english/int2word.pickle"
word2int_vietnamese_path = "vocab_vietnamese/word2int.pickle"
int2word_vietnamese_path = "vocab_vietnamese/int2word.pickle"

# Load vocab dictionary word2int, int2word
with open(word2int_english_path, 'rb') as fopen:
    word2int_english = pickle.load(fopen, encoding='latin1')

int2word_english = dict(
    zip(word2int_english.values(), word2int_english.keys()))


with open(word2int_vietnamese_path, 'rb') as fopen:
    word2int_vietnamese = pickle.load(fopen, encoding='latin1')

int2word_vietnamese = dict(
    zip(word2int_vietnamese.values(), word2int_vietnamese.keys()))

# Convert input data from text to int


def get_intSeq_english(data_list, max_length, padding=False):
    seq_list = list()
    for sent in data_list:
        # Get tokens in each sent
        words = word_tokenize(sent)

        # Use this for train_english
        if(padding):
            # Make all sent to have the same length as max_length
            if(len(words) < max_length):
                words = words + (max_length-len(words))*["<pad>"]
            else:
                words = words[:max_length]

        # Use this for train_vietnamese
        else:
            words = words[:(max_length-1)]

        # Convert word to its corresponding int value
        # If the word doesnt exist, use the value of "<unk>" by default
        int_seq = [word2int_english.get(
            word, word2int_english["<unk>"]) for word in words]

        # Add int_seq to seq_list
        seq_list.append(int_seq)

    return seq_list

# Convert input data from text to int


def get_intSeq_vietnamese(data_list, max_length, padding=False):
    seq_list = list()
    for sent in data_list:
        # Get tokens in each sent
        words = word_tokenize(sent)

        # Use this for train_english
        if(padding):
            # Make all sent to have the same length as max_length
            if(len(words) < max_length):
                words = words + (max_length-len(words))*["<pad>"]
            else:
                words = words[:max_length]

        # Use this for train_vietnamese
        else:
            words = words[:(max_length-1)]

        # Convert word to its corresponding int value
        # If the word doesnt exist, use the value of "<unk>" by default
        int_seq = [word2int_vietnamese.get(
            word, word2int_vietnamese["<unk>"]) for word in words]

        # Add int_seq to seq_list
        seq_list.append(int_seq)

    return seq_list


# Define the max length of english and vietnamese
english_max_len = 100
vietnamese_max_len = 100

# load model
word_embed_english_w2v = KeyedVectors.load_word2vec_format(
    'word_embedding/model_en.bin', binary=True, unicode_errors='ignore')
# Sort the int2word
int2word_sorted = sorted(int2word_english.items())

# Get the list of word embedding corresponding to int value in ascending order
word_emb_list = list()
embedding_size = len(word_embed_english_w2v['the'])
for int_val, word in int2word_sorted:
    # Add Glove embedding if it exists
    if(word in word_embed_english_w2v):
        word_emb_list.append(word_embed_english_w2v[word])

    # Otherwise, the value of word embedding is 0
    else:
        word_emb_list.append(np.zeros([embedding_size], dtype=np.float32))

# Assign random vector to <s>, </s> token
word_emb_list[2] = np.random.normal(0, 1, embedding_size)
word_emb_list[3] = np.random.normal(0, 1, embedding_size)

# the final word embedding
word_embed_english = np.array(word_emb_list)


# load model
word_embed_vietnamese_w2v = KeyedVectors.load_word2vec_format(
    'word_embedding/model_vn.bin', binary=True, unicode_errors='ignore')

# Sort the int2word
int2word_sorted = sorted(int2word_vietnamese.items())

# Get the list of word embedding corresponding to int value in ascending order
word_emb_list = list()
embedding_size = len(word_embed_vietnamese_w2v['the'])
for int_val, word in int2word_sorted:
    # Add Glove embedding if it exists
    if(word in word_embed_vietnamese_w2v):
        word_emb_list.append(word_embed_vietnamese_w2v[word])

    # Otherwise, the value of word embedding is 0
    else:
        word_emb_list.append(np.zeros([embedding_size], dtype=np.float32))

# Assign random vector to <s>, </s> token
word_emb_list[2] = np.random.normal(0, 1, embedding_size)
word_emb_list[3] = np.random.normal(0, 1, embedding_size)

# the final word embedding
word_embed_vietnamese = np.array(word_emb_list)


def get_batches(input_data, output_data, batch_size):
    # Convert input and output data from list to numpy array
    input_data = np.array(input_data)
    output_data = np.array(output_data)

    # Number of batches per epoch
    num_batches_epoch = math.ceil(len(input_data)/batch_size)
    for batch_num in range(num_batches_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(input_data))
        yield input_data[start_index:end_index], output_data[start_index:end_index]


#CNN + GLU
def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((sentence_size, embedding_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for k in range(1, le):
        for j in range(1, ls):
            encoding[j-1, k-1] = (1.0 - j/float(ls)) - (
                k / float(le)) * (1. - 2. * j/float(ls))

    return encoding


def _create_position_embedding(embedding_dim, num_positions, lengths, maxlen):
    # Create constant position encodings
    position_encodings = tf.constant(
        position_encoding(num_positions, embedding_dim))

    # Slice to size of current sequence
    pe_slice = position_encodings[:maxlen, :]
    # Replicate encodings for each element in the batch
    batch_size = tf.shape(lengths)[0]
    pe_batch = tf.tile([pe_slice], [batch_size, 1, 1])

    # Mask out positions that are padded
    positions_mask = tf.sequence_mask(
        lengths=lengths, maxlen=maxlen, dtype=tf.float32)
    positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)

    return positions_embed

# CNN layer


# padding should take attention
def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,  var_scope_name="conv_layer"):
    with tf.variable_scope("conv_layer_"+str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(
            mean=0, stddev=tf.sqrt(4.0*dropout/(kernel_size*in_dim))), trainable=True)
        # V shape is M*N*k,  V_norm shape is k
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])
        g = tf.get_variable('g', dtype=tf.float32,
                            initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[
                            out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, out_dim])*tf.nn.l2_normalize(V, [0, 1])
        inputs = tf.nn.bias_add(tf.nn.conv1d(
            value=inputs, filters=W, stride=1, padding=padding), b)
        return inputs

# tang kich thuoc layer truoc khi vao cnn dam bao shape ko thay doi, so luong du lieu van duoc giu lai


def linear_mapping_weightnorm(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name):
        input_shape = inputs.get_shape().as_list()    # static shape. may has None
        input_shape_tensor = tf.shape(inputs)
        # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
        V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(
            mean=0, stddev=tf.sqrt(dropout*1.0/int(input_shape[-1]))), trainable=True)
        # V shape is M*N,  V_norm shape is N
        V_norm = tf.norm(V.initialized_value(), axis=0)
        g = tf.get_variable('g', dtype=tf.float32,
                            initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(
        ), trainable=True)   # weightnorm bias is init zero

        assert len(input_shape) == 3
        inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        inputs = tf.matmul(inputs, V)
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
        # inputs = tf.matmul(inputs, V)    # x*v

        scaler = tf.div(g, tf.norm(V, axis=0))   # g/2-norm(v)
        # x*v g/2-norm(v) + b
        inputs = tf.reshape(scaler, [1, out_dim]) * \
            inputs + tf.reshape(b, [1, out_dim])

        return inputs


def position_encoding(inputs):
    T = tf.shape(inputs)[1]
    repr_dim = inputs.get_shape()[-1].value
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(
        tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1])


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape,
                            tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape,
                           tf.float32, tf.zeros_initializer())
    return gamma * normalized + beta


def cnn_block(x, dilation_rate, pad_sz, hidden_dim, kernel_size):
    x = layer_norm(x)
    pad = tf.zeros([tf.shape(x)[0], pad_sz, hidden_dim])
    x = tf.layers.conv1d(inputs=tf.concat([pad, x, pad], 1),
                         filters=hidden_dim,
                         kernel_size=kernel_size,
                         dilation_rate=dilation_rate)
    x = x[:, :-pad_sz, :]
    x = tf.nn.relu(x)
    return x
# GLU


def gated_linear_units(inputs):
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    input_pass = inputs[:, :, 0:int(input_shape[2]/2)]
    input_gate = inputs[:, :, int(input_shape[2]/2):]
    input_gate = tf.sigmoid(input_gate)
    return tf.multiply(input_pass, input_gate)

# Model for Machine Translation


class Seq2SeqModel(object):
    def __init__(self, vocab_size_en, vocab_size_vi, word_embedding_en, word_embedding_vi, input_len, output_len, params, train=True):
        # Get the vocab size
        self.vocab_size_en = vocab_size_en
        self.vocab_size_vi = vocab_size_vi

        # Get hyper-parameters from params
        self.num_layers = params['num_layers']
        self.num_hiddens = params['num_hiddens']
        self.learning_rate = params['learning_rate']
        self.keep_prob = params['keep_prob']
        self.beam_width = params['beam_width']

        self.kernel_size = params['kernel_size']
        # Using BasicLSTMCell as a cell unit
        self.cell = tf.nn.rnn_cell.LSTMCell

        # Define Place holders for the model
        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        # False means not adding the variable to the graph collection
        self.global_step = tf.Variable(0, trainable=False)

        # place holders for encoder
        self.inputSeq = tf.placeholder(tf.int32, [None, input_len])
        # Need to define the Shape as required in tf.contrib.seq2seq.tile_batch
        self.inputSeq_len = tf.placeholder(tf.int32, [None])

        # place holders for decoder
        self.decoder_input = tf.placeholder(tf.int32, [None, output_len])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, output_len])

        # Define projection_layer
        self.projection_layer = tf.layers.Dense(
            self.vocab_size_vi, use_bias=False)

        # Define the Embedding layer
        with tf.name_scope("embedding"):
            self.embeddings_en = tf.get_variable(
                "embeddings_en", initializer=tf.constant(word_embedding_en, dtype=tf.float32))
            self.embeddings_vi = tf.get_variable(
                "embeddings_vi", initializer=tf.constant(word_embedding_vi, dtype=tf.float32))

            # map the int value with its embeddings
            input_emb = tf.nn.embedding_lookup(
                self.embeddings_en, self.inputSeq)
            decoder_input_emb = tf.nn.embedding_lookup(
                self.embeddings_vi, self.decoder_input)

            #layer = 0
            input_emb += position_encoding(input_emb)
            for i in range(self.num_layers):
                next_layer = input_emb
                dilation_rate = 2 ** i
                pad_sz = (self.kernel_size - 1) * dilation_rate
                with tf.variable_scope('block_%d' % i, reuse=tf.AUTO_REUSE):
                    # input_emb += cnn_block(input_emb, dilation_rate,
                    #                              pad_sz, 100, self.kernel_size)
                    #layer += gated_linear_units(input_emb)
                    next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=i, out_dim=100*2,
                                                   kernel_size=self.kernel_size, padding="SAME", dropout=0.9, var_scope_name="conv_layer_"+str(i))
                    next_layer = gated_linear_units(next_layer)
                    input_emb = (next_layer + input_emb) * tf.sqrt(0.5)
            #input_emb = gated_linear_units(input_emb)

            #input_emb = layer
            #print("emb: ",input_emb)
            '''
            print("emb1: ", input_emb)
            #CNN encoder
            input_emb += position_encoding(input_emb)
            
            for i in range(self.num_layers): 
                next_layer = input_emb
                dilation_rate = 2 ** i
                pad_sz = (self.kernel_size - 1) * dilation_rate 
                with tf.variable_scope('block',reuse=tf.AUTO_REUSE):
                next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=i, out_dim=100*2, kernel_size=self.kernel_size, padding="SAME", dropout=0.9, var_scope_name="conv_layer_"+str(i))
                next_layer = gated_linear_units(next_layer)
                input_emb = (next_layer + input_emb) * tf.sqrt(0.5)

            print("ebm2: ", input_emb)
            '''

            # Convert from batch_size*seq_len*embedding to seq_len*batch_size*embedding to feed data with timestep
            # But, we need to set time_major=True during Training
            self.encoder_inputEmb = tf.transpose(input_emb, perm=[1, 0, 2])
            self.decoder_inputEmb = tf.transpose(
                decoder_input_emb, perm=[1, 0, 2])

        # Define the Encoder
        with tf.name_scope("encoder"):
            # Create RNN Cell for forward and backward direction
            fw_cells = list()
            bw_cells = list()
            for i in range(self.num_layers):
                fw_cell = self.cell(self.num_hiddens)
                bw_cell = self.cell(self.num_hiddens)

                # Add Dropout
                fw_cell = rnn.DropoutWrapper(
                    fw_cell, output_keep_prob=self.keep_prob)
                bw_cell = rnn.DropoutWrapper(
                    bw_cell, output_keep_prob=self.keep_prob)

                # Add cell to the list
                fw_cells.append(fw_cell)
                bw_cells.append(bw_cell)

            # Build a multi bi-directional model from fw_cells and bw_cells
            outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=fw_cells, cells_bw=bw_cells, inputs=self.encoder_inputEmb, time_major=True, sequence_length=self.inputSeq_len, dtype=tf.float32)

            # The ouput of Encoder (time major)
            self.encoder_outputs = outputs

            # Use the final state of the last layer as encoder_final_state
            encoder_state_c = tf.concat(
                (encoder_state_fw[-1].c, encoder_state_bw[-1].c), 1)
            encoder_state_h = tf.concat(
                (encoder_state_fw[-1].h, encoder_state_bw[-1].h), 1)
            self.encoder_final_state = rnn.LSTMStateTuple(
                c=encoder_state_c, h=encoder_state_h)

        # Define the Decoder for training
        with tf.name_scope("decoder"):
            # Define Decoder cell
            decoder_num_hiddens = self.num_hiddens * 2  # As we use bi-directional RNN
            decoder_cell = self.cell(decoder_num_hiddens)

            # Training mode
            if(train):
                # Convert from time major to batch major
                attention_states = tf.transpose(
                    self.encoder_outputs, [1, 0, 2])

                # Decoder with attention
                attention = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=decoder_num_hiddens, memory=attention_states, memory_sequence_length=self.inputSeq_len, normalize=True)
                attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell, attention_mechanism=attention, attention_layer_size=decoder_num_hiddens)

                # Use the final state of encoder as the initial state of the decoder
                decoder_initial_state = attention_decoder_cell.zero_state(
                    dtype=tf.float32, batch_size=self.batch_size)
                decoder_initial_state = decoder_initial_state.clone(
                    cell_state=self.encoder_final_state)

                # Use TrainingHelper to train the Model
                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.decoder_inputEmb, sequence_length=self.decoder_len, time_major=True)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=attention_decoder_cell, helper=training_helper, initial_state=decoder_initial_state, output_layer=self.projection_layer)
                logits, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True, maximum_iterations=output_len)

                # Convert from time major to batch major
                self.training_logits = tf.transpose(
                    logits.rnn_output, perm=[1, 0, 2])

                # Adding zero to make sure training_logits has shape: [batch_size, sequence_length, num_decoder_symbols]
                self.training_logits = tf.concat([self.training_logits, tf.zeros(
                    [self.batch_size, output_len - tf.shape(self.training_logits)[1], self.vocab_size_vi])], axis=1)

            # Inference mode
            else:
                # Using Beam search
                tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(tf.transpose(
                    self.encoder_outputs, perm=[1, 0, 2]), multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                    self.encoder_final_state, multiplier=self.beam_width)
                tiled_inputSeq_len = tf.contrib.seq2seq.tile_batch(
                    self.inputSeq_len, multiplier=self.beam_width)

                # Decoder with attention with Beam search
                attention = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=decoder_num_hiddens, memory=tiled_encoder_outputs, memory_sequence_length=tiled_inputSeq_len, normalize=True)
                attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell, attention_mechanism=attention, attention_layer_size=decoder_num_hiddens)

                # Use the final state of encoder as the initial state of the decoder
                decoder_initial_state = attention_decoder_cell.zero_state(
                    dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
                decoder_initial_state = decoder_initial_state.clone(
                    cell_state=tiled_encoder_final_state)

                # Build a Decoder with Beam Search
                beamSearch_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=attention_decoder_cell,
                    embedding=self.embeddings_vi,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.projection_layer
                )

                # Perform dynamic decoding with beamSearch_decoder
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=beamSearch_decoder, maximum_iterations=output_len, output_time_major=True)

                # Convert from seq_len*batch_size*beam_width to batch_size*beam_width*seq_len
                outputs = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

                # Take the first beam (best result) as Decoder ouput
                # self.decoder_outputs=outputs[:,0,:]
                self.decoder_outputs = outputs

        with tf.name_scope("optimization"):
            # Used for Training mode only
            if(train):
                # Caculate loss value
                masks = tf.sequence_mask(
                    lengths=self.decoder_len, maxlen=output_len, dtype=tf.float32)
                self.loss = tf.contrib.seq2seq.sequence_loss(
                    logits=self.training_logits, targets=self.decoder_target, weights=masks)

                # Using AdamOptimizer
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                # Compute gradient
                gradients = optimizer.compute_gradients(self.loss)
                # Apply Gradient Clipping
                gradients_clipping = [(tf.clip_by_value(
                    grad, clip_value_min=-1., clip_value_max=1.), var) for grad, var in gradients if grad is not None]

                # Apply gradients to variables
                self.train_update = optimizer.apply_gradients(
                    gradients_clipping, global_step=self.global_step)
                #self.train_update = optimizer.apply_gradients(gradients, global_step=self.global_step)


# define hyparamater
params = dict()
params['num_layers'] = 2
params['num_hiddens'] = 512
params['learning_rate'] = 0.001
params['keep_prob'] = 0.85
params['beam_width'] = 10
params['kernel_size'] = 3
# Path of the saved model
checkpoint = "NMT.ckpt"

# Reset the default graph
# tf.reset_default_graph()
# tf.compat.v1.reset_default_graph
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()


def pre(w):
    w = w.lower()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    return w


with sess.as_default():
    with graph.as_default():
        # Load saved model
        # Use Seq2SeqModel to create the same graph as saved model
        loaded_model = Seq2SeqModel(len(int2word_english), len(int2word_vietnamese), word_embed_english,
                                    word_embed_vietnamese, english_max_len, vietnamese_max_len, params, train=False)

    # Load the value of variables in saved model
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, checkpoint)


def getSeq(dic):
    if(len(dic) == 0):
        return
    index = 0
    maxL = model.score(dic[0])
    for i in range(1, len(dic)):
        temp = model.score(dic[i])
        if maxL < temp:
            maxL = temp
            index = i
    return dic[index]


def predict(inputSeq):
    #inputSeq = []
    #inputSeq.append('i go to school')

    # Get the sequence of int value
    input_intSeq = get_intSeq_english(inputSeq, english_max_len, padding=True)

    with sess.as_default():
        with graph.as_default():
            # Load saved model
            # Use Seq2SeqModel to create the same graph as saved model
            # loaded_model = Seq2SeqModel(len(int2word_english), len(int2word_vietnamese), word_embed_english,
            # word_embed_vietnamese, english_max_len, vietnamese_max_len, params, train=False)

            # Load the value of variables in saved model
            #saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            #saver.restore(sess, checkpoint)

            # convert
            input_data = np.array(input_intSeq)

            # The actual length of each sequence in the batch (excluding "<pad>")
            real_len = list(map(lambda seq: len(
                [word_int for word_int in seq if word_int != 0]), input_data))

            # Create a feed_dict for predict data
            valid_feed_dict = {
                loaded_model.batch_size: len(input_data),
                loaded_model.inputSeq: input_data,
                loaded_model.inputSeq_len: real_len,
            }
            # print(len(input_data))
            # print(input_data)
            # print(real_len)

            # Get the decoder output by Inference
            decoder_outputs = sess.run(
                loaded_model.decoder_outputs, feed_dict=valid_feed_dict)

            # Loop through each seq in decoder_outputs
            listOut = []
            for out10_seq in decoder_outputs:
                listSeq = []
                for out_seq in out10_seq:
                    out_sent = list()
                    for word_int in out_seq:
                        word = int2word_vietnamese[word_int]
                        if word == "</s>":
                            break
                        else:
                            out_sent.append(word)
                    listSeq.append(" ".join(out_sent))
                listOut.append(listSeq)

            listOutput = []
            print(listOut)
            print('listOut*****************************')
            # Using model language_moel
            for listSeq in listOut:
                listOutput.append(getSeq(listSeq))
    print(listOutput)
    return listOutput


@app.route('/')
def show_predict_stock_form():
    # return render_template('predictorform.html')
    # return render_template('try.html')
    return "API Model Dich May Tu Tieng Anh Sang Tieng Viet"


@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        if request.json.get('input') == '':
            return jsonify({'message': 'input is null'}), 400
        else:
            input = pre((request.json.get('input')).strip())
            # with sess.as_default():
            # with graph.as_default():

            inputArray = nltk.sent_tokenize(
                ' '.join(nltk.word_tokenize(input)).lower())
            outputArray = predict(inputArray)
            print(outputArray)
            output = str(" ").join(outputArray)

            # print(output)
        return jsonify({'output': output}), 200


#app.run("localhost", "9999", debug=True)
if __name__ == '__main__':
    # app.debug=True
    app.run()
