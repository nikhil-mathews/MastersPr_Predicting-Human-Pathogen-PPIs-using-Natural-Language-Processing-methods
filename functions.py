#Usage
# import sys
# sys.path.insert(0,'path to this file')
# import functions as f


import pickle
import pandas as pd
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Attention,Concatenate
from keras.models import Model
from sklearn.metrics import roc_auc_score,roc_curve, auc
from numpy import random
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import seaborn as sns

directory = '/content/drive/MyDrive/ML_Data/'

#Use this to create nD format input.
#For eg, to create 4D input, combine_AC(df,4)
def combine_AC(df,chunksize=3,seperate_chunks=False):
  if not seperate_chunks:
    df.Human = df.Human.apply(lambda x: [''.join(x)[i:i+chunksize] for i in range(0, len(''.join(x))) if len(''.join(x)[i:i+chunksize])>=chunksize])
    df.Yersinia = df.Yersinia.apply(lambda x: [''.join(x)[i:i+chunksize] for i in range(0, len(''.join(x))) if len(''.join(x)[i:i+chunksize])>=chunksize])
    try:
        df.Joined = [df.loc[row]['Human']+df.loc[row]['Yersinia'] for row in range(df.shape[0])]
    except:
        df.Joined = df.Joined.apply(lambda x: [''.join(x)[i:i+chunksize] for i in range(0, len(''.join(x))) if len(''.join(x)[i:i+chunksize])>=chunksize])
    return df
  #print("JHGVBJGHGHKHGKG")
  df.Human = df.Human.apply(lambda x: [''.join(x)[i:i+chunksize] for i in range(0, len(''.join(x)), chunksize)])
  df.Yersinia = df.Yersinia.apply(lambda x: [''.join(x)[i:i+chunksize] for i in range(0, len(''.join(x)), chunksize)])
  df.Joined = [df.loc[row]['Human']+df.loc[row]['Yersinia'] for row in range(df.shape[0])]
  return df

def shuff_together(df1,df2):
    joined = pd.concat([df1,df2], axis=0)
    joined = joined.iloc[np.random.permutation(len(joined))].reset_index(drop=True)
    return joined.iloc[:df1.shape[0],:],joined.iloc[df1.shape[0]:,:].reset_index(drop=True)
def load_data(D=1,randomize=False):
    try:
        with open(directory+'df_train_'+str(D)+'D.pickle', 'rb') as handle:
            df_train = pickle.load(handle)
    except:
        df_train = pd.read_pickle("C:/Users/nik00/py/proj/hyppi-train.pkl")
    try:
        with open(directory+'df_test_'+str(D)+'D.pickle', 'rb') as handle:
            df_test = pickle.load(handle)
    except:
        df_test = pd.read_pickle("C:/Users/nik00/py/proj/hyppi-independent.pkl")
    if randomize:
        return shuff_together(df_train,df_test)
    else:
        return df_train,df_test

#Creates tokenizers and inputs for doubleip configuration
def get_seq_data_doubleip(MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH,df_train,df_test, pad = 'center',show =False, saveTokrs = False):
  print("MAX_VOCAB_SIZE is",MAX_VOCAB_SIZE)
  print("MAX_SEQUENCE_LENGTH is",MAX_SEQUENCE_LENGTH)
  ip_train_Human = df_train[['Human']]
  ip_train_Yersinia = df_train[['Yersinia']]
  sentences_train_Human = pd.DataFrame(' '.join(ip_train_Human.loc[i]['Human']) for i in range(ip_train_Human.shape[0])).values.flatten()
  sentences_train_Yersinia = pd.DataFrame(' '.join(ip_train_Yersinia.loc[i]['Yersinia']) for i in range(ip_train_Yersinia.shape[0])).values.flatten()
  tokenizer1 = Tokenizer(num_words=MAX_VOCAB_SIZE)
  tokenizer1.fit_on_texts(sentences_train_Human)
  tokenizer2 = Tokenizer(num_words=MAX_VOCAB_SIZE)
  tokenizer2.fit_on_texts(sentences_train_Yersinia)
  sequences1_train = tokenizer1.texts_to_sequences(sentences_train_Human)
  sequences2_train = tokenizer2.texts_to_sequences(sentences_train_Yersinia)
  print("max sequences1_train length:", max(len(s) for s in sequences1_train))
  print("min sequences1_train length:", min(len(s) for s in sequences1_train))
  s = sorted(len(s) for s in sequences1_train)
  print("median sequences1_train length:", s[len(s) // 2])
  if show : show_stats(sequences1_train,MAX_SEQUENCE_LENGTH,'Human_train')  
  print("max word index sequences1_train:", max(max(seq) for seq in sequences1_train if len(seq) > 0))
  print("max sequences2_train length:", max(len(s) for s in sequences2_train))
  print("min sequences2_train length:", min(len(s) for s in sequences2_train))
  s = sorted(len(s) for s in sequences2_train)
  print("median sequences2_train length:", s[len(s) // 2])
  if show : show_stats(sequences2_train,MAX_SEQUENCE_LENGTH,'Yersinia_train')
  print("max word index sequences2_train:", max(max(seq) for seq in sequences2_train if len(seq) > 0))
  word2idx = tokenizer1.word_index
  print('Found %s unique tokens in tokenizer1.' % len(word2idx))
  word2idx = tokenizer2.word_index
  print('Found %s unique tokens in tokenizer2.' % len(word2idx))
  if pad is 'center':
      print("Center padding")
      data1 = pad_centered(sequences1_train, MAX_SEQUENCE_LENGTH)
      data2 = pad_centered(sequences2_train, MAX_SEQUENCE_LENGTH)
  else:
      print(pad+" padding")
      data1 = pad_sequences(sequences1_train, MAX_SEQUENCE_LENGTH,padding=pad, truncating=pad)
      data2 = pad_sequences(sequences2_train, MAX_SEQUENCE_LENGTH,padding=pad, truncating=pad)
  print('Shape of data1 tensor:', data1.shape)
  print('Shape of data2 tensor:', data2.shape)

  ip_test_Human = df_test[['Human']]
  ip_test_Yersinia = df_test[['Yersinia']]
  sentences1_test = pd.DataFrame(' '.join(ip_test_Human.loc[i]['Human']) for i in range(ip_test_Human.shape[0])).values.flatten()
  sentences2_test = pd.DataFrame(' '.join(ip_test_Yersinia.loc[i]['Yersinia']) for i in range(ip_test_Yersinia.shape[0])).values.flatten()
  test_sequences1 = tokenizer1.texts_to_sequences(sentences1_test)
  test_sequences2 = tokenizer2.texts_to_sequences(sentences2_test)
  print("max test_sequences1 length:", max(len(s) for s in test_sequences1))
  print("min test_sequences1 length:", min(len(s) for s in test_sequences1))
  s = sorted(len(s) for s in test_sequences1)
  print("median test_sequences1 length:", s[len(s) // 2])
  if show : show_stats(test_sequences1,MAX_SEQUENCE_LENGTH,'Human_test')
  print("max test_sequences2 length:", max(len(s) for s in test_sequences2))
  print("min test_sequences2 length:", min(len(s) for s in test_sequences2))
  s = sorted(len(s) for s in test_sequences2)
  print("median test_sequences2 length:", s[len(s) // 2])
  if show : show_stats(test_sequences2,MAX_SEQUENCE_LENGTH,'Yersinia_test')
  if pad is 'center':
      print("Center padding for test seq.")
      test_data1 = pad_centered(test_sequences1, MAX_SEQUENCE_LENGTH)
      test_data2 = pad_centered(test_sequences2, MAX_SEQUENCE_LENGTH)
  else:
      print(pad+" padding for test seq.")
      test_data1 = pad_sequences(test_sequences1, MAX_SEQUENCE_LENGTH,padding=pad, truncating=pad)
      test_data2 = pad_sequences(test_sequences2, MAX_SEQUENCE_LENGTH,padding=pad, truncating=pad)
  print('Shape of test_data1 tensor:', test_data1.shape)
  print('Shape of test_data2 tensor:', test_data2.shape)

  num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
  print("num_words is",num_words)
  if saveTokrs:
      save((tokenizer1,tokenizer2),'doubleip_tkrs')
      print('Saved tokenizers as doubleip_tkrs')
  return data1,data2,test_data1,test_data2,num_words,MAX_SEQUENCE_LENGTH,MAX_VOCAB_SIZE
  
  
#Creates tokenizers and inputs for join configuration
def get_seq_data_join(MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH,df_train,df_test, pad = 'center',show =False, saveTokrs = False):
  print("MAX_VOCAB_SIZE is",MAX_VOCAB_SIZE)
  print("MAX_SEQUENCE_LENGTH is",MAX_SEQUENCE_LENGTH)
  sentences = pd.DataFrame(' '.join(df_train.loc[i]['Joined']) for i in range(df_train.shape[0])).values.flatten()
  tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
  tokenizer.fit_on_texts(sentences)
  sequences = tokenizer.texts_to_sequences(sentences)
  print("max sequence_data length:", max(len(s) for s in sequences))
  print("min sequence_data length:", min(len(s) for s in sequences))
  s = sorted(len(s) for s in sequences)
  print("median sequence_data length:", s[len(s) // 2])
  if show : show_stats(sequences,MAX_SEQUENCE_LENGTH,'Joined_train')
  print("max word index:", max(max(seq) for seq in sequences if len(seq) > 0))
  word2idx = tokenizer.word_index
  print('Found %s unique tokens.' % len(word2idx))
  
  if pad is 'center':
      print("Center padding.")
      data = pad_centered(sequences, MAX_SEQUENCE_LENGTH)
  else:
      print(pad+" padding.")
      data = pad_sequences(sequences, MAX_SEQUENCE_LENGTH,padding=pad, truncating=pad)
  print('Shape of data tensor:', data.shape)
  sentences_test = pd.DataFrame(' '.join(df_test.loc[i]['Joined']) for i in range(df_test.shape[0])).values.flatten()
  sequences_test = tokenizer.texts_to_sequences(sentences_test)
  print("max sequences_test length:", max(len(s) for s in sequences_test))
  print("min sequences_test length:", min(len(s) for s in sequences_test))
  s = sorted(len(s) for s in sequences_test)
  print("median sequences_test length:", s[len(s) // 2])
  if show : show_stats(sequences_test,MAX_SEQUENCE_LENGTH,'Joined_test')  
  if pad is 'center':
      print("Center padding for test seq.")
      data_test = pad_centered(sequences_test, MAX_SEQUENCE_LENGTH)
  else:
      print(pad+" padding for test seq.")
      data_test = pad_sequences(sequences_test, MAX_SEQUENCE_LENGTH,padding=pad, truncating=pad)
  print('Shape of data_test tensor:', data_test.shape)
  num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
  print("num_words is",num_words)
  if saveTokrs:
      save(tokenizer,'join_tkr')
      print('Saved tokenizer as join_tkr')
  return data,data_test,num_words,MAX_SEQUENCE_LENGTH,MAX_VOCAB_SIZE

def test_functions():
    print ("Access to functions.py verified")
    print ("Access to functions.py verified")

import tensorflow as tf
def pad_centered(l,max_len):
    padded = []
    for item in l:
        #print(item)
        if len(item)<=max_len :
            left_zeros = (max_len - len(item))//2
            right_zeros = (max_len - len(item))//2 + (max_len - len(item))%2
            padded.append([0] * left_zeros + item + [0] * right_zeros)
        else:
            left_idx = (len(item) - max_len)//2 #- (len(item) - max_len)%2
            right_idx = left_idx + max_len
            padded.append(item[left_idx:right_idx])
    assert(np.array(padded).shape == (len(l),max_len))
    return tf.convert_to_tensor(padded)

def embedding_layer(num_words,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM):
  embedding_matrix = random.uniform(-1, 1,(num_words,EMBEDDING_DIM))
  embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=True)
  return embedding_layer
  
import warnings
warnings.filterwarnings("ignore")
def show_stats(sequence,MAX_SEQUENCE_LENGTH,title):
    lengths = [len(l) for l in sequence]
    sss = sorted(lengths)
    median  = sss[len(sss)//2]
    y_pos = np.arange(len(lengths))
    plt.bar(y_pos,lengths)
    plt.plot([0,len(lengths)], [MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH],color='red',linestyle='-',label = "MAX length cutoff")
    plt.plot([0,len(lengths)], [median,median],color='purple',linestyle='--',label = "Median = "+str(median)+"")#, ms=558,label = "Median")
    #plt.figure(figsize=(3, 3))
    plt.title(title+" seq lengths with max length = "+str(sss[-1])+"")
    plt.xlabel("seq[i]")
    plt.ylabel("seq length")
    plt.legend()
    plt.show()
    
def conv_model(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,num_words,DROP=0.2, Flatt = True,filters = 32, kernel_size = 3, MAXpool_size=3):
    inputA = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x1 = Embedding(num_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,trainable=True)(inputA)
    x1 = Conv1D(filters, kernel_size, activation='relu')(x1)
    x1= Dropout(DROP)(x1)
    x1 = MaxPooling1D(MAXpool_size)(x1)
    if Flatt: x1= Flatten()(x1)
    x1 = Dropout(DROP)(x1)
    x1 = Dense(128, activation='relu')(x1)
    return Model(inputs=inputA, outputs=x1)
    
def BiLSTM_model(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,num_words,M,DROP=0.2):
    ip = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(num_words,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)(ip)
    #x = Embedding(num_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,trainable=True)(ip)
    x = Bidirectional(LSTM(M, return_sequences=True))(x)
    x = Dropout(DROP)(x)
    x = Dense(128, activation='relu')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(DROP)(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=ip, outputs=x)
    

# from https://keras.io/api/layers/attention_layers/attention/
def att_model(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,num_words,DROP=0.2, BiLSTM = False):
    
    inputA = Input(shape=(MAX_SEQUENCE_LENGTH,))
    query_embeddings = embedding_layer(num_words,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)(inputA)
    
    inputB = Input(shape=(MAX_SEQUENCE_LENGTH,))
    value_embeddings = embedding_layer(num_words,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)(inputB)
    
    
    cnn_layer = Conv1D(32, 3)
    if BiLSTM: cnn_layer = Bidirectional(LSTM(15, return_sequences=True))
    
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)
    
    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = Attention()(
        [query_seq_encoding, value_seq_encoding])
    
    query_value_attention_seq = Dropout(DROP)(query_value_attention_seq)
    query_value_attention_seq = Dense(128, activation='relu')(query_value_attention_seq)
    
    query_seq_encoding = Dropout(DROP)(query_seq_encoding)
    query_seq_encoding = Dense(128, activation='relu')(query_seq_encoding)
    
    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = GlobalAveragePooling1D()(
        query_value_attention_seq)
    
    query_encoding = Dropout(DROP)(query_encoding)
    query_encoding = Dense(128, activation='relu')(query_encoding)
    
    query_value_attention = Dropout(DROP)(query_value_attention)
    query_value_attention = Dense(128, activation='relu')(query_value_attention)
    
    
    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = Concatenate()([query_encoding, query_value_attention])

    return Model(inputs=[inputA, inputB], outputs=input_layer)    
    # x = Dense(128, activation='relu')(input_layer)
    # x = Dropout(DROP)(x)
    # output = Dense(1, activation="sigmoid",name="Final")(x)
    # return Model(inputs=[inputA, inputB], outputs=output)


# from https://keras.io/examples/nlp/text_classification_with_transformer/
from tensorflow import keras
from tensorflow.keras import layers
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def transf_model(MAX_SEQUENCE_LENGTH,num_words, EMBEDDING_DIM, DROP = 0.3, num_heads = 2, ff_dim = 64):
    inputs=Input((MAX_SEQUENCE_LENGTH,))
    embedding_layer = TokenAndPositionEmbedding(MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(EMBEDDING_DIM, num_heads, ff_dim)
    x = transformer_block(x)
    x = Dropout(DROP)(x)
    x = Dense(256, activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    return Model(inputs,x)
    # ip = transf_model(MAX_SEQUENCE_LENGTH_,num_words_5D_join,5)
    # x = Dropout(DROP)(ip.output)
    # x = Dense(128, activation="relu")(x)
    # x = Dropout(DROP)(x)
    # outputs = Dense(1, activation="sigmoid")(x)
    # model1D_CNN_join=Model(ip.input,outputs)
    

def save(data,name):
  with open(directory+''+name+'.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(name):
  with open(directory+''+name+'.pickle', 'rb') as handle:
    return pickle.load(handle)
    
def create_tokenizers(df_train):
    ip_train_Human = df_train[['Human']]
    ip_train_Yersinia = df_train[['Yersinia']]
    sentences_train_Human = pd.DataFrame(' '.join(ip_train_Human.loc[i]['Human']) for i in range(ip_train_Human.shape[0])).values.flatten()
    sentences_train_Yersinia = pd.DataFrame(' '.join(ip_train_Yersinia.loc[i]['Yersinia']) for i in range(ip_train_Yersinia.shape[0])).values.flatten()
    tokenizer1 = Tokenizer(num_words=500000)
    tokenizer1.fit_on_texts(sentences_train_Human)
    tokenizer2 = Tokenizer(num_words=500000)
    tokenizer2.fit_on_texts(sentences_train_Yersinia)
    save((tokenizer1,tokenizer2),'doubleip_tkrs')
    print('Saved tokenizers as doubleip_tkrs')
    sentences = pd.DataFrame(' '.join(df_train.loc[i]['Joined']) for i in range(df_train.shape[0])).values.flatten()
    tokenizer = Tokenizer(num_words=1000000)
    tokenizer.fit_on_texts(sentences)
    save(tokenizer,'join_tkr')
    print('Saved tokenizer as join_tkr')
    
    
def preprocess(df_test, show =False, saveTokrs = True):
    D = len(df_test[['Human']].iloc[0][0][0])
    if D==1:
        print("Converting to 5D. This will take a few minutes")
        combine_AC(df_test,5)
    elif D!=5:
        print("Data should be in 1D format")
        sys.exit()
    else: pass
    
    if saveTokrs:
        if input("Create tokenizers? Enter y if this is new training data. y/n: ") is 'y': create_tokenizers(df_test)
    
    inputs = []
    MAX_SEQUENCE_LENGTH = 2000 #for joined
    print('Preprocessing...')
    #print("Seq length for joined is",MAX_SEQUENCE_LENGTH)
    tokenizer = load('join_tkr')
    sentences_test_J = pd.DataFrame(' '.join(df_test.loc[i]['Joined']) for i in range(df_test.shape[0])).values.flatten()
    sequences_test = tokenizer.texts_to_sequences(sentences_test_J)
    s = sorted(len(s) for s in sequences_test)
    if show : show_stats(sequences_test,MAX_SEQUENCE_LENGTH,'Joined_seq')
    data_test = pad_sequences(sequences_test, MAX_SEQUENCE_LENGTH,padding='pre', truncating='pre')
    inputs.append(data_test)
    sequences_test = tokenizer.texts_to_sequences(sentences_test_J)
    data_test = pad_centered(sequences_test, MAX_SEQUENCE_LENGTH)
    inputs.append(data_test)
    sequences_test = tokenizer.texts_to_sequences(sentences_test_J)
    data_test = pad_sequences(sequences_test, MAX_SEQUENCE_LENGTH,padding='post', truncating='post')
    inputs.append(data_test)
    MAX_SEQUENCE_LENGTH = 1000 #for doubleip
    #print("Seq length for doubleip is",MAX_SEQUENCE_LENGTH)
    ip_test_Human = df_test[['Human']]
    ip_test_Yersinia = df_test[['Yersinia']]
    sentences1_test = pd.DataFrame(' '.join(ip_test_Human.loc[i]['Human']) for i in range(ip_test_Human.shape[0])).values.flatten()
    sentences2_test = pd.DataFrame(' '.join(ip_test_Yersinia.loc[i]['Yersinia']) for i in range(ip_test_Yersinia.shape[0])).values.flatten()
    tokenizer1,tokenizer2 = load('doubleip_tkrs')
    test_sequences1 = tokenizer1.texts_to_sequences(sentences1_test)
    test_sequences2 = tokenizer2.texts_to_sequences(sentences2_test)
    if show : show_stats(test_sequences1,MAX_SEQUENCE_LENGTH,'doubleip seq')
    test_data1 = pad_sequences(test_sequences1, MAX_SEQUENCE_LENGTH,padding='pre', truncating='pre')
    inputs.append(test_data1)
    test_data2 = pad_sequences(test_sequences2, MAX_SEQUENCE_LENGTH,padding='pre', truncating='pre')
    inputs.append(test_data2)
    test_sequences1 = tokenizer1.texts_to_sequences(sentences1_test)
    test_sequences2 = tokenizer2.texts_to_sequences(sentences2_test)
    test_data1 = pad_centered(test_sequences1, MAX_SEQUENCE_LENGTH)
    inputs.append(test_data1)
    test_data2 = pad_centered(test_sequences2, MAX_SEQUENCE_LENGTH)
    inputs.append(test_data2)
    test_sequences1 = tokenizer1.texts_to_sequences(sentences1_test)
    test_sequences2 = tokenizer2.texts_to_sequences(sentences2_test)
    test_data1 = pad_sequences(test_sequences1, MAX_SEQUENCE_LENGTH,padding='post', truncating='post')
    inputs.append(test_data1)
    test_data2 = pad_sequences(test_sequences2, MAX_SEQUENCE_LENGTH,padding='post', truncating='post')
    inputs.append(test_data2)
    return inputs




