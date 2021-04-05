# !pip install progressbar
# !pip install contractions

# !pip install spacy
# !pip install gensim 
# !pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html
# !pip install fastBPE regex requests sacremoses subword_nmt
# !pip install -q hydra-core
# !pip install tqdm

# !pip install textattack
# # !pip install GPUtil

# !python -m spacy download en_core_web_md
# !python -m spacy download en_core_web_lg
# !python -m textblob.download_corpora
# !wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
# !pip install sentencepiece

import warnings
warnings.filterwarnings('ignore')

import itertools
import sklearn
import io
import tqdm
from progressbar import ProgressBar

import pandas as pd
import numpy as np
from random import randint
import os
import re
import pandas as pd

import contractions
import unicodedata
# from wordcloud import wordcloud
# from textblob import TextBlob
import joblib

# Visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Deep Learning 
import tensorflow as tf
import keras
from keras.models import Model,Sequential

from keras.layers import Activation,Flatten,Dense,Embedding,Input,GlobalAveragePooling1D,GlobalMaxPool1D,Conv1D,MaxPooling1D,concatenate,SpatialDropout1D
from keras.layers import LSTM,GRU,Bidirectional, Dropout,Permute,multiply,Layer
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import keras.backend as K

import torch
import tensorflow_hub as hub
import tokenization
from transformers import BertTokenizer
from transformers import TFBertModel,AutoTokenizer,BertForMaskedLM

# Used to standardise the date and undertake PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# Scikit-learn imports for data encoding and partitionning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD as TSVD

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV,learning_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import (confusion_matrix,roc_auc_score, precision_recall_curve, auc,
                             roc_curve, recall_score,accuracy_score, classification_report, f1_score,
                             precision_recall_fscore_support, log_loss)

# Spacy imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

# Gensim Imports
import gensim.downloader
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# NLTK 
from nltk import word_tokenize
from nltk.corpus import stopwords

from textattack.augmentation import WordNetAugmenter,EmbeddingAugmenter

def load_sub_data():
    df_submission = pd.read_csv(os.path.abspath('Dataset/test.tsv'),sep="\t")
    df_submission = df_submission.rename(columns={"Phrase": "review"}, errors="raise")
    df_submission = df_submission.drop(["SentenceId"], axis=1)
    return df_submission
    
def load_train_data():
    df_train = pd.read_csv(os.path.abspath('Dataset/train.tsv'),sep="\t")
    df_train = df_train.rename(columns={"Phrase": "review", "Sentiment": "sentiment"}, errors="raise")
    df_train = df_train.drop(["PhraseId","SentenceId"], axis=1)
    return df_train

def plot_class_distribution(field):
    print(field.value_counts())
    sns.countplot(x=field).set_title('Visualisation of the dataset class distribution')
    plt.show()
    
def data_cleaning(df):
    df["review"] = clean_sentences(df["review"],remove_stop_words=False)
    df = extract_most_uncommon_words(df)
    return df

def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x

def clean_sentences(review,remove_stop_words=False):
    #Lower Case Conversion 
    review = review.apply(lambda x: str(x).lower())
    # Contraction to Expansion 
    review = review.apply(lambda x: contractions.fix(str(x)))
    # Remove Emails and URL's from strings
    review = review.apply(lambda x: re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x))
    review = review.apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x))
    # Remove HTML tags
    review = review.apply(lambda x: re.sub('<.*?>',"",x))
    # Remove trailing whitespace
    review = review.apply(lambda x: ' '.join(x.split()))
    # Remove special characters and punctuation
    review = review.apply(lambda x: re.sub(r'[^\w ]+', "", x))
    # Remove accented characters
    review = review.apply(lambda x: remove_accented_chars(x))
    if remove_stop_words is True:
        # Remove Stop Words 
        review = review.apply(lambda x: ' '.join([t for t in x.split() if t not in stopwords]))
    return review

def extract_most_uncommon_words(df):
    text = ' '.join(df["review"])
    text = text.split()
    freq_comm = pd.Series(text).value_counts()
    uncommon_words= freq_comm[freq_comm == 1]
    df['review'] = df['review'].apply(lambda x: ' '.join([t for t in x.split() if t not in uncommon_words]))
    return df

def apply_TSVD(X_train,X_test,plot,components):
    tsvd = TSVD(n_components=components, random_state=0)
    print('Initial train matrix shape is: ', X_train.shape)
    print('Initial test shape is: ', X_test.shape)
    X_train = tsvd.fit_transform(X_train)
    X_test = tsvd.transform(X_test)
    print('PCA transformed train shape is: ', X_train.shape)
    print('PCA transformed test shape is: ', X_test.shape)
    if plot is True:
        plt.plot(np.cumsum(tsvd.explained_variance_ratio_))
        plt.show()
    return X_train,X_test,tsvd

def back_translation_augmentation(X_train,y_train):
    class0_rev=[]
    class0_sen=[]
    
    class1_rev=[]
    class1_sen=[]
    
    class3_rev=[]
    class3_sen=[]
    
    class4_rev=[]
    class4_sen=[]
    
    augmenter = WordNetAugmenter()
    
    data=pd.DataFrame(X_train)
    data['sentiment']= y_train
    data = data.reset_index(drop=True)    
    df_class_0_threshold = int(len(data[data["sentiment"]==0]))
    df_class_1_threshold = int(len(data[data["sentiment"]==1])/2)
    df_class_2_threshold = int(len(data[data["sentiment"]==2]))
    df_class_3_threshold = int(len(data[data["sentiment"]==3])/2)
    df_class_4_threshold = int(len(data[data["sentiment"]==4]))


    data["review"] = clean_sentences(data["review"],remove_stop_words=False)
    
    df_class_0 = data[data["sentiment"]==0].reset_index(drop=True)    
    df_class_1 = data[data["sentiment"]==1].reset_index(drop=True)
    df_class_2 = data[data["sentiment"]==2].reset_index(drop=True)
    df_class_3 = data[data["sentiment"]==3].reset_index(drop=True)
    df_class_4 = data[data["sentiment"]==4].reset_index(drop=True)
        
    backTranslation = True

    try:
        # torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]
        # Round-trip translations between English and German:
        en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
        en2de.cuda()

        de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
        de2en.cuda()
    except:
        backTranslation = False
        
    for i in range(df_class_0_threshold):
        paraphrase = df_class_0['review'].loc[i]
        if backTranslation is True:
            paraphrase = de2en.translate(en2de.translate(paraphrase))
        paraphrase= augmenter.augment(paraphrase)
        class0_rev.append(paraphrase[0])
        class0_sen.append(df_class_0['sentiment'].loc[i])
        print(f"Running {i}th sentence out of {df_class_0_threshold} for class 0")  
    for i in range(df_class_1_threshold):
        paraphrase = df_class_1['review'].loc[i]
        if backTranslation is True:
            paraphrase = de2en.translate(en2de.translate(paraphrase))
        paraphrase= augmenter.augment(paraphrase)
        class1_rev.append(paraphrase[0])
        class1_sen.append(df_class_1['sentiment'].loc[i])        
        print(f"Running {i}th sentence out of {df_class_1_threshold} for class 1")  
    for i in range(df_class_3_threshold):
        paraphrase = df_class_3['review'].loc[i]
        if backTranslation is True:
            paraphrase = de2en.translate(en2de.translate(paraphrase))
        paraphrase= augmenter.augment(paraphrase)
        class3_rev.append(paraphrase[0])
        class3_sen.append(df_class_3['sentiment'].loc[i])        
        print(f"Running {i}th sentence out of {df_class_3_threshold} for class 3" )  
    for i in range(df_class_4_threshold):
        paraphrase = df_class_4['review'].loc[i]
        if backTranslation is True:
            paraphrase = de2en.translate(en2de.translate(paraphrase))
        paraphrase= augmenter.augment(paraphrase)
        class4_rev.append(paraphrase[0])
        class4_sen.append(df_class_4['sentiment'].loc[i])        
        print(f"Running {i}th sentence out of {df_class_4_threshold} for class 4")  

        class0_aug=pd.DataFrame({'review':class0_rev, 'sentiment':class0_sen})
        class1_aug=pd.DataFrame({'review':class1_rev, 'sentiment':class1_sen})
        class3_aug=pd.DataFrame({'review':class3_rev, 'sentiment':class3_sen})
        class4_aug=pd.DataFrame({'review':class4_rev, 'sentiment':class4_sen})
        
    return class0_aug,class1_aug,class3_aug,class4_aug

def Tfidf_transformation(df,ngram_range):
    X_vect= TfidfVectorizer(stop_words='english',ngram_range=ngram_range)
    X= X_vect.fit_transform(df["review"])
    print("Before transformation: The data had shape: ", df["review"].shape)
    print("After transformation: The data has shape: ",  X.toarray().shape)
    return X

def return_best_tfidf(df):
    pipe = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LogisticRegression(max_iter = 10000))])
    hyperparameters = {'tfidf__ngram_range': ((1,1), (1,2), (1,3),(1,4))}
    clf = GridSearchCV(pipe, hyperparameters, n_jobs=-1, verbose=2)
    clf.fit(df["review"], df["sentiment"])
    print("The best ngram range parameters according to GridSearchCV: ",clf.best_params_)
    return clf.best_params_

def save_to_pkl(filename,variable):
    file=str(filename)  
    joblib.dump(variable, file)
    
def load_pkl(path):
    file = joblib.load(path)
    return file

def word2Vec_transform(df,save_pkl):
    temp=[]
    pbar = ProgressBar()
    nlp = spacy.load('en_core_web_md')
    for i in pbar(df["review"]):
        temp.append(nlp(i).vector)
    if save_pkl is True:
        save_to_pkl("X_word2Vec.pkl",np.asarray(temp))
    return np.asarray(temp)

def load_pretrained_embedding_model(df_train,df_submission,X_train,X_val,X_test,EMBEDDING_DIM, model):
    # Instantiate the tokenizer object
    tokenizer_obj = Tokenizer()
    total_reviews = pd.concat([df_train.review,df_submission.review])
    tokenizer_obj.fit_on_texts(total_reviews)

    # Pad sequences
    MAX_SEQUENCE_LENGTH = max([len(s.split()) for s in total_reviews])
    print("Maximum sentence lenght: ", MAX_SEQUENCE_LENGTH)

    # define the vocabulary size and number of unique tokens in the corpus
    MAX_VOCAB_SIZE = len(tokenizer_obj.word_index)+1
    word2idx = tokenizer_obj.word_index
    print('Found %s unique tokens in the dataset.' % len(word2idx))

    # prepare text samples and their labels
    X_train_sentences = X_train
    X_train_sequences = tokenizer_obj.texts_to_sequences(X_train_sentences)
    
    # prepare validation samples and their labels
    X_val_sentences = X_val
    X_val_sequences = tokenizer_obj.texts_to_sequences(X_val_sentences)
    
    # prepare validation samples and their labels
    X_test_sentences = X_test
    X_test_sequences = tokenizer_obj.texts_to_sequences(X_test_sentences)
    
    # prepare Kaggle unseen data and its labels    
    X_submission_sentences = df_submission["review"].values
    X_submission_sequences = tokenizer_obj.texts_to_sequences(X_submission_sentences)    

    # Load pretrained embedding model  
    try: 
        print('Loading word vectors from local file...')
        w2v_model = KeyedVectors.load("word2vec-google-news-300.wordvectors", mmap='r')
    except FileNotFoundError:
        print('Loading word vectors from server...')
        w2v_model = gensim.downloader.load(model)
        word_vectors = w2v_model
        word_vectors.save(str(model)+".wordvectors")

    # pad sequences so that we get a N x T matrix
    X_train = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of X_train data tensor:', X_train.shape)
    
    # pad sequences so that we get a N x T matrix
    X_val = pad_sequences(X_val_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of X_val data tensor:', X_val.shape)
    
    # pad sequences so that we get a N x T matrix
    X_test = pad_sequences(X_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of X_test data tensor:', X_test.shape)
    
    # pad sequences so that we get a N x T matrix
    X_submission = pad_sequences(X_submission_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of X_submission data tensor:', X_submission.shape)

    # prepare embedding matrix
    print('Filling pre-trained embeddings...')
    embedding_matrix = np.zeros((MAX_VOCAB_SIZE, EMBEDDING_DIM))
    for word, i in tokenizer_obj.word_index.items():
        if word in w2v_model:
            embedding_matrix[i] = w2v_model[word]
    print("The final embedding matrix has shape: ", embedding_matrix.shape)
    
    return EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,MAX_VOCAB_SIZE,w2v_model,embedding_matrix,X_train,X_val,X_test,X_submission

def train_Word2Vec_model(EMBEDDING_DIM = 100):
    # Instantiate the tokenizer object
    tokenizer_obj = Tokenizer()
    total_reviews = df.review
    tokenizer_obj.fit_on_texts(total_reviews)

    # Pad sequences
    MAX_SEQUENCE_LENGTH = max([len(s.split()) for s in total_reviews])
    print("Maximum sentence lenght: ", MAX_SEQUENCE_LENGTH)

    # define the vocabulary size and number of unique tokens in the corpus
    MAX_VOCAB_SIZE = len(tokenizer_obj.word_index)+1
    word2idx = tokenizer_obj.word_index
    print('Found %s unique tokens in the dataset.' % len(word2idx))

    # prepare text samples and their labels
    sentences = df["review"].fillna("DUMMY_VALUE").values
    sequences = tokenizer_obj.texts_to_sequences(sentences)

    w2v_model = Word2Vec(min_count=10,
                         window=2,
                         size=EMBEDDING_DIM,
                         sample=6e-5, 
                         alpha=0.03, 
                         min_alpha=0.007, 
                         negative=20)

    # Train Word2vec on the data corpus
    print('Training word vectors...')
    documents = [_text.split() for _text in df.review] 
    w2v_model.build_vocab(documents)
    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    print("Vocab size", vocab_size)
    w2v_model.train(documents, total_examples=w2v_model.corpus_count, epochs=30)
    
     # pad sequences so that we get a N x T matrix
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)

    # prepare embedding matrix
    print('Filling pre-trained embeddings...')
    embedding_matrix = np.zeros((MAX_VOCAB_SIZE, EMBEDDING_DIM))
    for word, i in tokenizer_obj.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    print("The final embedding matrix has shape: ", embedding_matrix.shape)
    
    return EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,MAX_VOCAB_SIZE, data,w2v_model,embedding_matrix

def bert_encode(texts, tokenizer, max_len):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def data_partitioning(X,y,test_size, summary):
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    if summary is True:
        print("Overall class distribution in this dataset")
        print(pd.Series(y_train).value_counts())
        print(pd.Series(y_val).value_counts())
        print("")
        print("X_train has shape:", X_train.shape)
        print("y_train has shape:", X_train.shape)
        print("X_val has shape:", X_val.shape)
        print("y_val has shape:", y_val.shape)
        print("X_test has shape:", X_test.shape)
        print("y_test has shape:", y_test.shape)
    return X_train,X_val,X_test,y_train,y_val,y_test

def train_bert_model(model,X_train, y_train, X_val, y_val, BATCH_SIZE, EPOCHS, plot, callback,model_name):
    # load pre-trained word embeddings into an Embedding layer
    # the trainable parameter is set to False so as to keep the embeddings fixed
    if callback is True:
        logdir = os.path.join("logs/"+str(model_name))        
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint('bert_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

        history = model.fit(X_train,
                  to_categorical(y_train),
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(X_val,to_categorical(y_val)),
                  callbacks=[checkpoint,earlystopping])
        
    else:
        history = model.fit(X_train,
                  to_categorical(y_train),
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(X_val,to_categorical(y_val)))

    if plot is True:
        fig,(ax) = plt.subplots(1, 2, figsize=(13,4))
        ax[0].set_title("Training and Validation Accuracy")
        ax[0].plot(history.history['loss'], label='loss')
        ax[0].plot(history.history['val_loss'], label='val_loss')
        ax[0].legend(loc='upper right')

        ax[1].set_title("Training and Validation Accuracy")
        ax[1].plot(history.history['accuracy'], label='acc')
        ax[1].plot(history.history['val_accuracy'], label='val_acc')
        ax[1].legend(loc='lower right')
        plt.show()
    return model
        
def plot_confusion_matrix_test(y_test,y_pred, multiclass):
    data = confusion_matrix(y_test, y_pred)
    if multiclass == 2:
        labels = ["Negative","Positive"]
    if multiclass == 3:
        labels = ["Negative","Neutral","Positive"]
    if multiclass == 5:
        labels = ["Very Negative","Negative","Neutral","Positive", "Very Positive"]
    df_cm = pd.DataFrame(data, columns=labels, index = labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (6,4))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, fmt='g')# font size
    plt.show()
    
def NN_predict(model,X_test, multiclass):
    if multiclass == 2:
        labels = [0, 1]
    if multiclass == 3:
        labels = [0, 1, 2]
    if multiclass == 5:
        labels = [0, 1, 2, 3, 4]
    y_pred = model.predict(X_test)
    score = tf.nn.softmax(y_pred[0])
    temp = []
    for i in y_pred:
        score = tf.nn.softmax(i)
        i = labels[np.argmax(score)]
        temp.append(i)
    return temp

def train_test(model,X_train,y_train,X_test,y_test):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Fitting accuracy'+"\n"+ '**************************')
    train_acc = model.score(X_train,y_train)
    print(train_acc)
    print('Prediction accuracy'+"\n"+'**************************')
    test_acc = model.score(X_test,y_test)
    print(test_acc)
    print("")
    print("************************************************************")
    print("                 LR classification report")
    print("************************************************************")
    print(classification_report(y_test, y_pred))
    return y_pred,train_acc,test_acc, model

def LR_predict(X_train_word2Vec,y_train_word2Vec,X_test_word2Vec,y_test_word2Vec):
    LR =  LogisticRegression(max_iter = 5000)
    plot_learning_curve (LR,"Learning curve for LR",X_train_word2Vec,y_train_word2Vec)
    y_pred_LR,train_acc_LR,test_acc_LR, LR = train_test(LR,X_train_word2Vec,y_train_word2Vec,X_test_word2Vec,y_test_word2Vec)
    plot_confusion_matrix_test(y_test_word2Vec,y_pred_LR,multiclass=5)

def generate_pred(model,X_test,y_test):
    y_pred = NN_predict(model,X_test, multiclass=5)
    plot_confusion_matrix_test(y_test,y_pred,multiclass=5)
    print(classification_report(y_test, y_pred))

def generate_submission(model,X_test,df_submission,model_name, save):
    y_pred_sub = NN_predict(model,X_test, multiclass=5)
    submission = df_submission.copy()
    submission = submission.drop(["review"], axis=1)
    submission["Sentiment"] = y_pred_sub
    if save is True:
        submission.to_csv('/kaggle/working/'+str(model_name)+".csv", index=False)
    return submission

def plot_learning_curve(estimator, title, X, y):
    train_sizes=np.linspace(.1, 1.0, 5)
    cv=3
    fig, ax = plt.subplots(1,1)
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax.legend(loc="best")
    return plt