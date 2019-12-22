# import libraries
import fastai
from fastai import *
from fastai.text import *
import pandas as pd
import numpy as np
from functools import partial
import io
import os
import pandas as pd
import nltk

# Read the test data on which you want to test your model
dataset = pd.read_csv('./files/datasets/brunet2014.csv')
df1 = pd.DataFrame({'label':dataset.iloc[:,1], 'text':dataset.iloc[:,0]})
df1.shape

df1['label'].value_counts()

# Load your saved language model encoder
data_lm = load_data('Path to your saved language model encoder', bs=32) # Batch size is 32

# Create text data bunch for classifier model data
data_clas = TextClasDataBunch.from_df(path = "", train_df = df1, valid_df = df1, vocab=data_lm.train_ds.vocab, bs=32)

# Create text classifier learner object
learn_test = text_classifier_learner(data_clas, drop_mult=0.5, arch = AWD_LSTM)

# Load your saved model
learn_test.load('Path to your saved model')

preds,y,losses = learn_test.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn_test, preds, y, losses)
interp.plot_confusion_matrix()

TN = interp.confusion_matrix()[0][0]
FP = interp.confusion_matrix()[0][1]
FN = interp.confusion_matrix()[1][0]
TP = interp.confusion_matrix()[1][1]

recall = TP/(TP+FN)
print('Recall = ' + str(recall))
precision = TP/(TP+FP)
print('Precision = ' + str(precision))
TPR = TP / (TP+FN)
TNR = TN / (TN + FP)
bal_acc = (TPR + TNR) / 2
print('Balanced Accuracy = ' + str(bal_acc))
