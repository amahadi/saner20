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

# Read the files in the dataframe
dataset1 = pd.read_csv('./files/datasets/stackoverflow.csv')
dataset2 = pd.read_csv('./files/datasets/brunet2014.csv')
df1 = pd.DataFrame({'label':dataset1.iloc[:,1], 'text':dataset1.iloc[:,0]})
df2 = pd.DataFrame({'label':dataset2.iloc[:,1], 'text':dataset2.iloc[:,0]})

df = pd.concat([df1, df2])

df.shape

def stratification(data, n_splits):
    stratified_data = pd.DataFrame()
    positive_data = data[data['label'] == 1]
    negative_data = data[data['label'] == 0]
    p_data_size = positive_data.shape[0]
    n_data_size = negative_data.shape[0]
    positive_chunk = ([p_data_size // n_splits + (1 if x < p_data_size % n_splits else 0) for x in range(n_splits)])
    negative_chunk = ([n_data_size // n_splits + (1 if x < n_data_size % n_splits else 0) for x in range(n_splits)])
    p_index, n_index = 0, 0
    for i in range(n_splits):
        tmp_data = positive_data[p_index:p_index+positive_chunk[i]]
        tmp_data = tmp_data.append(negative_data[n_index:n_index+negative_chunk[i]], ignore_index=True)
        tmp_data = tmp_data.sample(frac=1).reset_index(drop=True)
        stratified_data = stratified_data.append(tmp_data, ignore_index=True)
        p_index += positive_chunk[i]
        n_index += negative_chunk[i]
    return stratified_data

stratified_df = stratification(df, 10)

# Specifying the categorical values in the dataset
stratified_df = stratified_df[stratified_df['label'].isin([0,1])]
stratified_df = stratified_df.reset_index(drop = True)

# Print count of values under each category
stratified_df['label'].value_counts()

# Clean all the text to not contain any value except the alphabets and spaces
stratified_df['text'] = stratified_df['text'].str.replace("[^a-zA-Z]", " ")

# Get rid of stopwords

nltk.download('stopwords')

STOPSET_WORDS = ['might', 'may', 'would', 'must', 'lgtm', 'could', 'can', 'good', 'great', 'nice', 'well', \
                 'better', 'worse', 'worst', 'should', 'i', "i'll", "ill", "it's", "its", "im", "i'm", \
                 "they're", "theyre", "you're", "youre", "that's", 'btw', "thats", "theres", "shouldnt", \
                 "shouldn't", "didn't", "didnt", "dont", "don't", "doesn't", "doesnt", "wasnt", "wasn't", \
                 'sense', "mon", 'monday', 'tue', 'wed', 'wednesday', 'thursday', 'lgtm', 'pinging', 'thu', 'friday', 'fri', \
                 'sat', 'saturday', 'sun', 'sunday', 'jan', 'january', 'feb', 'february', 'mar', 'march', \
                 'apr', 'april', 'may', 'jun', 'june', 'july', 'jul', 'aug', 'august', 'sep', 'september', \
                 'oct', 'october', 'nov', 'novenber', 'dec', 'december', 'pm', 'am', '//'
]

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

for word in STOPSET_WORDS:
        stop_words.add(word)

# tokenization 
tokenized_doc = stratified_df['text'].apply(lambda x: x.split())

# remove stop-words 
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization 
detokenized_doc = [] 
for i in range(len(stratified_df)): 
    t = ' '.join(tokenized_doc[i]) 
    detokenized_doc.append(t) 

stratified_df['text'] = detokenized_doc

# split data into training and validation set with stratification
from sklearn.model_selection import train_test_split
df_trn, df_val = train_test_split(stratified_df, stratify = stratified_df['label'], test_size = 0.1, random_state = 12)

df_val, df_test = train_test_split(df_val, stratify = df_val['label'], test_size = 0.5, random_state = 12)

# Prepare data for language model and classification model

# Language model data
data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")

# Classifier model data
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, test_df = df_test, vocab=data_lm.train_ds.vocab, bs=32)

data_lm.save()

# Use pre-trained AWD_LSTM language model and fine tune it
learn = language_model_learner(data_lm,  arch = AWD_LSTM, pretrained = True, drop_mult=0.5)

# Find the optimal learning rate
# learn.lr_find()
# learn.recorder.plot(suggestion=True)
# min_grad_lr = learn.recorder.min_grad_lr

# We got the optimal learning rate as 1e-2 and train the learner object with learning rate = 1e-2
learn.fit_one_cycle(2, 1e-2)
learn.recorder.plot()

# unfreezing weights and training the rest of the NN
learn.unfreeze()
learn.fit_one_cycle(1, 1e-2)

# Save encoder to use for classification later
learn.save_encoder('s_o_encoder')

# Build classifier using data_clas object with fine tuned language model encoder
learn = text_classifier_learner(data_clas, drop_mult=0.5, arch = AWD_LSTM)
learn.load_encoder('s_o_encoder')

# Run 3 epochs with learning rate of 1e-2 for text classifier
learn.fit_one_cycle(3, 1e-2)

# Plot losses
learn.recorder.plot_losses()

preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)
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

# Save your model
# learn.save('your-model-name')