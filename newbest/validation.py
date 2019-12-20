import warnings
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from progressbar import progressbar
from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models.keyedvectors import KeyedVectors
from sklearn import linear_model, svm, tree, naive_bayes, ensemble
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import pdb
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
from metaflow import FlowSpec, step

class Validation(FlowSpec):

    @step
    def start(self):
        self.datasets = ['brunet2014', 
                         'shakiba2016']
        # self.datasets = ['brunet2014', 
        #                 'shakiba2016', 
        #                 'viviani2018', 
        #                 'satd', 
        #                 'stackoverflow']
        self.STOPSET_WORDS = ['might', 'may', 'would', 'must', 'lgtm', 'could', 'can', 'good', 'great', 'nice', 'well', \
                 'better', 'worse', 'worst', 'should', 'i', "i'll", "ill", "it's", "its", "im", "i'm", \
                 "they're", "theyre", "you're", "youre", "that's", 'btw', "thats", "theres", "shouldnt", \
                 "shouldn't", "didn't", "didnt", "dont", "don't", "doesn't", "doesnt", "wasnt", "wasn't", \
                 'sense', "mon", 'monday', 'tue', 'wed', 'wednesday', 'thursday', 'lgtm', 'pinging', 'thu', 'friday', 'fri', \
                 'sat', 'saturday', 'sun', 'sunday', 'jan', 'january', 'feb', 'february', 'mar', 'march', \
                 'apr', 'april', 'may', 'jun', 'june', 'july', 'jul', 'aug', 'august', 'sep', 'september', \
                 'oct', 'october', 'nov', 'novenber', 'dec', 'december', 'pm', 'am', '//'
        ]
        n = 10
        self.WIKI_WORDS = './files/pre_trained_word_vectors/wiki-news-300d-1M.vec'
        self.WORD_DICTIONARY = {}
        self.n_splits = n
        self.folds = np.arange(n)
        self.next(self.read_data, foreach='datasets')

    # @step
    # def make_word_dictionary(self):
    #     print('Loading word-embedding file and making dictionary...')
    #     for line in progressbar(open(self.WIKI_WORDS)):
    #         values = line.split()
    #         self.WORD_DICTIONARY[values[0]] = np.array(values[1:], dtype='float32')
    #     self.next(self.read_data, foreach='datasets')

    @step
    def read_data(self):
        self.data = pd.read_csv("./files/datasets/"+self.input+".csv", sep=",", header=None, names=['text', 'label'])
        self.next(self.remove_stopwords)

    @step
    def remove_stopwords(self):
        stopset = set(stopwords.words('english'))
        for word in self.STOPSET_WORDS:
            stopset.add(word)

        self.data['text'] = self.data['text'].apply(lambda sentence: ' '.join([word for word in sentence.lower().split() \
                                                                    if word not in (stopset)]))
        self.next(self.stratification)

    @step
    def stratification(self):
        stratified_data = pd.DataFrame()
        positive_data = self.data[self.data['label'] == 1]
        negative_data = self.data[self.data['label'] == 0]
        p_data_size = positive_data.shape[0]
        n_data_size = negative_data.shape[0]
        positive_chunk = ([p_data_size // self.n_splits + (1 if x < p_data_size % self.n_splits else 0) for x in range(self.n_splits)])
        negative_chunk = ([n_data_size // self.n_splits + (1 if x < n_data_size % self.n_splits else 0) for x in range(self.n_splits)])
        p_index, n_index = 0, 0
        for i in range(self.n_splits):
            tmp_data = positive_data[p_index:p_index+positive_chunk[i]]
            tmp_data = tmp_data.append(negative_data[n_index:n_index+negative_chunk[i]], ignore_index=True)
            tmp_data = tmp_data.sample(frac=1).reset_index(drop=True)
            stratified_data = stratified_data.append(tmp_data, ignore_index=True)
            p_index += positive_chunk[i]
            n_index += negative_chunk[i]

        self.data = stratified_data
        self.next(self.divide_train_test, foreach='folds')

    @step
    def divide_train_test(self):
        chunk = round(self.data.shape[0] / 10)
        train_data = self.data
        start = (self.input+1) * chunk
        end = start + chunk
        test_data = train_data[start:end]
        train_data = train_data.drop(test_data.index)
        self.data = [train_data, test_data]

        self.next(self.word_embed, foreach='data')
        
    @step
    def word_embed(self):
        self.word_vector = np.zeros(np.array((self.input.shape[0], 300)))
        i = 0
        for sentence in self.input['text']:
            words = sentence.split()
            for word in words:
                    if word in self.WORD_DICTIONARY:
                        self.word_vector[i] = np.add(self.word_vector[i], self.WORD_DICTIONARY[word])
            i += 1
        self.next(self.join_word_vectors)

    @step 
    def join_word_vectors(self, inputs):
        self.train_vector = inputs[0].word_vector
        self.test_vector = inputs[1].word_vector
        self.next(self.oversample)

    @step
    def oversample(self):
        sm = SMOTE(random_state=42)
        self.train_vector['text'], self.train_vector['label'] = sm.fit_resample(self.train_vector['text'], self.train_vector['label'])
        self.next(self.train)

    @step
    def train(self):
        self.trained_classifier = svm.SVC().fit(self.train_vector['text'], self.train_vector['label'])
        self.next(self.validate)

    @step
    def validate(self):
        self.prediction = self.trained_classifier.predict(self.test_vectors)
        self.next(self.join_predictions)

    @step 
    def join_predictions(self, inputs):
        self.k_fold_predictions = [input.prediction for input in inputs]
        self.next(self.join)

    @step
    def join(self, inputs):
        self.results = [input.k_fold_predictions for input in inputs]
        self.next(self.end)

    @step
    def end(self):
        print(self.results)

if __name__ == "__main__":
    Validation()