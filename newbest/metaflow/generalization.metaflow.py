from progressbar import progressbar
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from sklearn import svm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from metaflow import FlowSpec, step, Parameter, IncludeFile

class Generalization(FlowSpec):

    # WORD_DICTIONARY = IncludeFile("WORD_DICTIONARY",
    #                   help="Dictionary of words",
    #                   default=make_word_dictionary())

    train_data = Parameter('train_data',
                      help="dataset to train the model",
                      default='brunet2014')

    test_data = Parameter('test_data',
                     help='dataset to test the model',
                     default='shakiba2016')

    # WORD_DICTIONARY = Parameter('WORD_DICTIONARY',
    #                   help='Contain the word dictionary',
    #                   default=WORD_DICTIONARY)

    @step # 1
    def start(self):
        self.datasets = [self.train_data, self.test_data]
        self.STOPSET_WORDS = ['might', 'may', 'would', 'must', 'lgtm', 'could', 'can', 'good', 'great', 'nice', 'well', \
                 'better', 'worse', 'worst', 'should', 'i', "i'll", "ill", "it's", "its", "im", "i'm", \
                 "they're", "theyre", "you're", "youre", "that's", 'btw', "thats", "theres", "shouldnt", \
                 "shouldn't", "didn't", "didnt", "dont", "don't", "doesn't", "doesnt", "wasnt", "wasn't", \
                 'sense', "mon", 'monday', 'tue', 'wed', 'wednesday', 'thursday', 'lgtm', 'pinging', 'thu', 'friday', 'fri', \
                 'sat', 'saturday', 'sun', 'sunday', 'jan', 'january', 'feb', 'february', 'mar', 'march', \
                 'apr', 'april', 'may', 'jun', 'june', 'july', 'jul', 'aug', 'august', 'sep', 'september', \
                 'oct', 'october', 'nov', 'novenber', 'dec', 'december', 'pm', 'am', '//'
        ]
        self.WIKI_WORDS = './files/pre_trained_word_vectors/wiki-news-300d-1M.vec'
        self.WORD_DICTIONARY = {}
        self.next(self.make_word_dictionary)

    @step
    def make_word_dictionary(self):
        for line in open(self.WIKI_WORDS):
            values = line.split()
            self.WORD_DICTIONARY[values[0]] = np.array(values[1:], dtype='float32')
        self.next(self.read_data, foreach='datasets')
    
    @step # 3
    def read_data(self):
        self.data = pd.read_csv("./files/datasets/"+self.input+".csv", sep=",", header=None, names=['X', 'Y'])
        self.next(self.remove_stopwords)

    @step # 4
    def remove_stopwords(self):
        stopset = set(stopwords.words('english'))
        for word in self.STOPSET_WORDS:
            stopset.add(word)

        self.data['X'] = self.data['X'].apply(lambda sentence: ' '.join([word for word in sentence.lower().split() \
                                                                    if word not in (stopset)]))
        self.next(self.word_embed)

    @step # 5
    def word_embed(self):
        self.word_vector = np.zeros(np.array((self.data['X'].shape[0], 300)))
        i = 0
        for sentence in self.data['X']:
            words = sentence.split()
            for word in words:
                    if word in self.WORD_DICTIONARY:
                        self.word_vector[i] = np.add(self.word_vector[i], self.WORD_DICTIONARY[word])
            i += 1
        self.next(self.join)

    @step # 6
    def join(self, inputs):
        self.train_X = inputs[0].word_vector
        self.train_Y = inputs[0].data['Y']
        self.test_X = inputs[1].word_vector
        self.test_Y = inputs[1].data['Y']
        self.next(self.oversample)

    @step # 7
    def oversample(self):
        sm = SMOTE(random_state=42)
        self.train_X, self.train_Y = sm.fit_resample(self.train_X, self.train_Y)
        self.next(self.train)

    @step # 8
    def train(self):
        self.trained_classifier = svm.SVC().fit(self.train_X, self.train_Y)
        self.next(self.validate)

    @step # 9
    def validate(self):
        predictions = self.trained_classifier.predict(self.test_X)
        cm = confusion_matrix(self.test_Y, predictions)
        tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        self.auc = round((tpr + tnr) / 2, 4)
        self.next(self.end)

    @step # 10
    def end(self):
        print(self.auc)


if __name__ == '__main__':
    Generalization()