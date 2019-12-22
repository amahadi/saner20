import warnings
import pandas as pd
from nltk.corpus import stopwords
import numpy as np

from scipy import interp
import matplotlib.pyplot as plt
from progressbar import progressbar
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.keyedvectors import KeyedVectors
from sklearn import linear_model, svm, tree, naive_bayes
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import pdb
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable


brunet2014 = './files/datasets/brunet2014.csv'
shakiba2016 = './files/datasets/shakiba2016.csv'
satd = './files/datasets/satd.csv'
viviani2018 = './files/datasets/viviani2018.csv'
stackoverflow = './files/datasets/stackoverflow.csv'


def structure(data_file_path):
    data = pd.read_csv(data_file_path, sep=",", header=None, names=['text', 'label'])
    return data


train_data = structure(brunet2014)
test_data = structure(viviani2018)
# print(data.head())


STOPSET_WORDS = ['might', 'may', 'would', 'must', 'lgtm', 'could', 'can', 'good', 'great', 'nice', 'well', \
                 'better', 'worse', 'worst', 'should', 'i', "i'll", "ill", "it's", "its", "im", "i'm", \
                 "they're", "theyre", "you're", "youre", "that's", 'btw', "thats", "theres", "shouldnt", \
                 "shouldn't", "didn't", "didnt", "dont", "don't", "doesn't", "doesnt", "wasnt", "wasn't", \
                 'sense', "mon", 'monday', 'tue', 'wed', 'wednesday', 'thursday', 'lgtm', 'pinging', 'thu', 'friday', 'fri', \
                 'sat', 'saturday', 'sun', 'sunday', 'jan', 'january', 'feb', 'february', 'mar', 'march', \
                 'apr', 'april', 'may', 'jun', 'june', 'july', 'jul', 'aug', 'august', 'sep', 'september', \
                 'oct', 'october', 'nov', 'novenber', 'dec', 'december', 'pm', 'am', '//'
]


def remove_stopwords(data):
    stopset = set(stopwords.words('english'))
    for word in STOPSET_WORDS:
        stopset.add(word)

    data['text'] = data['text'].apply(lambda sentence: ' '.join([word for word in sentence.lower().split() \
                                                                 if word not in (stopset)]))
    return data

WIKI_WORDS = './files/pre_trained_word_vectors/wiki-news-300d-1M.vec'
WORD_DICTIONARY = {}


def make_word_dictionary():
    print('Loading word-embedding file and making dictionary...')
    for line in progressbar(open(WIKI_WORDS)):
        values = line.split()
        WORD_DICTIONARY[values[0]] = np.array(values[1:], dtype='float32')


def word_embed(data):
    # load the pre-trained word-embedding vectors
    if len(WORD_DICTIONARY) == 0:
        make_word_dictionary()
    word_vector = np.zeros(np.array((data.shape[0], 300)))
    i = 0
    # print('embedding word and converting to vector...')
    for sentence in data['text']:
        words = sentence.split()
        for word in words:
                if word in WORD_DICTIONARY:
                        word_vector[i] = np.add(word_vector[i], WORD_DICTIONARY[word])
        i += 1
    # print("Word Embedding is completed")
    return word_vector


def tf_idf_vectorizer(data, train_data, test_data):
    tf_idf_vector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    tf_idf_vector.fit(data)
    train_data_tf_idf = tf_idf_vector.transform(train_data)
    test_data_tf_idf = tf_idf_vector.transform(test_data)

    return train_data_tf_idf, test_data_tf_idf


def oversample(X, Y):
    sm = SMOTE(random_state=42)

    return sm.fit_resample(X, Y)


def train(train_data, train_label):
    # trained_classifier = linear_model.LogisticRegression().fit(train_data, train_label)
    # trained_classifier = naive_bayes.MultinomialNB().fit(train_data, train_label)
    # trained_classifier = tree.DecisionTreeClassifier().fit(train_data, train_label)
    trained_classifier = svm.SVC().fit(train_data, train_label)

    return trained_classifier


train_data = remove_stopwords(train_data)
test_data = remove_stopwords(test_data)
data = pd.concat([train_data, test_data])
# data = data.sample(frac=1).reset_index(drop=True)
# data = stratification(data, 10)

# ptrain_sizes = []
# ptest_sizes = []
#
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
# precisions = []
# recalls = []
# f_measures = []
# balanced_accuracies = []

table = PrettyTable()
table.field_names = ['Precision', 'Recall', 'F-Measure', \
                     'Balanced Accuracy', 'AUC']

# train_vectors, test_vectors = tf_idf_vectorizer(data['text'], train_data['text'], test_data['text'])
train_vectors = word_embed(train_data)
test_vectors = word_embed(test_data)
os_train_vectors, os_train_label = oversample(train_vectors, train_data['label'])
# pdb.set_trace()

# trained_classifier = train(train_vectors, train_data['label'])
trained_classifier = train(os_train_vectors, os_train_label)

predictions = trained_classifier.predict(test_vectors)
cm = confusion_matrix(test_data['label'], predictions)

precision = round(cm[1, 1] / (cm[0, 1] + cm[1, 1]), 4)
recall = round(cm[1, 1] / (cm[1, 1] + cm[1, 0]), 4)
f_measure = round((2 * precision * recall) / (precision + recall), 4)
tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1])
balanced_accuracy = round((tpr + tnr) / 2, 4)

fpr, tpr, thresholds = roc_curve(test_data['label'], predictions)
# tprs.append(interp(mean_fpr, fpr, tpr))
# tprs[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
# aucs.append(roc_auc)
plt.plot(fpr, tpr, lw=1, alpha=0.8, color='b',
          label='ROC (AUC = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.legend(loc="lower right")
plt.show()
table.add_row([precision, recall, f_measure, balanced_accuracy, round(roc_auc, 4)])

print(table)
