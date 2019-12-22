import warnings
import pandas as pd
from nltk.corpus import stopwords
import numpy as np

from scipy import interp
import matplotlib.pyplot as plt
from progressbar import progressbar
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.keyedvectors import KeyedVectors
from sklearn import linear_model, svm, tree, naive_bayes, ensemble
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


data = structure(brunet2014)
data = structure(shakiba2016)
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


WIKI_WORDS = '../files/pre_trained_word_vectors/wiki-news-300d-1M.vec'
GLOVE_WORDS = '../files/pre_trained_word_vectors/glove.42B.300d.txt'
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


def tfidf_vectorizer(data):
    tf_idf_vector = TfidfVectorizer()
    data_tf_idf = tf_idf_vector.fit_transform(data['text'])

    return data_tf_idf


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
    # trained_classifier = ensemble.RandomForestClassifier().fit(train_data, train_label)

    return trained_classifier


data = remove_stopwords(data)
# data = data.sample(frac=1).reset_index(drop=True)
data = stratification(data, 10)

ptrain_sizes = []
ptest_sizes = []

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
precisions = []
recalls = []
f_measures = []
balanced_accuracies = []

table = PrettyTable()
table.field_names = ['', '% True class in Train', '% True class in Test', 'Precision', 'Recall', 'F-Measure', \
                     'Accuracy', 'Balanced Accuracy', 'AUC']

# text = tfidf_vectorizer(data)
text = word_embed(data)
text, label = oversample(text, data['label'])
# accuracies = cross_val_score(linear_model.LogisticRegression(), text, label, cv=10)
accuracies = cross_val_score(svm.SVC(), text, label, cv=10)

chunk = round(data.shape[0] / 10)

for i in range(10):
    train_data = data
    start = i * chunk
    end = start + chunk
    test_data = train_data[start:end]
    train_data = train_data.drop(test_data.index)
    p_size_in_train = round(train_data[train_data['label'] == 1].shape[0] / train_data.shape[0] * 100, 4)
    p_size_in_test = round(test_data[test_data['label'] == 1].shape[0] / test_data.shape[0] * 100, 4)
    ptrain_sizes.append(p_size_in_train)
    ptest_sizes.append(p_size_in_test)

    # train_vectors, test_vectors = tf_idf_vectorizer(data['text'], train_data['text'], test_data['text'])
    train_vectors, test_vectors = word_embed(train_data), word_embed(test_data)
    os_train_vectors, os_train_label = oversample(train_vectors, train_data['label'])
    # pdb.set_trace()

    # trained_classifier = train(train_vectors, train_data['label'])
    trained_classifier = train(os_train_vectors, os_train_label)

    predictions = trained_classifier.predict(test_vectors)
    cm = confusion_matrix(test_data['label'], predictions)

    precision = round(cm[1, 1] / (cm[0, 1] + cm[1, 1]), 4)
    recall = round(cm[1, 1] / (cm[1, 1] + cm[1, 0]), 4)
    f_measure = round((2 * precision * recall) / (precision + recall), 4)
    accuracy = round(accuracies[i], 4)
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    balanced_accuracy = round((tpr + tnr) / 2, 4)

    fpr, tpr, thresholds = roc_curve(test_data['label'], predictions)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
    table.add_row(['Fold '+ str(i+1), p_size_in_train, p_size_in_test, precision, recall, f_measure, accuracy,\
                   balanced_accuracy, round(roc_auc, 4)])
    precisions.append(precision)
    recalls.append(recall)
    f_measures.append(f_measure)
    balanced_accuracies.append(balanced_accuracy)
    # i += 1


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')


table.add_row(['-----------------------', '----------------------', '----------', '-------------', '------------',\
               '---------------------', '----------', '-------------------', '----------'])
table.add_row(['Mean', round(np.mean(ptrain_sizes), 4), round(np.mean(ptest_sizes), 4), \
               round(np.mean(precisions), 4), round(np.mean(recalls), 4), \
               round(np.mean(f_measures), 4), round(np.mean(accuracies), 4), \
               round(np.mean(balanced_accuracies), 4), round(np.mean(aucs), 4)])

table.add_row(['-----------------------', '----------------------', '----------', '-------------', '------------',\
               '---------------------', '----------', '-------------------', '----------'])
table.add_row(['Median', round(np.median(ptrain_sizes), 4), round(np.median(ptest_sizes), 4), \
               round(np.median(precisions), 4), round(np.median(recalls), 4), \
               round(np.median(f_measures), 4), round(np.median(accuracies), 4), \
               round(np.median(balanced_accuracies), 4), round(np.median(aucs), 4)])
print(table)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
