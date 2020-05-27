import pandas
import csv
import sys
import numpy
import ast
import sklearn
import json
import pickle
from functools import reduce
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


csv.field_size_limit(2147483647)
data = pandas.read_csv("data.csv", engine='python', error_bad_lines=False, sep=';')
data = data[["type", "text"]]
class_names = ['bank', 'cbr', 'government', 'thesis', 'invoice']

predict = 'type'

documents = pickle.load(open('documents.pickle', 'rb'))
types = numpy.array(data[predict])

docs_train, docs_test, types_train, types_test = sklearn.model_selection.train_test_split(documents, types, test_size=0.2, random_state=42)

def do_nothing(tokens):
    return tokens

model = Pipeline(
    [
        ('vect', CountVectorizer(tokenizer=do_nothing, preprocessor=None, lowercase=False)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ]
)

knn = Pipeline(
    [
        ('vect', CountVectorizer(tokenizer=do_nothing, preprocessor=None, lowercase=False)),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=35))
    ]
)
svc = Pipeline(
    [
        ('vect', CountVectorizer(tokenizer=do_nothing, preprocessor=None, lowercase=False)),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC())
    ]
)

logistic = Pipeline(
    [
        ('vect', CountVectorizer(tokenizer=do_nothing, preprocessor=None, lowercase=False)),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ]
)
model.fit(docs_train, types_train)
knn.fit(docs_train, types_train)
svc.fit(docs_train, types_train)
logistic.fit(docs_train, types_train)

types_prediction = model.predict(docs_test)
knn_prediction = knn.predict(docs_test)
svc_prediction = svc.predict(docs_test)
logistic_prediction = logistic.predict(docs_test)
print('naive bayes accuracy: ', accuracy_score(types_test, types_prediction))
print('knn accuracy: ', accuracy_score(types_test, knn_prediction))
print('svc accuracy: ', accuracy_score(types_test, svc_prediction))
print('logistic accuracy: ', accuracy_score(types_test, logistic_prediction))
# print(classification_report(types_test, logistic_prediction, target_names=class_names))