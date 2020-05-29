import pandas
import csv
import numpy
import sklearn
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from document_classification.document_preparer import prepare_document


def do_nothing(tokens):
    return tokens


def train_model():
    csv.field_size_limit(2147483647)

    data = pandas.read_csv("dataset.csv", engine='python', error_bad_lines=False, sep=';')
    data = data[["type", "text"]]
    class_names = ['bank', 'cbr', 'government', 'thesis', 'invoice']

    #documents = pickle.load(open('documents.pickle', 'rb'))
    documents_numpy = numpy.array(data['text'])
    documents = []
    for x, doc in enumerate(documents_numpy):
        print(x, "/", len(documents_numpy))
        documents.append(pandas.eval(doc))

    types = numpy.array(data['type'])

    docs_train, docs_test, types_train, types_test = sklearn.model_selection.train_test_split(documents, types, test_size=0.2, random_state=42)

    logistic = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=do_nothing, preprocessor=None, lowercase=False)),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression())
        ]
    )

    logistic.fit(docs_train, types_train)
    pickle.dump(logistic, open('model.pickle', 'wb'))
    logistic_prediction = logistic.predict(docs_test)

    print('logistic accuracy: ', accuracy_score(types_test, logistic_prediction))
    print(classification_report(types_test, logistic_prediction, target_names=class_names))


def load_model():
    model = pickle.load(open('document_classification/model.pickle', 'rb'))
    return model

