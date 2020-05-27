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
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# csv.field_size_limit(2147483647)
# data = pandas.read_csv("data.csv", engine='python', error_bad_lines=False, sep=';')
# data = data[["type", "text"]]
# class_names = ['bank', 'cbr', 'government', 'thesis', 'invoice']
#
# predict = 'type'
#
# documents = numpy.array(data['text'])
# types = numpy.array(data[predict])

encoded_documents = pickle.load(open('encoded_documents.pickle', 'rb'))
encoded_types = pickle.load(open('encoded_types.pickle', 'rb'))

label_index = {'bank': 0, 'cbr': 1, 'government': 2, 'thesis': 3, 'invoice': 4}
word_index = pickle.load(open('word_index.pkl', 'rb'))
# word_index = {0: '<PAD>', 1: '<START>', 2: '<UNK>', 3: '<UNUSED>'}
# word_index_file = open('word_index.txt', 'r')
# word_index_extra = pandas.eval(word_index_file.read())
# word_index_extra = {i: word_index_extra[i] for i in range(0, len(word_index_extra))}
# word_index_extra = {(key + 3): value for key, value in word_index_extra.items()}
# word_index.update(word_index_extra)

#word_index = dict([(value, key) for (key, value) in word_index.items()])

#pickle.dump(word_index, open("word_index.pkl", "wb"))

# for x, document in enumerate(documents):
#     print(x, '/', len(documents))
#     document = pandas.eval(document)
#     for word in document:
#         if word not in word_index:
#             word_index.append(word)


def encode_document(document):
    encoded = [1]
    document = pandas.eval(document)
    for word in document:
        if word in word_index:
            encoded.append(word_index[word])
        else:
            encoded.append(2)
    return encoded
# encoded_docs = []
# for x, doc in enumerate(documents):
#     print(x, '/', len(documents))
#     encoded_docs.append(encode_document(doc))
#
# pickle.dump(encoded_docs, open('encoded_documents.pickle', 'wb'))


def encode_types(types):
    encoded = []
    for type in types:
        encoded.append(label_index[type])
    pickle.dump(encoded, open('encoded_types.pickle', 'wb'))

encoded_types = keras.utils.to_categorical(encoded_types)

docs_train, docs_test, types_train, types_test = sklearn.model_selection.train_test_split(encoded_documents,
                                                                                          encoded_types, test_size=0.3, random_state=42)
### KERAS ###
docs_train = keras.preprocessing.sequence.pad_sequences(docs_train, value=0, padding="post", maxlen=1000)
docs_test = keras.preprocessing.sequence.pad_sequences(docs_test, value=0, padding="post", maxlen=1000)

print('shape docs train: ',  docs_train.shape)
model = keras.Sequential([
    keras.layers.Embedding(len(word_index)+1, 100, input_length=docs_train.shape[1]),
    keras.layers.SpatialDropout1D(0.2),
    keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.Dense(5, activation="softmax")
])


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(docs_train, types_train, epochs=10, batch_size=64)

prediction = model.predict(docs_test)
test_loss, test_acc = model.evaluate(docs_test, types_test)

print("test acc: ", test_acc)


