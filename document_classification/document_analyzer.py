import pickle
import tempfile
import os

import joblib

from document_classification.document_preparer import prepare_document
from flask import Blueprint, request, app

document_analyzer = Blueprint("document_analyzer", __name__)
tempdir = tempfile.gettempdir()


class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "model_trainer"
        return super().find_class(module, name)


@document_analyzer.route("/", methods=['POST'])
def predict_category():
    file = request.files['document']
    path = os.path.join(tempdir, file.filename)
    file.save(path)
    unpickler = MyCustomUnpickler(joblib.load(open('document_classification/model.pickle', 'rb')))
    model = unpickler.load()
    prediction = model.predict([prepare_document(path)])
    return prediction[0]


def predict_category_test(document):
    model = pickle.load(open('model.pickle', 'rb'))
    prediction = model.predict([prepare_document(document)])
    return prediction[0]


def test():
    print('thesis - prediction: ', predict_category_test('data/PRAKT_ws_S1031879_Bakker.pdf'))
    print('statement of bank account (rekeningafschrift) - prediction: ', predict_category_test('data/Rekeningafschrift_20200002_2020_20200002-11.pdf'))
    print('invoice - prediction: ', predict_category_test('data/factuur_446500903.pdf'))
    #print('government - prediction: ', predict_category_test('data/gmb-2020-81656.pdf'))
    print('cbr - prediction: ', predict_category_test('data/rijgesd411.pdf'))



