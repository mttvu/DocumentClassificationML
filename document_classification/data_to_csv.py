import csv

import pandas
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer
from tika import parser
from os import listdir
from os.path import isfile, join
from document_classification.document_preparer import prepare_document
bank_documents_path = 'data/bank-documents'
cbr_documents_path = 'data/cbr-documents'
government_documents_path = 'data/government-documents'
thesis_documents_path = 'data/hva-kennisbank-scripties'
invoice_documents_path = 'data/invoices'

bank_documents = [join(bank_documents_path, file) for file in listdir(bank_documents_path)]
cbr_documents = [join(cbr_documents_path, file) for file in listdir(cbr_documents_path)]
government_documents = [join(government_documents_path, file) for file in listdir(government_documents_path)]
thesis_documents = [join(thesis_documents_path, file) for file in listdir(thesis_documents_path)]
invoice_documents = [join(invoice_documents_path, file) for file in listdir(invoice_documents_path)]

stopwords_dutch = set(stopwords.words('dutch'))
stemmer = DutchStemmer(ignore_stopwords=True)


def documents_to_csv(documents, document_type):
    print(document_type)

    for x, doc in enumerate(documents):
        try:
            print(x, '/', len(documents))

            filtered_text = prepare_document(doc)

            # write to csv
            with open('dataset.csv', 'a', encoding="UTF-8") as file:
                fieldnames = ['type', 'text']
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=';')
                writer.writerow({'type': document_type, 'text': filtered_text})
        except:
            pass


def delete_invoices():
    csv.field_size_limit(2147483647)
    data = pandas.read_csv("data.csv", engine='python', error_bad_lines=False, sep=';')
    data = data[["type", "text"]]
    print(len(data))
    data = data[data["type"] != "invoice"]
    print(len(data))
    data.to_csv('dataset.csv', sep=';', index=False)
    # with open('data.csv', 'r') as inp, open('dataset.csv', 'w') as out:
    #     writer = csv.writer(out)
    #     reader = csv.reader(inp)
    #     next(reader, None)
    #     for row in reader:
    #         eval_row = pandas.eval(row)
    #         if eval_row[0] != "invoice":
    #             writer.writerow(row)


# documents_to_csv(bank_documents, 'bank')
# documents_to_csv(cbr_documents, 'cbr')
# documents_to_csv(government_documents, 'government')
# documents_to_csv(thesis_documents, 'thesis')
# documents_to_csv(invoice_documents, 'invoice')
