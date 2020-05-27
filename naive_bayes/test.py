import csv
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer
from tika import parser
from os import listdir
from os.path import isfile, join

stopwords_dutch = set(stopwords.words('dutch'))
stemmer = DutchStemmer(ignore_stopwords=True)

parsed_text = parser.from_file('../document_classification/data/Orderbevestiging3999857.pdf')
tokenized_text = word_tokenize(parsed_text['content'])
# stem words
for i, token in enumerate(tokenized_text):
    tokenized_text[i] = stemmer.stem(token)

# clean text
# remove punctuation
tokenized_text = [word for word in tokenized_text if word.isalpha()]
# remove stopwords
filtered_text = [word for word in tokenized_text if word not in stopwords_dutch]
filtered_text = [filtered_text]

def do_nothing(tokens):
    return tokens


model = pickle.load(open('naive_bayes_model.pickle', 'rb'))
prediction = model.predict(filtered_text)

print(prediction)
