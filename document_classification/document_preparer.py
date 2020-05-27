from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer
from tika import parser


def prepare_document(doc):
    stopwords_dutch = set(stopwords.words('dutch'))
    stemmer = DutchStemmer(ignore_stopwords=True)

    parsed_text = parser.from_file(doc)
    tokenized_text = word_tokenize(parsed_text['content'])
    # stem words
    for i, token in enumerate(tokenized_text):
        tokenized_text[i] = stemmer.stem(token)

    # clean text
    # remove punctuation
    tokenized_text = [word for word in tokenized_text if word.isalpha()]
    # remove stopwords
    filtered_text = [word for word in tokenized_text if word not in stopwords_dutch]
    return filtered_text

