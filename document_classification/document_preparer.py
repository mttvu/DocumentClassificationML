from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import DutchStemmer
from tika import parser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def prepare_document(doc):
    stopwords_dutch = set(stopwords.words('dutch'))
    stemmer = DutchStemmer(ignore_stopwords=True)
    parsed_text = convert_pdf_to_txt(doc)

   # print(contents)
#    parsed_text = parser.from_file(doc)
    tokenized_text = word_tokenize(parsed_text)
    # stem words
    for i, token in enumerate(tokenized_text):
        tokenized_text[i] = stemmer.stem(token)

    # clean text
    # remove punctuation
    tokenized_text = [word for word in tokenized_text if word.isalpha()]
    # remove stopwords
    filtered_text = [word for word in tokenized_text if word not in stopwords_dutch]
    return filtered_text


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


def do_nothing(tokens):
    return tokens
