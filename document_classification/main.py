from flask import Flask
from document_classification.document_analyzer import document_analyzer
from document_classification.model_trainer import do_nothing
app = Flask(__name__)
app.register_blueprint(document_analyzer, url_prefix='/analyzer')

if __name__ == '__main__':
    app.run(debug=True)