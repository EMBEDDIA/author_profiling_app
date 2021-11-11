import os
import dill
import joblib

from flask import (
    Flask,
    jsonify,
    send_from_directory,
    request,
    redirect,
    url_for
)

from flask_restx import Api, Resource, fields, abort, reqparse
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

from . import api_functions
from . import tfidf_kingdom as ap

import sys
sys.modules['tfidf_kingdom'] = ap

import nltk
nltk.download('stopwords')

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0',
          title='API services',
          description='English and Spanish author profiling classifier REST API')
ns = api.namespace('rest_api', description='REST services API')



## load bot model
en_bot_vectorizer_file = open("project/train_data/vectorizer_bot_en.pickle", 'rb')
en_bot_vectorizer = dill.load(en_bot_vectorizer_file)
en_bot_encoder_file = open("project/train_data/encoder_bot_en.pickle", 'rb')
en_bot_encoder = dill.load(en_bot_encoder_file)
en_bot_model = joblib.load("project/train_data/trained_LR_bot_en.pkl")

## load gender model
en_gender_vectorizer_file = open("project/train_data/vectorizer_gender_en.pickle", 'rb')
en_gender_vectorizer = dill.load(en_gender_vectorizer_file)
en_gender_encoder_file = open("project/train_data/encoder_gender_en.pickle", 'rb')
en_gender_encoder = dill.load(en_gender_encoder_file)
en_gender_model = joblib.load("project/train_data/trained_LR_gender_en.pkl")

## load es bot model
es_bot_vectorizer_file = open("project/train_data/vectorizer_bot_es.pickle", 'rb')
es_bot_vectorizer = dill.load(es_bot_vectorizer_file)
es_bot_encoder_file = open("project/train_data/encoder_bot_es.pickle", 'rb')
es_bot_encoder = dill.load(es_bot_encoder_file)
es_bot_model = joblib.load("project/train_data/trained_LR_bot_es.pkl")

## load gender model
es_gender_vectorizer_file = open("project/train_data/vectorizer_gender_es.pickle", 'rb')
es_gender_vectorizer = dill.load(es_gender_vectorizer_file)
es_gender_encoder_file = open("project/train_data/encoder_gender_es.pickle", 'rb')
es_gender_encoder = dill.load(es_gender_encoder_file)
es_gender_model = joblib.load("project/train_data/trained_LR_gender_es.pkl")



# input and output definitions
ap_classifier_input = api.model('APClassifierInput', {
    'text': fields.String(required=True, description='One or (ideally) more tweets or comments (concatenated together) belonging to a specific user'),
})

ap_classifier_output = api.model('APClassifierOutput', {
    'decision': fields.String(description='list of predicted classes'),
})


@ns.route('/author_profiling/')
class KeywordExtractor(Resource):
    @ns.doc('Determine the author of the text')
    @ns.expect(ap_classifier_input, validate=True)
    @ns.marshal_with(ap_classifier_output)
    def post(self):
        decision = api_functions.ap_classify(api.payload['text'], en_bot_vectorizer, en_bot_model, en_bot_encoder, en_gender_vectorizer,
                                             en_gender_model, en_gender_encoder, es_bot_vectorizer, es_bot_model, es_bot_encoder,
                                             es_gender_vectorizer, es_gender_model, es_gender_encoder)
        return {"decision": decision}


@ns.route('/health/')
class Health(Resource):
    @ns.response(200, "successfully fetched health details")
    def get(self):
        return {"status": "running", "message": "Health check successful"}, 200, {}
