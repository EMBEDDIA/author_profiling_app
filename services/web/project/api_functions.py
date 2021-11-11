import langid
from . import tfidf_kingdom as ap

def get_language(text):
	lang = langid.classify(text)[0]
	if lang == 'es':
		lang = 'spanish'
	else:
		lang = 'english'
	return lang


def ap_classify(text, en_bot_vectorizer, en_bot_model, en_bot_encoder, en_gender_vectorizer, en_gender_model, en_gender_encoder,
                      es_bot_vectorizer, es_bot_model, es_bot_encoder, es_gender_vectorizer, es_gender_model, es_gender_encoder):
    lang = get_language(text)
    print('Language detected: ', lang)
    docs = [text]
    test_df = ap.build_dataframe(docs)

    if lang == 'english':
        predict_features = en_bot_vectorizer.transform(test_df)

        predictions = en_bot_model.predict(predict_features)
        predictions = en_bot_encoder.inverse_transform(predictions)
        if predictions[0] == 'bot':
            return predictions[0]

        else:
            predict_features = en_gender_vectorizer.transform(test_df)

            predictions = en_gender_model.predict(predict_features)
            predictions = en_gender_encoder.inverse_transform(predictions)
            return predictions[0]
    else:
        predict_features = es_bot_vectorizer.transform(test_df)

        predictions = es_bot_model.predict(predict_features)
        predictions = es_bot_encoder.inverse_transform(predictions)
        if predictions[0] == 'bot':
            return predictions[0]

        else:
            predict_features = es_gender_vectorizer.transform(test_df)

            predictions = es_gender_model.predict(predict_features)
            predictions = es_gender_encoder.inverse_transform(predictions)
            return predictions[0]


                                   