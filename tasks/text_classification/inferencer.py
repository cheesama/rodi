from fastapi import FastAPI
from pororo import Pororo
from pydantic import BaseModel

class SimilarityQuery(BaseModel):
    sentence1: str
    sentence2: str

review_score_model = None
embedding_model = None
sentiment_model = None
similarity_model = None

app = FastAPI()

review_score_model = Pororo(task='review', lang='ko')
embedding_model = Pororo(task='sentence_embedding', lang='ko')
sentiment_model = Pororo(task='sentiment', model='brainbert.base.ko.shopping', lang='ko')
similarity_model = Pororo(task='similarity', lang='ko')

@app.get('/text_classification')
def model_health_check():
    if review_score_model is None:
        return {'status': 400}

    if embedding_model is None:
        return {'status': 400}

    if sentiment_model is None:
        return {'status': 400}

    if similarity_model is None:
        return {'status': 400}

    return {'status': 200}

@app.get('/text_classification/review_score/{query}')
def predict_review_score(query: str):
    pred = review_score_model(query)
    return pred

@app.get('/text_classification/embedding/{query}')
def predict_embedding(query: str):
    pred = embedding_model(query)
    return pred
      
@app.get('/text_classification/sentiment/{query}')
def predict_sentiment(query: str):
    pred = sentiment_model(query)
    return pred

@app.post('/text_classification/similarity')
def predict_similarity(query: SimilarityQuery):
    pred = similarity_model.predict(query.sentence1, query.sentence2)
    return pred
