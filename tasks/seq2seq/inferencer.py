from fastapi import FastAPI
from pororo import Pororo
from pydantic import BaseModel

class DocumentQuery(BaseModel):
    document: str

spacing_model = None
paraphrase_model = None
summiraztion_model = None

app = FastAPI()

spacing_model = Pororo(task='gec', lang='ko')
paraphrase_model = Pororo(task='pg', lang='ko')
summarization_model = Pororo(task='summarization', model='extractive', lang='ko')

@app.get('/seq2seq')
def model_health_check():
    if spacing_model is None:
        return {'status': 400}

    if paraphrase_model is None:
        return {'status': 400}

    if summarization_model is None:
        return {'status': 400}

    return {'status': 200}

@app.get('/seq2seq/spacing/{query}')
def predict_spacing(query: str):
    pred = spacing_model(query)
    return pred

@app.post('/seq2seq/paraphrase')
def predict_paraphrase(query: DocumentQuery):
    pred = paraphrase_model.predict(query.document)
    return pred

@app.post('/seq2seq/summarization')
def predict_summarization(query: DocumentQuery):
    pred = summarization_model.predict(query.document)
    return pred
