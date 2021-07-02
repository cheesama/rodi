from fastapi import FastAPI
from pororo import Pororo
from pydantic import BaseModel

class DocumentQuery(BaseModel):
    question: str
    document: str

fib_model = None
mrc_model = None
ner_model = None
pos_model = None

app = FastAPI()

fib_model = Pororo(task='fib', lang='ko') # fill in the blank task
mrc_model = Pororo(task='mrc', lang='ko')
ner_model = Pororo(task='ner', lang='ko')
pos_model = Pororo(task='pos', lang='ko')

@app.get('/sequence_tagging')
def model_health_check():
    if fib_model is None:
        return {'status': 400}

    if mrc_model is None:
        return {'status': 400}

    if ner_model is None:
        return {'status': 400}

    if pos_model is None:
        return {'status': 400}

    return {'status': 200}

@app.get('/sequence_tagging/ner/{query}')
def predict_ner(query: str):
    pred = ner_model(query)
    return pred

@app.get('/sequence_tagging/pos/{query}')
def predict_pos(query: str):
    pred = pos_model(query)
    return pred
      
@app.get('/sequence_tagging/fib/{query}')
def predict_fib(query: str):
    pred = fib_model(query)
    return pred

@app.post('/sequence_tagging/mrc')
def predict_mrc(query: DocumentQuery):
    pred = mrc_model.predict(query.question, query.document)
    return pred
