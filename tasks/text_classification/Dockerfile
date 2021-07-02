FROM python:3.8.10-slim

RUN apt-get update && apt-get install -y wget gcc build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt

# download pre-trained models
## text classification
RUN python -c "from pororo import Pororo; \
similarity_model = Pororo(task='similarity', lang='ko'); \
embedding_model = Pororo(task='sentence_embedding', lang='ko'); \
review_score_model = Pororo(task='review', lang='ko'); \ 
sentiment_model = Pororo(task='sentiment', model='brainbert.base.ko.shopping', lang='ko')"

COPY inferencer.py .

CMD ["uvicorn", "inferencer:app", "--reload", "--host", "0.0.0.0"]


