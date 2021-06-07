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

## sqeuence tagging
RUN python -c "from pororo import Pororo; \
fib = Pororo(task='fib', lang='ko'); \ 
mrc = Pororo(task='mrc', lang='ko'); \
ner = Pororo(task='ner', lang='ko'); \
pos = Pororo(task='pos', lang='ko')"

## seq2seq
RUN python -c "from pororo import Pororo; \
spacing = Pororo(task='gec', lang='ko'); \
pg = Pororo(task='pg', lang='ko'); \
summ = Pororo(task='summarization', model='extractive', lang='ko')"

## misc
#RUN python -c "from pororo import Pororo; \
#tk = Pororo(task='tokenization', lang='ko', model='bpe32k.ko'); \
#word2vec = Pororo('word2vec', lang='ko'); \
#ocr = Pororo(task='ocr', lang='ko')"

CMD ["streamlit","run","streamlit_app.py", "--server.port","8080"]
