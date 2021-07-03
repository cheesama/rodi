FROM python:3.8.10-slim

RUN apt-get update && apt-get install -y wget gcc build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt

# download pre-trained models
## seq2seq
RUN python -c "from pororo import Pororo; \
spacing_model = Pororo(task='gec', lang='ko'); \ 
pg_model = Pororo(task='pg', lang='ko'); \ 
summ_model = Pororo(task='sentiment', model='brainbert.base.ko.shopping', lang='ko')"

COPY inferencer.py .

CMD ["uvicorn", "inferencer:app", "--reload", "--host", "0.0.0.0"]


