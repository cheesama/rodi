FROM python:3.8.10-slim

RUN apt-get update && apt-get install -y wget gcc build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt

# download pre-trained models
## sequence tagging
RUN python -c "from pororo import Pororo; \
fib = Pororo(task='fib', lang='ko'); \
mrc = Pororo(task='mrc', lang='ko'); \
ner = Pororo(task='ner', lang='ko'); \
pos = Pororo(task='pos', lang='ko')"

COPY inferencer.py .

CMD ["uvicorn", "inferencer:app", "--reload", "--host", "0.0.0.0"]


