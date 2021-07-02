FROM python:3.8.10-slim

RUN apt-get update && apt-get install -y wget gcc build-essential python3-opencv
COPY requirements.txt .
RUN pip install -r requirements.txt

# download pre-trained models
## misc
RUN python -c "from pororo import Pororo; \
ocr = Pororo(task='ocr', lang='ko')"

COPY streamlit_app.py .
CMD ["streamlit","run","streamlit_app.py", "--server.port","8080"]
