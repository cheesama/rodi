from pororo import Pororo

import streamlit as st

# Set page title
st.title('Prototyping NLP models with Pororo')

# Load similarity model
with st.spinner('Loading similarity model...'):
    similarity_model = Pororo(task="similarity", lang="ko")

# Load sentiment_analysis model
with st.spinner('Loading sentiment_analysis model...'):
    sentiment_model = Pororo(task="sentiment", model="brainbert.base.ko.shopping", lang="ko")

# Load named entity recognition model
with st.spinner('Loading sentiment_analysis model...'):
    ner_model = Pororo(task="ner", lang="ko")

