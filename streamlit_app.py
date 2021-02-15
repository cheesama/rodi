from pororo import Pororo

import streamlit as st

# Set page title
st.title('Prototyping NLP models with Pororo')

# Load similarity model
st.subheader('Semantic Textual Similarity')
with st.spinner('Loading similarity model...'):
    similarity_model = Pororo(task="similarity", lang="ko")

    query1_input = st.text_input('query1:')
    query2_input = st.text_input('query2:')

    if query1_input != '' and query2_input != '':
        with st.spinner('Calculating Similarity...'):
             st.write(f'Similarity: {similarity_model(query1_input, query2_input)}')
             
# Load sentiment_analysis model
st.subheader('Sentiment Analysis')
with st.spinner('Loading sentiment_analysis model...'):
    sentiment_model = Pororo(task="sentiment", model="brainbert.base.ko.shopping", lang="ko")

# Load named entity recognition model
st.subheader('Named Entity Recognition')
with st.spinner('Loading sentiment_analysis model...'):
    ner_model = Pororo(task="ner", lang="ko")

