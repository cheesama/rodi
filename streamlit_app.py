from pororo import Pororo
from spacy_streamlit import visualize_ner

import streamlit as st

# Set page title
st.title('Prototyping NLP models with Pororo')

# Load similarity model
@st.cache(allow_output_mutation=True)
def load_similarity_model():
    st.subheader('Semantic Textual Similarity')
    with st.spinner('Loading similarity model...'):
        similarity_model = Pororo(task="similarity", lang="ko")

        return similarity_model

# Load sentiment_analysis model
@st.cache(allow_output_mutation=True)
def load_sentiment_model():
    st.subheader('Sentiment Analysis')
    with st.spinner('Loading sentiment_analysis model...'):
        sentiment_model = Pororo(task="sentiment", model="brainbert.base.ko.shopping", lang="ko")

        return sentiment_model

# Load named entity recognition model
@st.cache(allow_output_mutation=True)
def load_ner_model():
    st.subheader('Named Entity Recognition')
    with st.spinner('Loading sentiment_analysis model...'):
        ner_model = Pororo(task="ner", lang="ko")

        return ner_model

if __name__ == '__main__':
    similarity_model = load_similarity_model()
    sim_query1_input = st.text_input('query1:')
    sim_query2_input = st.text_input('query2:')
    if sim_query1_input != '' and sim_query2_input != '':
        with st.spinner('Predicting...'):
            st.write(f'Similarity: {similarity_model(sim_query1_input, sim_query2_input)}')

    sentiment_model = load_sentiment_model()
    sentiment_query_input = st.text_input('query:')
    if sentiment_query_input != '':
        with st.spinner('Predicting...'):
            st.write('Result:')
            st.json(sentiment_model(sentiment_query_input, show_probs=True))

    


