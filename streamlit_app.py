from pororo import Pororo
from spacy import displacy

import streamlit as st
import random

# Set page title
st.title('Rapid pOroro Demo Inferencer')

# Load similarity model
@st.cache(allow_output_mutation=True)
def load_similarity_model():
    with st.spinner('Loading similarity model...'):
        similarity_model = Pororo(task="similarity", lang="ko")

        return similarity_model

# Load sentiment_analysis model
@st.cache(allow_output_mutation=True)
def load_sentiment_model():
    with st.spinner('Loading sentiment_analysis model...'):
        sentiment_model = Pororo(task="sentiment", model="brainbert.base.ko.shopping", lang="ko")

        return sentiment_model

# Load named entity recognition model
@st.cache(allow_output_mutation=True)
def load_ner_model():
    with st.spinner('Loading sentiment_analysis model...'):
        ner_model = Pororo(task="ner", lang="ko")
        
        return ner_model

# Load tts model
@st.cache(allow_output_mutation=True)
def load_tts_model():
    with st.spinner('Loading tts model...'):
        tts_model = Pororo(task="tts", lang="multi")
        
        return tts_model

def hf_ents_to_displacy_format(ents, ignore_entities=[]):
    s_ents = {}
    s_ents["text"] = " ".join([e[0] for e in ents])
    spacy_ents = []
    start_pointer = 0
    if "entity_group" in ents[0]:
        entity_key = "entity_group"
    else:
        entity_key = "entity"
    for i, ent in enumerate(ents):
        if ent[1] not in ignore_entities:
            spacy_ents.append(
                {
                    "start": start_pointer,
                    "end": start_pointer + len(ent[0]),
                    "label": ent[1],
                }
            )
        start_pointer = start_pointer + len(ent[0]) + 1
    s_ents["ents"] = spacy_ents
    s_ents["title"] = None

    return s_ents

def add_colormap(labels):
    color_map = {}
    for label in labels:
        #if label not in color_map:
        rand_color = "#"+"%06x" % random.randint(0, 0xFFFFFF)
        color_map[label]=rand_color

    return color_map

# from https://github.com/explosion/spacy-streamlit/util.py#L26
WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

if __name__ == '__main__':
    st.subheader('Semantic Textual Similarity')
    similarity_model = load_similarity_model()
    sim_query1_input = st.text_input('query1:')
    sim_query2_input = st.text_input('query2:')
    if sim_query1_input != '' and sim_query2_input != '':
        with st.spinner('Predicting...'):
            st.write(f'Similarity: {similarity_model(sim_query1_input, sim_query2_input)}')

    st.subheader('Sentiment Analysis')
    sentiment_model = load_sentiment_model()
    sentiment_query_input = st.text_input('sentiment query:')
    if sentiment_query_input != '':
        with st.spinner('Predicting...'):
            st.write('Result:')
            st.json(sentiment_model(sentiment_query_input, show_probs=True))

    st.subheader('Named Entity Recognition')
    ner_model = load_ner_model()
    ner_query_input = st.text_input('ner query:')
    if ner_query_input != '':
        with st.spinner('Predicting...'):
            st.write('Result:')
            bert_doc = hf_ents_to_displacy_format(ner_model(ner_query_input), ignore_entities=["O"])
            labels = list(set([a["label"] for a in bert_doc["ents"]]))
            color_map = add_colormap(labels)
            html = displacy.render(bert_doc, manual=True, style="ent", options={"colors": color_map})
            html = html.replace("\n", " ")
            st.write(WRAPPER.format(html), unsafe_allow_html=True)

    #st.subheader('Speech Synthesis')
    #tts_model = load_tts_model()
    #tts_query_input = st.text_input('tts query:')
    #if tts_query_input != '':
    #    with st.spinner('Predicting...'):
    #        st.audio(tts_model(tts_query_input, lang='ko', speaker='ko'), format='audio/wav')
        

            
    

    


