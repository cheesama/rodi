from pororo import Pororo
from spacy import displacy

import streamlit as st
import random

# Set page title
st.title("Rapid pOroro Demo Inferencer")

## text classification

# Load similarity model
@st.cache(allow_output_mutation=True)
def load_similarity_model():
    with st.spinner("Loading similarity model..."):
        similarity_model = Pororo(task="similarity", lang="ko")

        return similarity_model


# Load review_score model
@st.cache(allow_output_mutation=True)
def load_review_score_model():
    with st.spinner("Loading review_score model..."):
        review_score_model = Pororo(task="review", lang="ko")

        return review_score_model


# Load sentiment_analysis model
@st.cache(allow_output_mutation=True)
def load_sentiment_model():
    with st.spinner("Loading sentiment_analysis model..."):
        sentiment_model = Pororo(
            task="sentiment", model="brainbert.base.ko.shopping", lang="ko"
        )

        return sentiment_model


## sequence tagging

# Load machine reading comprehension model
@st.cache(allow_output_mutation=True)
def load_mrc_model():
    with st.spinner("Loading machine reading comprehension model..."):
        mrc_model = Pororo(task="mrc", lang="ko")

        return mrc_model


# Load named entity recognition model
@st.cache(allow_output_mutation=True)
def load_ner_model():
    with st.spinner("Loading sentiment_analysis model..."):
        ner_model = Pororo(task="ner", lang="ko")

        return ner_model


# Load part of speech model
@st.cache(allow_output_mutation=True)
def load_pos_model():
    with st.spinner("Loading POS model..."):
        pos_model = Pororo(task="pos", lang="ko")

        return pos_model


## seq2seq

# Load Paraphrase Identification model
@st.cache(allow_output_mutation=True)
def load_paraphrase_identification_model():
    with st.spinner("Loading paraphrase_identification model..."):
        paws_model = Pororo(task="para", lang="ko")

        return paws_model

# load machine_translation model
@st.cache(allow_output_mutation=True)
def load_machine_translation_model():
    with st.spinner("Loading machine_translation model..."):
        mt = Pororo(task="translation", lang="multi")

        return mt

# load text_summarization model
@st.cache(allow_output_mutation=True)
def load_text_summarization_model():
    with st.spinner("Loading text_summarization model..."):
        summ = Pororo(task="summarization", model="extractive", lang="ko")

        return summ

# Load tts model
@st.cache(allow_output_mutation=True)
def load_tts_model():
    with st.spinner("Loading TTS model..."):
        tts_model = Pororo(task="tts", lang="multi")

        return tts_model

# Load ocr model
@st.cache(allow_output_mutation=True)
def load_ocr_model():
    with st.spinner("Loading OCR model..."):
        ocr_model = Pororo(task="ocr", lang="ko")

        return ocr_model


def format_func(option):
    return CHOICES[option]


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
        # if label not in color_map:
        rand_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        color_map[label] = rand_color

    return color_map


# from https://github.com/explosion/spacy-streamlit/util.py#L26
WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

if __name__ == "__main__":
    st.sidebar.title("Task Type")
    select = st.sidebar.selectbox('Type', 
        [
            'TEXT CLASSIFICATION', 
            'SEQUENCE TAGGING', 
            'SEQ2SEQ', 
            'MISC', 
        ], 
        key='0'
    )

    if select == 'TEXT CLASSIFICATION':
        st.subheader("Semantic Textual Similarity")
        similarity_model = load_similarity_model()
        sim_query1_input = st.text_input("query1:")
        sim_query2_input = st.text_input("query2:")
        if sim_query1_input != "" and sim_query2_input != "":
            with st.spinner("Predicting..."):
                st.write(
                    f"Similarity: {similarity_model(sim_query1_input, sim_query2_input)}"
                )

        st.markdown("""---""")

        st.subheader("Sentiment Analysis")
        sentiment_model = load_sentiment_model()
        sentiment_query_input = st.text_input("sentiment query:")
        if sentiment_query_input != "":
            with st.spinner("Predicting..."):
                st.write("Result:")
                st.json(sentiment_model(sentiment_query_input, show_probs=True))

        st.markdown("""---""")

    elif select == 'SEQUENCE TAGGING':
        ## machine reading comprehension
        st.subheader("Machine Reading Comprehension")
        mrc_model = load_mrc_model()
        mrc_document_input = st.text_area("document content:")
        mrc_query_input = st.text_input("mrc query:")
        if mrc_document_input != "" and mrc_query_input != "":
            with st.spinner("Predicting..."):
                st.write(f"Result: {mrc_model(mrc_query_input, mrc_document_input)[0]}")

        st.markdown("""---""")

        st.subheader("Named Entity Recognition")
        ner_model = load_ner_model()
        ner_query_input = st.text_input("ner query:")
        if ner_query_input != "":
            with st.spinner("Predicting..."):
                st.write("Result:")
                bert_doc = hf_ents_to_displacy_format(
                    ner_model(ner_query_input), ignore_entities=["O"]
                )
                labels = list(set([a["label"] for a in bert_doc["ents"]]))
                color_map = add_colormap(labels)
                html = displacy.render(
                    bert_doc, manual=True, style="ent", options={"colors": color_map}
                )
                html = html.replace("\n", " ")
                st.write(WRAPPER.format(html), unsafe_allow_html=True)

        st.markdown("""---""")

        ## part-of-speech tagging
        st.subheader("Part Of Speech Tagging")
        pos_model = load_pos_model()
        pos_query_input = st.text_input("pos query:")
        if pos_query_input != "":
            with st.spinner("Predicting..."):
                st.write(f"Result: {pos_model(pos_query_input)}")

        st.markdown("""---""")

    elif select == 'SEQ2SEQ':
        ## machine translation
        #st.subheader("Machine Translation")

        # select input language
        #CHOICES = {"ko": "한국어", "en": "영어", "jp": "일본어", "zhi": "중국어"}
        #src_option = st.selectbox(
        #    "입력 언어 선택", options=list(CHOICES.keys()), format_func=format_func
        #)

        # select target language
        #tgt_option = st.selectbox(
        #    "타겟 언어 선택", options=list(CHOICES.keys()), format_func=format_func
        #)

        #input_text = st.text_input("번역 할 문장 입력:")

        #mt_model = load_machine_translation_model()

        #if input_text != "":
        #    with st.spinner("Predicting..."):
        #        st.write(f"result : {mt_model(input_text, src=src_option, tgt=tgt_option)}")

        #st.markdown("""---""")

        ## text summarization
        st.subheader("Text Summarization")
        summ_model = load_text_summarization_model()
        summarization_query = st.text_area('full content: ')
        if summarization_query != "":
            with st.spinner("Predicting..."):
                st.write(f"result : {summ_model(summarization_query)}")

        st.markdown("""---""")

    elif select == 'MISC':
        ## optical character recognition
        st.subheader("Optical Character Recognition")
        ocr_model = load_ocr_model()
        uploaded_file = st.file_uploader("Upload Image file", type=['png','jpg','jpeg'])

        print (uploaded_file)

        if uploaded_file is not None:
            st.json(ocr_model(uploaded_file, detail=True))
        
        st.markdown("""---""")

    
