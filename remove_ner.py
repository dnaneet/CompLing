import streamlit as st
import numpy as np
import pandas as pd

import spacy
import en_core_web_sm
#nlp = en_core_web_sm.load()
nlp = spacy.load("en_core_web_sm")

import spacy_streamlit
from spacy_streamlit import visualize_ner


#USER DEFINED FUNCTIONS


def replace_ner(mytxt):
    clean_text = mytxt
    doc = nlp(mytxt)
    for ent in reversed(doc.ents):
        clean_text = clean_text[:ent.start_char] +ent.label_ + clean_text[ent.end_char:]
    return clean_text

# EMBELLISHMENTS
st.set_page_config(page_title='Text De-identifier',layout='wide')

st.title("Text De-identifier")
st.sidebar.markdown("## This python web app was created by Aneet Narendranath Ph.D.  It may only be used for academic purposes.")


#MAIN

#models = ["en_core_web_sm", "en_core_web_md"]
#default_text = "Sundar Pichai is the CEO of Google."

st.write("## Enter the text you wish to deidentify in the text box below.")
txt = st.text_area('', """ """)
#txt_clean = replace_ner(txt).replace('ORG','ACME Co.')
#st.write("## Deidentified text")
#st.write("You may copy and paste this de-identified text.")
#st.text_area('', txt_clean)
#st.write(txt_clean)
#st.write("\n")
#st.write("## Named entities")
#doc = nlp("Sundar Pichai is the CEO of Google.")
#doc = nlp(txt)
#visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
#spacy_streamlit.visualize(en_core_web_sm, txt)

#eof
