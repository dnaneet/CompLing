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
st.sidebar.write("## This python web app was created by Aneet Narendranath Ph.D.  This code is governed under the GPL 3.0 license.")
st.sidebar.markdown("\n")
st.sidebar.markdown("## The de-identification process uses [standard Python libraries](https://spacy.io/api/entityrecognizer).")


#MAIN

#models = ["en_core_web_sm", "en_core_web_md"]

st.write("### Enter the text you wish to deidentify in the text box below and click 'ctrl + enter.'")
txt = st.text_area('', """  """)
txt_clean = replace_ner(txt).replace('ORG','ACME Co.')
st.write("### Deidentified text")
st.write("You may copy and paste this de-identified text once it appears.")
st.write(txt_clean)
#st.text_area('', txt_clean)
#st.write(txt_clean)
#st.write("\n")
#st.write("## Named entities")

        
doc = nlp(txt)
visualize_ner(doc, labels=nlp.get_pipe("ner").labels)

st.write("The following entity libraries may be selected")


st.write("""
PERSON - People, including fictional.
NORP - Nationalities or religious or political groups. 
FAC - Buildings, airports, highways, bridges, etc.  
ORG - Companies, agencies, institutions, etc.  
GPE - Countries, cities, states.  
LOC - Non-GPE locations, mountain ranges, bodies of water.  
PRODUCT - Objects, vehicles, foods, etc. (Not services.)  
EVENT - Named hurricanes, battles, wars, sports events, etc. 
WORK_OF_ART - Titles of books, songs, etc.  
LAW - Named documents made into laws.  
LANGUAGE - Any named language.  
DATE - Absolute or relative dates or periods.  
TIME - Times smaller than a day.  
PERCENT - Percentage, including %.  
MONEY - Monetary values, including unit.  
QUANTITY - Measurements, as of weight or distance.  
ORDINAL - first, second, etc.  
CARDINAL - Numerals that do not fall under another type.""")

#spacy_streamlit.visualize(en_core_web_sm, txt)

#eof
