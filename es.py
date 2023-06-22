import streamlit as st
import numpy as np
import pandas as pd

#import spacy
#import en_core_web_md
#nlp = en_core_web_sm.load()
#nlp = spacy.load("en_core_web_md")

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text

#import spacy_streamlit
#from spacy_streamlit import visualize_ner


import re

#### Dictionary of meta-discourse markers

PersonMarkers = ["i", "we", "me", "mine", "our", "my", "us", "we", 
   "you", "your", "yours", "your's", "ones", "one's", "their"];
AnnounceGoals = ["purpose", "aim", "intend", "seek", "wish", "argue", 
   "propose", "suggest", "discuss", "like", "focus", "emphasize", 
   "goal", "this", "do", "will"];# contains two instances of "this"
CodeGloss = ["example", "instance", "e.g", "e.g.", "i.e", "i.e.", 
   "namely", "other", "means", "specifically", "known", "such", 
   "define", "call"];
AttitudeMarkers = ["admittedly", "agree", "amazingly", "correctly", 
   "curiously", "disappointing", "disagree", "even", "fortunate", 
   "hope", "hopeful", "hopefully", "important", "interest", "prefer", 
   "must", "ought", "remarkable", "surprise", "surprisingly", 
   "unfortunate", "unfortunately", "unusual", "unusually", 
   "understandably"];
Endophorics = ["see", "note", "noted", "above", "below", "section", 
   "chapter", "discuss", "e.g.", "e.g", "example", "chapter", 
   "figure", "fig", "plot", "chart"];
Hedges = ["almost", "apparent", "apparently", "assume", "assumed", 
   "believe", "believed", "certain", "extent", "level", "amount", 
   "could", "couldnt", "couldn't", "doubt", "essentially", "estimate",
    "frequent", "frequently", "general", "generally", "indicate", 
   "largely", "likely", "mainly", "may", "maybe", "mostly", "might", 
   "often", "perhaps", "possible", "probable", "probably", "relative",
    "seem", "seems", "sometime", "sometimes", "somewhat", "suggest", 
   "suspect", "unlikely", "uncertain", "unclear", "usual", "usually", 
   "would", "wouldnt", "wouldn't", "little", "bit"];
Emphatics = ["actually", "always", "certainly", "certainty", "clear", 
   "clearly", "conclusively", "decidedly", "demonstrate", 
   "determine", "doubtless", "essential", "establish", "indeed", 
   "know", "must", "never", "obvious", "obviously", "prove", "show", 
   "sure", "true", "absolutely", "undoubtedly", "very"];
FrameMarkersStages = ["start", "first", "firstly", "second", 
   "secondly", "third", "thirdly", "fourth", "fourthly", "fifth", 
   "fifthly", "next", "last", "begin", "lastly", "finally", 
   "subsequently", "one", "two", "three", "four", "five"];
Evidentials = ["according", "cite", "cites", "quote", "establish", 
   "established", "said", "say", "says", "argue", "argues", "claim", 
   "claims", "believe", "believes", "suggest", "suggests", "show", 
   "shows", "prove", "proves", "demonstrate", "demonstrates", 
   "literature", "study", "studys", "research"];

####


# EMBELLISHMENTS
st.set_page_config(page_title='Executive Summary Linguistic Signature',layout='wide')

st.title("Executive Summary Linguistic Signature")
st.sidebar.markdown("##### This python web app was created by [Aneet Narendranath Ph.D.](mailto:dnaneet@mtu.edu)  This code is governed under the GPL 3.0 license.")
st.sidebar.write("##### The development of this app was partial supported by the 2022 Michigan Tech MEEM EAB Grant ####")
st.sidebar.markdown("\n")

selection=st.sidebar.radio(label=' ',options=['Meta-discourse analysis', 'AI-interpretation'])
st.sidebar.write("The AI-interpretation page may take several minutes to load.  This uses a large language model (like the one's that power chatGPT) that is several GB in size.")

if selection == "Meta-discourse analysis":
  st.text("meta discourse analysis is performed here.  We recommend that you disable Grammarly for this page as it can distract you from this analysis.")
  st.write("#### Enter the text you wish to count metadiscursive markers into this textbox.")
  txt = st.text_area('', """  """)
  words = strip_multiple_whitespaces(strip_punctuation(txt.lower())).split()
  epsilon = 0.000000000001;
  
  nWords = len(txt.split())
  st.write("Word count: ", nWords)    
    
  n_person_markers = len([i for i in words if i in PersonMarkers])
  st.write("Number of Person Markers: ", n_person_markers, "Percentage: ", np.round(float(n_person_markers/(nWords+epsilon))*100,3))
    
    
  n_announce_goals = len([i for i in words if i in AnnounceGoals])
  st.write("Number of 'Goal announcements': ", n_announce_goals, "Percentage: ", np.round(float(n_announce_goals/(nWords+epsilon))*100,3))

  n_code_gloss = len([i for i in words if i in CodeGloss])
  st.write("Number of 'Code Gloss': ", n_code_gloss, "Percentage: ", np.round(float(n_code_gloss/(nWords+epsilon))*100,3))

  n_att_markers = len([i for i in words if i in AttitudeMarkers])
  st.write("Number of 'Attitude Markers': ", n_att_markers, "Percentage: ", np.round(float(n_att_markers/(nWords+epsilon))*100,3))

  n_endophorics = len([i for i in words if i in Endophorics])
  st.write("Number of 'Endophorics': ", n_endophorics, "Percentage: ", np.round(float(n_endophorics/(nWords+epsilon))*100,3))

  n_hedges = len([i for i in words if i in Hedges])
  st.write("Number of 'Hedges': ", n_hedges, "Percentage: ", np.round(float(n_hedges/(nWords+epsilon))*100,3))

  n_emphatics = len([i for i in words if i in Emphatics])
  st.write("Number of 'Emphatics': ", n_emphatics, "Percentage: ", np.round(float(n_emphatics/(nWords+epsilon))*100,3))

  n_frm_markers_stgs = len([i for i in words if i in FrameMarkersStages])
  st.write("Number of 'Frame Markers': ", n_frm_markers_stgs, "Percentage: ", np.round(float(n_frm_markers_stgs/(nWords+epsilon))*100,3))

  n_evidentials = len([i for i in words if i in Evidentials])
  st.write("Number of 'Evidentials': ", n_evidentials, "Percentage: ", np.round(float(n_evidentials/(nWords+epsilon))*100,3))      


elif selection == "AI-interpretation":
  from transformers import pipeline
  from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
  pipe = pipeline(model = "typeform/distilbert-base-uncased-mnli")
  tokenizer = AutoTokenizer.from_pretrained("typeform/distilbert-base-uncased-mnli")
  st.write("Replace the placedholder text below with your executive summary paragraph.  Enter only one paragraph.  The AI large language    model will analyze the paragraph and report its interpretation of the content of the paragraph via probabilities. **Once you have entered the paragraph in the textbox, please click the mouse pointer outside the box for analysis to begin and results to be reported.**")
  try:
      txt = st.text_area('Text to analyze', ''' This is placeholder text. ''')
      probs = pipe(txt,
                   candidate_labels=["problem",
                          "research", 
                          "uses",
                          "requirements",
                          "deliverables"],
                  )
      st.write(probs["labels"][0], np.round(100* probs["scores"][0],1))
      st.write(probs["labels"][1], np.round(100* probs["scores"][1],1))
      st.write(probs["labels"][2], np.round(100* probs["scores"][2],1))
      st.write(probs["labels"][3], np.round(100* probs["scores"][3],1))
      st.write(probs["labels"][4], np.round(100* probs["scores"][4],1))
  except ValueError:
      st.markdown("## You cannot leave the textbox empty!")      
