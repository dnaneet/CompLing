hasimport streamlit as st
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
Logicals = ["but", "since", "however", "because", "also", "yet", "therefore", "thereby"]

####


def authorial_stance(text):
  epsilon = 0.000000000001;
  words = strip_multiple_whitespaces(strip_punctuation(text.lower())).split()
  pc_person_markers = np.round(float(len([i for i in words if i in PersonMarkers])/(len(text.split())+epsilon))*100,3)
  pc_announce_goals = np.round(float(len([i for i in words if i in AnnounceGoals])/(len(text.split())+epsilon))*100,3)
  pc_code_gloss = np.round(float(len([i for i in words if i in CodeGloss])/(len(text.split())+epsilon))*100,3)
  pc_att_markers = np.round(float(len([i for i in words if i in AttitudeMarkers])/(len(text.split())+epsilon))*100,3)
  pc_endophorics = np.round(float(len([i for i in words if i in Endophorics])/(len(text.split())+epsilon))*100,3)
  pc_hedges = np.round(float(len([i for i in words if i in Hedges])/(len(text.split())+epsilon))*100,3)
  pc_emphatics = np.round(float(len([i for i in words if i in Emphatics])/(len(text.split())+epsilon))*100,3)
  pc_frame_markers_stages = np.round(float(len([i for i in words if i in FrameMarkersStages])/(len(text.split())+epsilon))*100,3)
  pc_evidentials = np.round(float(len([i for i in words if i in Evidentials])/(len(text.split())+epsilon))*100,3)
  pc_logicals = np.round(float(len([i for i in words if i in Logicals])/(len(text.split())+epsilon))*100,3) 

  pc_interactional = np.array([pc_person_markers, pc_hedges, pc_emphatics, pc_att_markers])
  pc_interactive = np.array([pc_code_gloss, pc_endophorics, pc_evidentials, pc_frame_markers_stages, pc_logicals])

  interactional = np.sum(pc_interactional, axis=0)
  interactive = np.sum(pc_interactive, axis=0)

  return interactional, interactive


# EMBELLISHMENTS
st.set_page_config(page_title='Executive Summary Linguistic Signature',layout='wide')

st.title("Executive Summary Linguistic Signature")
st.sidebar.markdown("##### This python web app was created by [Aneet Narendranath Ph.D.](mailto:dnaneet@mtu.edu)  This code is governed under the GPL 3.0 license.")
st.sidebar.write("##### The development of this app was partial supported by the 2022 Michigan Tech MEEM EAB Grant ####")
st.sidebar.markdown("\n")

selection=st.sidebar.radio(label=' ',options=['Meta-discourse analysis', 'AI-interpretation'])
st.sidebar.write("The AI-interpretation page may take several minutes to load.  This uses a large language model (like the one's that power chatGPT) that is several GB in size.")

if selection == "Meta-discourse analysis":
  st.write("meta discourse analysis is performed here.  We recommend that you disable Grammarly (if you have it) for this page as it can distract you from this analysis.")

  with st.expander("See explanation"): 
   st.write("Discourse is written or spoken communication.  Metadiscourse is the _discourse_ about discourse.  Each genre of discourse has a specific metadiscursive signature. For _example_, abstracts to scientific articles have a different metadiscursive signature than newspaper articles.  This is because the authors are saying things differently or taking a stance that is unique or particular to that type of writing or speech.  There are two major classes of metadisourse markers, viz., interactional metadiscourse and interactive metadiscourse.  International metadiscourse are those markers employed by the author or the speaker to interact with the reader or the audience and attempt to put them in his/her shoes.  Interactive metadiscourse are those markers employed by the author or speaker to help the reader interact with the content.  Some examples of interactional metadiscourse are: I, me, we, you, us, amazingly, hopefully, curiously, doubt, likely, maybe, etc.  Some examples of interactive metadiscourse are: firstly, secondly, next, then, according [to], but, since, claim [to], etc.")
   st.write("Executive summaries are a unique genre of engineering communication.  They are structured around paragraphs, each of which have an intent and a desired impact.  _In addition_, each paragraph has a metadiscursive signature that can be expressed as a percentage of interactional and interactive markers.  Such a signature is computed by collecting many examples of a genre of communication and _then_ counting the metadiscourse markers within.")
   
  st.write("#### Enter the text you wish to count metadiscursive markers into this textbox.")
  txt = st.text_area('', """  """)
  interactional, interactive = authorial_stance(txt)
  st.write("interactional: ", interactional)
  st.write("interactive: ", interactive) 
  
  
  df = pd.read_csv("mdm_pud_keywordsDataset_2.csv")
  option_transcript_class = st.selectbox(
    'How would you like to be contacted?',
    list(np.unique(df.transcript_class)))
  st.table(df[df.transcript_class == option_transcript_class][["interactive", "interactional"]].describe() )


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
