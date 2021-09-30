import nltk
lemmatizer = nltk.stem.WordNetLemmatizer()
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import json
import io
import random
import string
import warnings
warnings.filterwarnings('ignore')
import joblib
import pandas as pd
from pprint import pprint
from decision_tree_functions import decision_tree_algorithm, decision_tree_predictions
from helper_functions import train_test_split, calculate_accuracy
import re

# load the model from disk
filename = 'Trained_Model.sav' 
forest = joblib.load(filename)

# load test dataset
dtest = pd.read_csv("./datasets/Testing_1.csv")
dtest["label"] = dtest.prognosis
dtest = dtest.drop("prognosis", axis=1)

column_names = []
for column in dtest.columns:
    name = column.replace("_", " ")
    column_names.append(name)

column_names.remove('label')

bag_of_words = list()
for word in column_names:
    word = nltk.word_tokenize(word)
    bag_of_words.append(word)

column_names = []
for column in dtest.columns:
    name = column.replace(" ", "_")
    column_names.append(name)
dtest.columns = column_names
random.seed(0)
test_df =  dtest

user_provided = pd.DataFrame()
for name in range(len(column_names)):
    user_provided[column_names[name]] = 0
user_provided = user_provided.append(pd.Series('0', index=user_provided.columns) ,ignore_index=True)

intents = json.loads(open('./intents/intents.json').read())
DiseaseWithSymptoms = pd.read_csv("./datasets/dataset.csv")
DiseaseDesc = pd.read_csv("./datasets/symptom_Description.csv")
DiseasePrecautions = pd.read_csv("./datasets/symptom_precaution.csv")

def extract_keywords(sent):
    grammar = r"""
    NBAR:
        # Nouns and Adjectives, terminated with Nouns
        {<NN.*>*<NN.*>|<JJ>*<NN>|<NN>*<JJ>|<NN><VB.*><JJ>|<VBG>|<VBN>}
    NP:
        {<NBAR>}
        # Above, connected with in/of/etc...
        {<NBAR><IN><NBAR>}
    """
    chunker = nltk.RegexpParser(grammar)
    ne = set()
    chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.add(' '.join([child[0] for child in tree.leaves()]))
    return ne

desc_INPUTS = ("what", "tell", "about", "symptom","symptoms")
def check_if_desc(sentence):
    for word in sentence.split():
        if word.lower() in desc_INPUTS:
            return True
        else: return False
            
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    sentence_words = TreebankWordDetokenizer().detokenize(sentence_words)  # ['the', 'quick', 'brown'] => 'The quick brown'
    result = list()
    Keywords = extract_keywords(sentence_words)
    asking_desc = check_if_desc(sentence_words)
    result.insert(0,Keywords)
    result.insert(1,asking_desc)
    # print('keywords found: ',Keywords,' asking_desc: ',asking_desc)
    return result 

def Mark_the_msg(sentence):
    # get keyword and question etc
    cleaned_sentence = clean_up_sentence(sentence)
    keywords = cleaned_sentence[0]
    asking_desc = cleaned_sentence[1]
    symptoms_found = list()
    results = list()
    if asking_desc:
        for words in keywords:
            for i,disse in enumerate(DiseaseDesc['Disease'].tolist()):
                if words.lower() == disse.lower(): 
                    results.append(DiseaseDesc.iloc[i][1])
                    results.append(True)
                    results.append(False)
                    return results
    if len(keywords)>0:
        for ww in keywords:
            for ss in bag_of_words:
                if ww in ss:
                    symptom = ''
                    for s in ss:
                        if symptom == '':
                            symptom = s
                        else:
                            symptom = symptom+'_'+s
                    user_provided.at[0,symptom]= '1'
                    # print('symptom in markmsg: ',symptom)
                    symptoms_found.append(symptom)
    results.append(symptoms_found)
    results.append(False)
    results.append(True)
    return results
        
 
def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions
    # print("\ndf_predictions in RFP function\n\n")
    # print(df_predictions,'\n')
    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    return random_forest_predictions
    

def predict_class(sentence):
    for (k, v) in intents.items():
        for vv in v:
            tag = ''
            pattern = list()
            response = list()
            for (key,value) in vv.items():
                if key == 'tag':
                    tag = value
                if key == 'patterns':
                    pattern = value
                if key == 'responses':
                    response = value
            pattern = map(lambda x:x.lower(),pattern)
            # print('tag: ',tag,'\npattern: ',pattern,'\nresponses: ',response)
            for word in sentence.split():
                if word.lower() in pattern:
                    res = list()
                    res.append(tag)
                    res.append(random.choice(response))
                    return res
    
    result = Mark_the_msg(sentence)
    if (result[1] == False and result[2] == True):
        diseases = str(random_forest_predictions(user_provided,forest))
        diseases = diseases.split()
        diseases.remove('0')
        diseases.remove('0,')
        diseases.remove('Name:')
        diseases.remove('dtype:')
        diseases.remove('object')
        disease = ''
        for dd in diseases:
            disease = disease+' '+dd
        res = list()
        res.append('According to the symptoms recieved, You may have'+disease)
        return (res)
    else: 
        # print(result[0])
        res = list()
        res.append(result[0])
        return (res)
 

def chatbot_response(msg):      
    res = predict_class(msg)
    # print(res,' len: ',len(res))
    if len(res)>1:
        if res[0] == 'goodbye':
            user_provided.drop(0, inplace=True)
            user_provided.append(pd.Series('0', index=user_provided.columns) ,ignore_index=True)
        res = res[1]
    else:
        res = res[0]
    # print(res)
    return res

# msg = input('user=> ')
# while True:
#     print(chatbot_response(msg))
#     msg = input('user=> ')

#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        # print(res)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("MedBot")
base.geometry("450x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=440,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=439)
EntryBox.place(x=128, y=401, height=90, width=320)
SendButton.place(x=6, y=401, height=90)

base.mainloop()