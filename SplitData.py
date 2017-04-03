import re
import os
import nltk
import pickle
from nltk.tag.stanford import StanfordNERTagger

# sets of data pulled from string put here
jsonForm = []

# read the file and pull out the data into a set
with open(r"PS3_training_data.txt") as file:
    lines = file.readlines()
    for line in lines:
        temp = re.split(r'\t', line)
        obj = {}
        obj['id'] = temp[0]
        obj['sentence'] = temp[1]
        obj['polarity'] = temp[2]
        obj['topic'] = temp[3]
        obj['genre'] = temp[4]
        jsonForm.append(obj)

# store all the features
features = []

# configure the named entity recognizer
nltk.internals.config_java(r'C:\Program Files\Java\jdk1.8.0_74\bin\java.exe')
os.environ['JAVAHOME'] = r'C:\Program Files\Java\jdk1.8.0_74\bin\java.exe'
st = StanfordNERTagger(r'C:\Users\Jared Wagner\PycharmProjects\ProblemSet3\english.all.3class.distsim.crf.ser.gz', r'C:\Users\Jared Wagner\PycharmProjects\ProblemSet3\stanford-ner.jar', encoding='utf-8')

# loop each sentence and generate the features
for item in jsonForm:
    # store features here
    featSet = {}
    # initialize the values
    featSet['id'] = item['id']
    featSet['topic'] = item['topic']
    featSet['genre'] = item['genre']
    featSet['polarity'] = item['polarity']
    featSet['adjective_count'] = 0
    featSet['noun_count'] = 0
    featSet['verb_count'] = 0
    featSet['punctuation_count'] = 0
    featSet['number_count'] = 0
    wordCount = 0
    entCount = 0
    # split the sentence into its words
    tokens = nltk.word_tokenize(item['sentence'])
    # generate the features for a word scale
    for word in tokens:
        pos = nltk.pos_tag([word])[0][1]
        if pos == 'JJ' or pos == 'JJR' or pos == 'JJS':
            featSet['adjective_count'] += 1
        if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS':
            featSet['noun_count'] += 1
        if pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ':
            featSet['verb_count'] += 1
        if pos == '.' or pos == ',' or pos == ':' or pos == "''":
            featSet['punctuation_count'] += 1
        if pos == 'CD':
            featSet['number_count'] += 1
        if pos != 'CD' and pos != '.' and pos != ',' and pos != ':' and pos != "''":
            wordCount += 1
    # generate features for the sentence
    featSet['sentence_length'] = len(item['sentence'])
    featSet['start_with_personal_pronoun'] = (nltk.pos_tag([tokens[0]])[0][1] == 'PRP')
    featSet['word_count'] = wordCount
    # check for named entities
    for word, ent in st.tag(tokens):
        if ent != 'O':
            entCount += 1
    featSet['named_entity'] = entCount
    features.append(featSet)
    # display
    print(featSet)

# print to file
with open(r'PS3_training_data_features.bin', 'wb') as file:
    pickle.dump(features, file)

