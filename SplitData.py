import re
import os
import nltk
import pickle
import argparse
from nltk.tag.stanford import StanfordNERTagger


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='The input text file, training or test', nargs='+')
parser.add_argument('-o', '--output', type=str, help='The output binary file, training or test', nargs='+')
args = parser.parse_args()


if args.input is None or args.output is None:
    print('Please specify input text file and output binary file')
    exit()

# sets of data pulled from string put here
jsonForm = []

# read the file and pull out the data into a set
with open(str(args.input), 'r') as file:
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
with open(args.output, 'wb') as file:
    pickle.dump(features, file)

