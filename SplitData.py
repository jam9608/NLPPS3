import re
import os
import nltk
import time
import pickle
import argparse
from nltk.tag.stanford import StanfordNERTagger


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='The input text file, training or test', nargs='+')
parser.add_argument('-o', '--output', type=str, help='The output binary file feature set', nargs='+')
parser.add_argument('-a', '--array', type=str, help='The output binary feature array', nargs='+')
parser.add_argument('-z', '--zip', type=str, help='The zip archive that the stanford POS tagger comes from', nargs='+')
parser.add_argument('-r', '--jar', type=str, help='The jar file that is the stanford POS tagger', nargs='+')
parser.add_argument('-j', '--java', type=str, help='The path to your java executable on the computer', nargs='+')
args = parser.parse_args()


if args.input is None or args.output is None or args.zip is None or args.jar is None or args.java is None or args.array is None:
    print('Please specify all required files')
    exit()

# sets of data pulled from string put here
jsonForm = []

# read the file and pull out the data into a set
with open(' '.join(args.input), 'r') as file:
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
nltk.internals.config_java(' '.join(args.java))
os.environ['JAVAHOME'] = ' '.join(args.java)
st = StanfordNERTagger(' '.join(args.zip), ' '.join(args.jar), encoding='utf-8')

print('all values read from file, now generating features')
start = time.time()
counter = 0

# loop each sentence and generate the features
for item in jsonForm:
    # store features here
    featSet = {}
    # initialize the values
    featSet['id'] = item['id']
    featSet['sentence'] = item['sentence']
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
    counter += 1
    num = counter / (len(jsonForm)/20)
    print('\r0%' + ('=' * int(num)) + ('_' * int(20-num)) + '100% -> ' + str(counter/len(jsonForm)*100) + '%', end='', flush=True)

print('\nfeatures have been generated')
print('time elapsed: ' + str(time.time() - start))

D2 = [[]]
for item in features:
    D2.append([item['id'], item['sentence'], item['topic'], item['genre'], item['polarity'], item['adjective_count'],
    item['noun_count'], item['verb_count'], item['punctuation_count'], item['number_count'], item['sentence_length'],
    int(item['start_with_personal_pronoun']), item['word_count'], item['named_entity']])

# print to file
with open(' '.join(args.output), 'wb') as file:
    pickle.dump(features, file)

# print to file array
with open(' '.join(args.array), 'wb') as file:
    pickle.dump(D2, file)

