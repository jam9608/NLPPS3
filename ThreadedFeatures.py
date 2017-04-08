import re
import os
import nltk
import time
import pickle
import argparse
import threading
from nltk.tag.stanford import StanfordNERTagger


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='The input text file, training or test', nargs='+')
parser.add_argument('-o', '--output', type=str, help='The output binary file feature set', nargs='+')
parser.add_argument('-a', '--array', type=str, help='The output binary feature array', nargs='+')
args = parser.parse_args()


if args.input is None or args.output is None or args.array is None:
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


# method for feature generation
def worker(lb, ub):
    global jsonForm
    global features
    print('Thread started running on sentences ' + str(lb) + ' to ' + str(ub))
    # loop each sentence and generate the features
    for item in jsonForm[lb:ub]:
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
        # generate the features for a word scale
        tags = nltk.pos_tag(item['sentence'].split())
        '''
        nes = nltk.ne_chunk(tags, binary=True)
        for tree in nes.subtrees():
            if tree.label() == 'NE':
                entCount += 1
        '''
        for word, pos in tags:
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
        featSet['start_with_personal_pronoun'] = (tags[0][1] == 'PRP')
        featSet['word_count'] = wordCount
        featSet['named_entity'] = entCount
        features.append(featSet)
    print('Thread finished running on sentences ' + str(lb) + ' to ' + str(ub))


# begin features generate
print('all values read from file, now generating features')
start = time.time()
threads = []
count = 2
breakcount = int(len(jsonForm)/count)
# create threads for feature generation
for i in range(0, count):
    low = i * breakcount
    high = (i+1) * breakcount
    t = threading.Thread(target=worker, args=(low, high), name='POS worker thread: '+str(i))
    threads.append(t)
# get the remaining lines
low = high
high = len(jsonForm)
t = threading.Thread(target=worker, args=(low, high), name='POS worker thread: remainder')
threads.append(t)
print('Created threads, now running')
for t in threads:
    t.setDaemon(True)
    t.start()

# features generated
for t in threads:
    t.join()
print('\nall features have been generated')
print('time elapsed: ' + str(time.time() - start))

# make set an array
D2 = [[]]
for item in features:
    D2.append([item['id'], item['sentence'], item['topic'], item['genre'], item['polarity'], item['adjective_count'],
    item['noun_count'], item['verb_count'], item['punctuation_count'], item['number_count'], item['sentence_length'],
    int(item['start_with_personal_pronoun']), item['word_count'], item['named_entity']])

# print set to file
with open(' '.join(args.output), 'wb') as file:
    pickle.dump(features, file)

# print array file array
with open(' '.join(args.array), 'wb') as file:
    pickle.dump(D2, file)

