import re
import nltk
import pickle
import math


# The sentences grouped on genre
GenreA = []
GenreB = []

# The sentences of genre b grouped on polarity
PosB = []
NegB = []

# The sentences of genre a grouped on polarity
PosA = []
NegA = []
NeuA = []

# The sentences of group a grouped on topic
Event = []
Communication = []
Going = []
Legal = []
Money = []
Outdoor = []
Personal = []
Pain = []

with open("../PS3_training_data.txt", "r", encoding="utf8") as file:
    lines = file.readlines()
    for line in lines:
        temp = re.split(r'\t', line)
        obj = {}
        obj['id'] = temp[0]
        obj['sentence'] = temp[1]
        obj['polarity'] = temp[2]
        obj['topic'] = temp[3]
        obj['genre'] = temp[4]

        if obj['genre'] == 'GENRE_A':
            GenreA.append(obj)
        else:
            GenreB.append(obj)

for item in GenreB:
    if item['polarity'] == 'POSITIVE':
        PosB.append(item)
    else:
        NegB.append(item)

for item in GenreA:
    if item['polarity'] == 'POSITIVE':
        PosA.append(item)
    elif item['polarity'] == 'NEGATIVE':
        NegA.append(item)
    else:
        NeuA.append(item)

    if item['topic'] == 'ATTENDING_EVENT':
        Event.append(item)
    elif item['topic'] == 'COMMUNICATION_ISSUE':
        Communication.append(item)
    elif item['topic'] == 'GOING_TO_PLACES':
        Going.append(item)
    elif item['topic'] == 'LEGAL_ISSUE':
        Legal.append(item)
    elif item['topic'] == 'MONEY_ISSUE':
        Money.append(item)
    elif item['topic'] == 'OUTDOOR_ACTIVITY':
        Outdoor.append(item)
    elif item['topic'] == 'PERSONAL_CARE':
        Personal.append(item)
    elif item['topic'] == '(FEAR_OF)_PHYSICAL_PAIN':
        Pain.append(item)


GenreA_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in GenreA]).lower()))
GenreB_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in GenreB]).lower()))

# The sentences of genre b grouped on polarity
PosB_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in PosB]).lower()))
NegB_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in NegB]).lower()))

# The sentences of genre a grouped on polarity
PosA_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in PosA]).lower()))
NegA_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in NegA]).lower()))
NeuA_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in NeuA]).lower()))

# The sentences of group a grouped on topic
Event_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in Event]).lower()))
Communication_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in Communication]).lower()))
Going_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in Going]).lower()))
Legal_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in Legal]).lower()))
Money_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in Money]).lower()))
Outdoor_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in Outdoor]).lower()))
Personal_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in Personal]).lower()))
Pain_dist = nltk.FreqDist(nltk.word_tokenize(" ".join([x['sentence'] for x in Pain]).lower()))

def getValue(tuple):
    return tuple[0]

dists = [GenreA_dist, GenreB_dist]

def getLikely(dists):
    '''
    words = set()
    
    
    for dist in dists:
    for key in dist.keys():
        words.add(key)
    '''
    
    for dist in range(len(dists)):
        ordered_words = []
        
        for word in dists[dist].keys():
            in_dist = float(dists[dist][word])
            other_dists = 0.0
            occurrences = 0.0
            for x in range(len(dists)):
                if word in dists[x]:
                    occurrences += dists[x][word]
                    if x == dist:
                        continue
                    other_dists += dists[x][word]
            # Remove hapaxes
            if occurrences == 1:
                continue
            ll = math.log(((in_dist/occurrences)+0.001)/((other_dists/occurrences)+0.001))
            ordered_words.append((ll, word))
            ordered_words = sorted(ordered_words, key=getValue, reverse=True)
        for x in range(5):
            print(ordered_words[x])
        print()

print("Log likelihoods for genre A and genre B")
getLikely(dists)

print("Log likelihoods for genre B positive and negative")
dists = [PosB_dist, NegB_dist]
getLikely(dists)

print("Log likelihoods for Genre A positive, negative, and neutral")
dists = [PosA_dist, NegA_dist, NeuA_dist]
getLikely(dists)

print("Log likelihoods for events, communication, going, legal, money, outdoor, personal, and pain")
dists = [Event_dist, Communication_dist, Going_dist, Legal_dist, Money_dist, Outdoor_dist, Personal_dist, Pain_dist]
getLikely(dists)