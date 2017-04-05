import re
import numpy as np
from nltk import FreqDist


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


def getCounts(dist):
    here = []
    for word in dist:
        math = (dist[word] + 1) / (AllDist[word] + AllDist.N())
        add = (word, np.abs(np.log(math)))
        here.append(add)
    return here


# Get all the lines and put in json form
with open(r"C:\Users\Jared Wagner\PycharmProjects\ProblemSet3\PS3_training_data.txt") as file:
    lines = file.readlines()
    for line in lines:
        temp = re.split(r'\t', line)
        obj = {}
        obj['id'] = temp[0]
        obj['sentence'] = temp[1]
        obj['polarity'] = temp[2]
        obj['topic'] = temp[3]
        obj['genre'] = temp[4]


        # sort on genre
        if obj['genre'] == 'GENRE_A':
            GenreA.append(obj)
        else:
            GenreB.append(obj)


# sort genre b for polarity
for item in GenreB:
    if item['polarity'] == 'POSITIVE':
        PosB.append(item)
    else:
        NegB.append(item)


# sort genre a for polarity and topic
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


# concat all sentences to strings, create distribution
AllString = ''
for item in GenreA:
    AllString += item['sentence'] + ' '
for item in GenreB:
    AllString += item['sentence'] + ' '
AllDist = FreqDist(AllString.split())





# concat all genre sentences to strings, create distribution and create frequencies
GenreAStrings = ''
for item in GenreA:
    GenreAStrings += item['sentence'] + ' '
ADist = FreqDist(GenreAStrings.split())
AList = sorted(getCounts(ADist), key=lambda tup: tup[1])
print('top 5 Genre A words')
print(AList[-5:])

GenreBStrings = ''
for item in GenreB:
    GenreBStrings += item['sentence'] + ' '
BDist = FreqDist(GenreBStrings.split())
BList = sorted(getCounts(BDist), key=lambda tup: tup[1])
print('top 5 Genre B words')
print(BList[-5:])





# concat all polarity sentences to strings, create distribution and create frequencies
PolarityPositiveStrings = ''
for item in PosA:
    PolarityPositiveStrings += item['sentence'] + ' '
for item in PosB:
    PolarityPositiveStrings += item['sentence'] + ' '
PosDist = FreqDist(PolarityPositiveStrings.split())
PosList = sorted(getCounts(PosDist), key=lambda tup: tup[1])
print('top 5 Polarity Pos words')
print(PosList[-5:])

PolarityNeutralStrings = ''
for item in NeuA:
    PolarityNeutralStrings += item['sentence'] + ' '
NeuDist = FreqDist(PolarityNeutralStrings.split())
NeuList = sorted(getCounts(NeuDist), key=lambda tup: tup[1])
print('top 5 Polarity Neu words')
print(NeuList[-5:])

PolarityNegativeStrings = ''
for item in NegA:
    PolarityNegativeStrings += item['sentence'] + ' '
for item in NegB:
    PolarityNegativeStrings += item['sentence'] + ' '
NegDist = FreqDist(PolarityNegativeStrings.split())
NegList = sorted(getCounts(NegDist), key=lambda tup: tup[1])
print('top 5 Polarity Neg words')
print(NegList[-5:])





# concat all topic sentences to strings, create distribution and create frequencies
TypeEventStrings = ''
for item in Event:
    TypeEventStrings += item['sentence'] + ' '
EventDist = FreqDist(TypeEventStrings.split())
EventList = sorted(getCounts(EventDist), key=lambda tup: tup[1])
print('top 5 Topic Attending Event words')
print(EventList[-5:])

TypeCommunicationStrings = ''
for item in Communication:
    TypeCommunicationStrings += item['sentence'] + ' '
CommDist = FreqDist(TypeCommunicationStrings.split())
CommList = sorted(getCounts(CommDist), key=lambda tup: tup[1])
print('top 5 Communication Issue words')
print(CommList[-5:])

TypePlacesStrings = ''
for item in Going:
    TypePlacesStrings += item['sentence'] + ' '
PlaceDist = FreqDist(TypePlacesStrings.split())
PlaceList = sorted(getCounts(PlaceDist), key=lambda tup: tup[1])
print('top 5 Going to Places words')
print(PlaceList[-5:])

TypeLegalStrings = ''
for item in Legal:
    TypeLegalStrings += item['sentence'] + ' '
LegalDist = FreqDist(TypeLegalStrings.split())
LegalList = sorted(getCounts(LegalDist), key=lambda tup: tup[1])
print('top 5 Legal Issue words')
print(LegalList[-5:])

TypeMoneyStrings = ''
for item in Money:
    TypeMoneyStrings += item['sentence'] + ' '
MoneyDist = FreqDist(TypeMoneyStrings.split())
MoneyList = sorted(getCounts(MoneyDist), key=lambda tup: tup[1])
print('top 5 Money Issue words')
print(MoneyList[-5:])

TypeOutdoorStrings = ''
for item in Outdoor:
    TypeOutdoorStrings += item['sentence'] + ' '
OutdoorDist = FreqDist(TypeOutdoorStrings.split())
OutdoorList = sorted(getCounts(OutdoorDist), key=lambda tup: tup[1])
print('top 5 Outdoor Activity words')
print(OutdoorList[-5:])

TypePersonalStrings = ''
for item in Personal:
    TypePersonalStrings += item['sentence'] + ' '
PersonalDist = FreqDist(TypePersonalStrings.split())
PersonalList = sorted(getCounts(PersonalDist), key=lambda tup: tup[1])
print('top 5 Personal Care words')
print(PersonalList[-5:])

TypePainStrings = ''
for item in Pain:
    TypePainStrings += item['sentence'] + ' '
PainDist = FreqDist(TypePainStrings.split())
PainList = sorted(getCounts(PainDist), key=lambda tup: tup[1])
print('top 5 (Fear_of_)Physical_Pain words')
print(PainList[-5:])

