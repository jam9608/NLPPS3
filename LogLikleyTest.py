import re
import nltk
import pickle


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


