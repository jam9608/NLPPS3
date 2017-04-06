import re

DictList = []

# read the file and pull out the data into a set
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
        DictList.append(obj)


for item in DictList:
    print(item['id'] + '\t' + item['sentence'] + '\t' + item['polarity'] + '\t' + item['topic'] + '\t' + item['genre'])


