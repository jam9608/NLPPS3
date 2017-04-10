import pickle
import numpy
import re
from sklearn import svm

with open(r'PS3_training_data_features.bin', 'rb') as file:
     features = pickle.load(file)

###################################WORD COUNT CODE#############################
i = 0
with open(r"PS3_training_data.txt") as file:
    training = file.readlines()

trainingData = ["" for x in range(2564)]
trainingTargetGenre = ["" for x in range(2564)]
trainingTargetPositivity = ["" for x in range(2564)]
trainingTargetEmotion = ["" for x in range(2564)]

for line in training:
    ##Line sentance Pos/neg event genre \n##
    data = re.split(r'\t', line)
    #dataset[i] = data[2:-1]
    #print(data[1])
    trainingData[i] = data[1]
    trainingTargetGenre[i] = data[4]
    trainingTargetPositivity[i] = data[2]
    trainingTargetEmotion[i] = data[3]
    
    i+=1

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainingData)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

from sklearn import svm
C=1.0
svc1 = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTargetGenre)
svc2 = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTargetPositivity)
svc3 = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTargetEmotion)
predicted1 = svc1.decision_function(X_train_tf)
predicted2 = svc2.decision_function(X_train_tf)
print(predicted2.shape)
predicted3 = svc3.decision_function(X_train_tf)

##############################################################################


D2 = []
D3 = []
D4 = []
##print(features)

i = 0
for item in features:
    newfeaturelist = numpy.array([item['adjective_count'], item['noun_count'], item['verb_count'], item['punctuation_count'], item['number_count'], item['sentence_length'], item['start_with_personal_pronoun'], item['word_count'], item['named_entity']])
    print(newfeaturelist.shape)
    newfeaturelist1 = numpy.concatenate((newfeaturelist, [predicted1[i]]),axis=0)
    D2.append(newfeaturelist1) 
    newfeaturelist2 = numpy.concatenate((newfeaturelist, predicted2[i]),axis=0)
    D3.append(newfeaturelist2)
    newfeaturelist3 = numpy.concatenate((newfeaturelist, predicted3[i]),axis=0)
    D4.append(newfeaturelist3)
#    D4.append(newfeaturelist + predicted3[i])
    i+=1
npD2= numpy.array(D2)
print(npD2.shape)
npD3 = numpy.array(D3)
print(npD3.shape)

npD4 = numpy.array(D4)
print(npD4.shape)

trainingFeatures = D2[:2000]
testFeatures = D2[2000:]

i = 0
with open(r"PS3_training_data.txt") as file:
    training = file.readlines()

#dataset = numpy.zeros((2564,3),numpy.str)
trainingData = ["" for x in range(2000)]

testData = ["" for x in range(564)]
trainingTarget = ["" for x in range(2000)]

testTarget = ["" for x in range(564)]
for line in training[:2000]:
    ##Line sentance Pos/neg event genre \n##
    data = re.split(r'\t', line)
    #dataset[i] = data[2:-1]
    #print(data[1])
    trainingData[i] = data[1]
    trainingTarget[i] = data[2]
    
    i+=1
    
i=0
for line in training[2000:]:
    data = re.split(r'\t', line)
    #dataset[i] = data[2:-1]
    #print(data[1])
    testData[i] = data[1]
    testTarget[i] = data[2]

    i+=1
    
trainGenre = trainingTargetGenre[:2000]
testGenre = trainingTargetGenre[2000:]
nptraining = numpy.array(trainingFeatures)
nptarget = numpy.array(trainGenre)
nptest = numpy.array(testFeatures)
nptestTarget = numpy.array(testGenre)

trainPos =numpy.array(trainingTargetPositivity[:2000])
testPos = numpy.array(trainingTargetPositivity[2000:])
nptraining2 = numpy.array(D3[:2000])
nptest2 = numpy.array(D3[2000:])

trainEm =numpy.array(trainingTargetEmotion[:2000])
testEm = numpy.array(trainingTargetEmotion[2000:])
nptraining3 = numpy.array(D4[:2000])
nptest3 = numpy.array(D4[2000:])

print(nptraining.shape)
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(nptraining, nptarget)
predicted = svc.predict(nptest)
print(numpy.mean(predicted == nptestTarget))
##doing some feature selection


svc = svm.SVC(kernel='linear', C=C).fit(nptraining2, trainPos)
predicted = svc.predict(nptest2)
print(numpy.mean(predicted == testPos))

svc = svm.SVC(kernel='linear', C=C).fit(nptraining3, trainEm)
predicted = svc.predict(nptest3)
print(numpy.mean(predicted == testEm))
