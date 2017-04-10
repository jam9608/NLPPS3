import pickle
import argparse
import numpy
from sklearn import svm
from Metrics import print_metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='The input binary file feature array', nargs='+')
parser.add_argument('-o', '--output', type=str, help='The output text file for results', nargs='+')
args = parser.parse_args()


########################################################################################################################
# pull in data

#if args.input is None or args.output is None:
#    print('Please specify all required files')
#    exit()

testfeatures = []
with open(str(args.input[0]), 'rb') as file:
    testfeatures = pickle.load(file)

trainfeatures = []
with open(r'OUTPUT_BINARY_FEATURE_ARRAY_TRAINING.bin', 'rb') as file:
    trainfeatures = pickle.load(file)

# feature array syntax:
# 0:id, 1:sentence, 2:topic, 3:genre, 4:polarity, 5:adjective_count, 6:noun_count, 7:verb_count, 8:punctuation_count,
# 9:number_count, 10:sentence_length, 11:start_with_personal_pronoun, 12:word_count, 13:named_entity
# Metrics Usage:
    
###################################WORD COUNT CODE#############################
i = 0
trainingData = ["" for x in range(len(trainfeatures))]
trainingTargetGenre = ["" for x in range(len(trainfeatures))]
trainingTargetPositivity = ["" for x in range(len(trainfeatures))]
trainingTargetEmotion = ["" for x in range(len(trainfeatures))]

for item in trainfeatures:
    trainingData[i] = item[1]
    trainingTargetGenre[i] = item[3]
    trainingTargetPositivity[i] = item[4]
    trainingTargetEmotion[i] = item[2]
    
    i+=1

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainingData)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

C=1.0
svc1 = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTargetGenre)
svc2 = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTargetPositivity)
svc3 = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTargetEmotion)
predicted1 = svc1.decision_function(X_train_tf)
predicted2 = svc2.decision_function(X_train_tf)
predicted3 = svc3.decision_function(X_train_tf)

##############################################################################

##arrayify our featuresets
D2 = []
D3 = []
D4 = []

i = 0
for item in trainfeatures:
    newfeaturelist = numpy.array(item[5:])
    newfeaturelist1 = numpy.concatenate((newfeaturelist, [predicted1[i]]),axis=0)
    D2.append(newfeaturelist1) 
    newfeaturelist2 = numpy.concatenate((newfeaturelist, predicted2[i]),axis=0)
    D3.append(newfeaturelist2)
    newfeaturelist3 = numpy.concatenate((newfeaturelist, predicted3[i]),axis=0)
    D4.append(newfeaturelist3)
    i+=1
    
    
GenreTarget = numpy.array(trainingTargetGenre)
GenreFeatureset = numpy.array(D2)
PosTarget = numpy.array(trainingTargetPositivity)
PosFeatureset = numpy.array(D3)
EmTarget = numpy.array(trainingTargetEmotion)
EmFeatureset = numpy.array(D4)

GenreClassifier = svm.SVC(kernel='linear', C=C).fit(GenreFeatureset, GenreTarget)
PolarityClassifier = svm.SVC(kernel='linear', C=C).fit(PosFeatureset, PosTarget)
TopicClassifier = svm.SVC(kernel='linear', C=C).fit(EmFeatureset, EmTarget)

##classifiers complete, lets get to our test data
i = 0
testData = ["" for x in range(len(testfeatures))]
for item in testfeatures:
    testData[i] = item[1]
    i+=1

count_vect = CountVectorizer()
Y_train_counts = count_vect.fit_transform(testData)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(Y_train_counts)
Y_train_tf = tf_transformer.transform(Y_train_counts)

out1 = svc1.decision_function(Y_train_tf)
out2 = svc2.decision_function(Y_train_tf)
out3 = svc3.decision_function(Y_train_tf)

D5 = []
D6 = []
D7 = []

i = 0
for item in testfeatures:
    newfeaturelist = numpy.array(item[5:])
    newfeaturelist1 = numpy.concatenate((newfeaturelist, [out1[i]]),axis=0)
    D5.append(newfeaturelist1) 
    newfeaturelist2 = numpy.concatenate((newfeaturelist, out2[i]),axis=0)
    D6.append(newfeaturelist2)
    newfeaturelist3 = numpy.concatenate((newfeaturelist, out3[i]),axis=0)
    D7.append(newfeaturelist3)
    i+=1
    
    

Featureset1 = numpy.array(D5)
Featureset2 = numpy.array(D6)
Featureset3 = numpy.array(D7)

GenrePredictions = GenreClassifier.predict(Featureset1)
PolarityPredictions = PolarityClassifier.predict(Featureset2)
EmotionPredictions = TopicClassifier.predict(Featureset3)

with open(str(args.output[0]),'w') as file:
    i = 0
    for item in testData:
        file.write(str(i)+ '\t' + str(item) + '\t' + str(PolarityPredictions[i]) + '\t' + str(EmotionPredictions[i]) + '\t' + str(GenrePredictions[i]) + '\t\n' )
        i+=1
'''
array = []
for item in trainfeatures:
    print(item)
    array.append(item[2])
print_metrics(array, array)
'''

########################################################################################################################
##don't mind me just generating some test stuff

########################################################################################################################
# task 1

print('Task 1 Results:')
gpred = GenreClassifier.predict(GenreFeatureset)
print_metrics(gpred.tolist(), GenreTarget.tolist())

########################################################################################################################
# task 2

print('Task 2 Results:')
ppred = PolarityClassifier.predict(PosFeatureset)
print_metrics(ppred.tolist(), PosTarget.tolist())

########################################################################################################################
# task 3

print('Task 3 Results:')
epred = TopicClassifier.predict(EmFeatureset)
print_metrics(epred.tolist(), EmTarget.tolist())

########################################################################################################################

