import pickle
import argparse
import numpy
from Metrics import print_metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='The input binary file feature array', nargs='+')
parser.add_argument('-o', '--output', type=str, help='The output text file for results', nargs='+')
args = parser.parse_args()


########################################################################################################################
# pull in data

if args.input is None or args.output is None:
    print('Please specify all required files')
    exit()

testfeatures = []
with open(' '.join(args.input), 'rb') as file:
    testfeatures = pickle.load(file)

trainfeatures = []
with open(r'OUTPUT_BINARY_FEATURE_ARRAY_TRAINING.bin', 'rb') as file:
    trainfeatures = pickle.load(file)

# feature array syntax:
# 0:id, 1:sentence, 2:topic, 3:genre, 4:polarity, 5:adjective_count, 6:noun_count, 7:verb_count, 8:punctuation_count,
# 9:number_count, 10:sentence_length, 11:start_with_personal_pronoun, 12:word_count, 13:named_entity

########################################################################################################################
# Build classifiers

dimsize = len(trainfeatures) + len(testfeatures)
breaklen = len(trainfeatures)

trainingData = ["" for x in range(dimsize)]
trainingTargetGenre = ["" for x in range(dimsize)]
trainingTargetPositivity = ["" for x in range(dimsize)]
trainingTargetEmotion = ["" for x in range(dimsize)]

i=0
allvalues = trainfeatures+testfeatures
for line in allvalues:
    trainingData[i] = line[1]
    trainingTargetGenre[i] = line[3]
    trainingTargetPositivity[i] = line[4]
    trainingTargetEmotion[i] = line[2]
    i += 1

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainingData)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

C = 1.0
svc1 = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTargetGenre)
svc2 = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTargetPositivity)
svc3 = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTargetEmotion)
predicted1 = svc1.decision_function(X_train_tf)
predicted2 = svc2.decision_function(X_train_tf)
predicted3 = svc3.decision_function(X_train_tf)

########################################################################################################################

D2 = []
D3 = []
D4 = []

i = 0
for item in allvalues:
    newfeaturelist = numpy.array([item[5], item[6], item[7], item[8], item[9], item[10], item[11], item[12], item[13]])
    newfeaturelist1 = numpy.concatenate((newfeaturelist, [predicted1[i]]), axis=0)
    D2.append(newfeaturelist1)
    newfeaturelist2 = numpy.concatenate((newfeaturelist, predicted2[i]), axis=0)
    D3.append(newfeaturelist2)
    newfeaturelist3 = numpy.concatenate((newfeaturelist, predicted3[i]), axis=0)
    D4.append(newfeaturelist3)
    i += 1

trainingFeatures = D2[:breaklen]
testFeatures = D2[breaklen:]

trainGenre = trainingTargetGenre[:breaklen]
testGenre = trainingTargetGenre[breaklen:]

nptraining = numpy.array(trainingFeatures)
nptarget = numpy.array(trainGenre)
nptest = numpy.array(testFeatures)
nptestTarget = numpy.array(testGenre)

trainPos = numpy.array(trainingTargetPositivity[:breaklen])
testPos = numpy.array(trainingTargetPositivity[breaklen:])
nptraining2 = numpy.array(D3[:breaklen])
nptest2 = numpy.array(D3[breaklen:])

trainEm = numpy.array(trainingTargetEmotion[:breaklen])
testEm = numpy.array(trainingTargetEmotion[breaklen:])
nptraining3 = numpy.array(D4[:breaklen])
nptest3 = numpy.array(D4[breaklen:])

C = 1.0

########################################################################################################################
# task 1

print('Task 1 Results:')
svc = svm.SVC(kernel='linear', C=C).fit(nptraining, nptarget)
predicted = svc.predict(nptest)
print_metrics(predicted.tolist(), nptestTarget.tolist())

########################################################################################################################
# task 2

print('Task 2 Results:')
svc = svm.SVC(kernel='linear', C=C).fit(nptraining2, trainPos)
predicted = svc.predict(nptest2)
print_metrics(predicted.tolist(), testPos.tolist())

########################################################################################################################
# task 3

print('Task 3 Results:')
svc = svm.SVC(kernel='linear', C=C).fit(nptraining3, trainEm)
predicted = svc.predict(nptest3)
print_metrics(predicted.tolist(), testEm.tolist())

########################################################################################################################
# print output text file
# TODO make the output file
testResults = []
with open(' '.join(args.output), 'w') as out:
    for item in testResults:
        out.write(testResults[0] + '\t' + testResults[1] + '\t' + testResults[4] + '\t' + testResults[2]
                  + '\t' + testResults[3] + '\n')

