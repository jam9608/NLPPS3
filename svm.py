import pickle
import numpy
import re
from sklearn import svm

with open(r'PS3_training_data_features.bin', 'rb') as file:
     features = pickle.load(file)

###################################WORD COUNT CODE#############################
i = 0
with open(r"C:\Users\John\PS3_training_data.txt") as file:
    training = file.readlines()

trainingData = ["" for x in range(2564)]
trainingTarget = ["" for x in range(2564)]

for line in training:
    ##Line sentance Pos/neg event genre \n##
    data = re.split(r'\t', line)
    #dataset[i] = data[2:-1]
    #print(data[1])
    trainingData[i] = data[1]
    trainingTarget[i] = data[4]
    
    i+=1

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainingData)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

from sklearn import svm
C=1.0
svcwc = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTarget)
predicted = svcwc.predict(X_train_tf)

##############################################################################


D2 = []
##print(features)


for item in features:
    D2.append([item['adjective_count'], item['noun_count'], item['verb_count'], item['punctuation_count'], item['number_count'], item['sentence_length'], item['start_with_personal_pronoun'], item['word_count'], item['named_entity']])

npD2= numpy.array(D2)
print(npD2.shape)
trainingFeatures = D2[:2000]
testFeatures = D2[2000:]

i = 0
with open(r"C:\Users\John\PS3_training_data.txt") as file:
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
    
tf_2D = X_train_tf.toarray()
i=0
for n in training[:2000]:
    trainingFeatures[i] = trainingFeatures[i] + tf_2D[i]
z=0
for n in training[2000:]:
    
    trainingFeatures[z] = trainingFeatures[z] + tf_2D[i]
    z+=1
    i+=1

nptraining = numpy.array(trainingFeatures)
nptarget = numpy.array(trainingTarget)
nptest = numpy.array(testFeatures)
nptestTarget = numpy.array(testTarget)


print(nptraining.shape)
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(nptraining, nptarget)
predicted = svc.predict(nptest)
print(numpy.mean(predicted == nptestTarget))
##doing some feature selection
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.model_selection import cross_val_score

skb = SelectKBest(chi2, k=3).fit(nptraining, nptarget)

npdata_knew = skb.transform(nptraining)
npdata_test = skb.transform(nptest)

svc2 = svm.SVC(kernel='linear', C=C).fit(npdata_knew, nptarget)
predicted = svc2.predict(npdata_test)
print(numpy.mean(predicted == nptestTarget))

print(svcwc.decision_function(X_train_tf[i]))