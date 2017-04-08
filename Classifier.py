# -*- coding: utf-8 -*-
import numpy
import re

def ReadData():
    i = 0
    with open(r"C:\Users\John\PS3_training_data.txt") as file:
        training = file.readlines()
    
    #dataset = numpy.zeros((2564,3),numpy.str)
    trainingData = ["" for x in range(2001)]
    
    testData = ["" for x in range(564)]
    trainingTarget = ["" for x in range(2001)]
    
    testTarget = ["" for x in range(564)]
    for line in training[:2000]:
        ##Line sentance Pos/neg event genre \n##
        data = re.split('\t+', line)
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
        testTarget[i] = data[4]
    
        i+=1
        
    
    print(i)
    #print(len(actualData))
    #print(len(dataset))
    #print(actualData)
    
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(trainingData)
    print(X_train_counts.shape)
    
    from sklearn.feature_extraction.text import TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)
    
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_tf,trainingTarget)
    x_new_counts = count_vect.transform(testData)
    x_new_tfidf = tf_transformer.transform(x_new_counts)
    predicted = clf.predict(x_new_tfidf)
    print(numpy.mean(predicted == testTarget))
    
    with open('output.txt','w') as outputfile:
        z = 0
        for prediction in predicted:
            outputfile.write(testData[z] + '\t' + prediction + '\t\n')
            z+=1
            
    from sklearn import svm
    C=1.0
    svc = svm.SVC(kernel='linear', C=C).fit(X_train_tf, trainingTarget)
    predicted = svc.predict(x_new_tfidf)
    print(numpy.mean(predicted == testTarget))
    
    print(svc.decision_function(X_train_tf[i]))
    
   
    
    
ReadData()