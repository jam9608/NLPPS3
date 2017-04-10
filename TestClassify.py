import pickle
import argparse
from Metrics import print_metrics


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

# Metrics Usage:
'''
array = []
for item in trainfeatures:
    print(item)
    array.append(item[2])
print_metrics(array, array)
'''

########################################################################################################################
# task 1

print('Task 1 Begin')

print('Task 1 Results:')
#print_metrics()

########################################################################################################################
# task 2

print('Task 2 Begin')

print('Task 2 Results:')
#print_metrics()

########################################################################################################################
# task 3

print('Task 3 Begin')

print('Task 3 Results:')
#print_metrics()

########################################################################################################################
# print output text file

testResults = []
with open(' '.join(args.output), 'w') as out:
    for item in testResults:
        out.write(testResults[0] + '\t' + testResults[1] + '\t' + testResults[4] + '\t' + testResults[2]
                  + '\t' + testResults[3] + '\n')

