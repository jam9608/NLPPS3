from nltk.metrics import ConfusionMatrix
from collections import Counter

def print_metrics(reference, tagged, tags = None):
	assert len(reference) == len(tagged)

	if tags is None:
		tags = set(reference)

	true_positives = Counter()
	false_positives = Counter()
	false_negatives = Counter()

	for test in range(len(reference)):
		if reference[test] == tagged[test]:
			true_positives[reference[test]] += 1
		else:
			false_positives[tagged[test]] += 1
			false_negatives[reference[test]] += 1

	cm = ConfusionMatrix(reference, tagged)
	print(cm)
	
	accuracy = float(sum(true_positives.values()))/len(tagged)
	precision = float(sum(true_positives.values()))/(sum(true_positives.values()) + sum(false_positives.values()))
	recall = float(sum(true_positives.values()))/(sum(true_positives.values()) + sum(false_negatives.values()))
	fscore = float(2 * precision * recall)/(precision + recall)
	print("Accuracy: " + str(accuracy))
	print("Precision: " + str(precision))
	print("Recall: " + str(recall))
	print("F-score: " + str(fscore))
	print()

	for tag in tags:
		print(tag + "----")
		if true_positives[tag] == 0:
			accuracy = 0
			precision = 0
			recall = 0
			fscore = 0
		else:
			accuracy = float(true_positives[tag])/len(tags)
			precision = float(true_positives[tag])/(true_positives[tag] + false_positives[tag])
			recall = float(true_positives[tag])/(true_positives[tag] + false_negatives[tag])
			fscore = float(2 * precision * recall)/(precision + recall)
		print("Accuracy: " + str(accuracy))
		print("Precision: " + str(precision))
		print("Recall: " + str(recall))
		print("F-score: " + str(fscore))
		print()