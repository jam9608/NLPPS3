import pickle


with open(r'PS3_training_data_features.bin', 'rb') as file:
     features = pickle.load(file)

D2 = [[]]
print(features)


for item in features:
    D2.append([item['id'], item['sentence'], item['topic'], item['genre'], item['polarity'], item['adjective_count'],
    item['noun_count'], item['verb_count'], item['punctuation_count'], item['number_count'], item['sentence_length'],
    int(item['start_with_personal_pronoun']), item['word_count'], item['named_entity']])

print(D2)
print(len(D2))