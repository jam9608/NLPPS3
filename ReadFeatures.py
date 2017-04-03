import pickle


with open(r'PS3_training_data_features.bin', 'rb') as file:
     features = pickle.load(file)

print(features[2272])

