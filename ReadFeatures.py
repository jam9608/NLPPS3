import pickle


with open(r'PS3_training_data_features.bin', 'rb') as file:
    print(pickle.load(file))

