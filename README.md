# NLPPS3

Jared Wagner
Ian Gilbert
John Murry

1. To create test results first run the test code through the ThreadedFeatures.py script.
    This script takes command line parameters of the input text file and the output file names
    Usage: python ThreadedFeatures.py -i INPUT_TEXT_FILE_NAME.txt -a OUTPUT_BINARY_FEATURE_ARRAY.bin
    Where the INPUT_TEXT_FILE_NAME.txt is the name of the test data file which is assumed to be in the same directory as the project files
    and the OUTPUT_BINARY_FEATURE_ARRAY.bin is the name you make up
    Output: progress of the generation and the binary file of the pickled features set

2. To use the classifier run the script TestClassify.py
    This script takes the command line argument of the test binary feature array file
    Usage: python TestClassifier.py -i OUTPUT_BINARY_FEATURE_ARRAY.bin -o OUTPUT_TEXT_FILE_NAME.txt
    Where the OUTPUT_BINARY_FEATURE_ARRAY.bin file has the same name as the one used in the feature generation
    and the file OUTPUT_BINARY_FEATURE_ARRAY_TRAINING.bin is assumed to be in the directory
    Output: the accuracy for the individual tasks and confusion matrices as well as the text file with the estimated properties

3. Metrics.py is called from the TestClassify.py to generate the results of the data and is assumed to be in the same directory

Other files:
    Classifier.py: used to begin the bag of words classification testing
    svm.py: used to begin the support vector classifier
    FeaturesToArray.py: test code to transform the set of the features to an array
    FeaturesWithNE.py: test code to generate features that include named entity recognition using nltk.ne_chunck
    LogLikelyTest.py: test code to find most likely words for each classification task
    PrintInFormat.py: test code to create output to match the input
    ReadFeatures.py: test code to open a pickled file
    SplitData.py: test code to generate features with the Stanford named entity recognizer
    TestDataFlow: ideas of how to solve the tasks