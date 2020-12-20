### Description of this assignment's completing process
This program runs on the dataset provided in the oos-eval github repository. It use the dat_full.json file as its input. 

The "random_select_labels" method randomly select 20 features as required by the assignment, and we only choose this 20 feature related data record as input, and also we only predict labels inside the scope of this 20 features.

The "read_data" method read the "data_full.json" file as the input, and also filter the input dataset by the 20 selected features.

For the input features, we need to tokenize them. We use the tokenize method provided by nltk package. This method is processed the input data into different features, like unigram features, bigram features, trigram features, remove stopwords, etc. For each row of record, it output the features with the corresponding count.

Finally, for the NaiveBayesClassifier class implementation, the "fit" method calculate features_prob and label_prob, which is the probability of each feature inside each label and the probability of each label.

The result using "unigram" is around 99% for the training dataset, and 95% for the testing dataset.

Because this NaiveBayes method is a generative method, it does not have a trained model, so their is no saved model.

For this whole assignemnt process, because of time limit, I only finish the work of figure out python file implementation and correspondent accuracy result. After the time limit, I spent extra time to finish the documentation and jupyter notebook.
