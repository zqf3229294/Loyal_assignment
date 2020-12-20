from utils import *
import numpy as np
import time
import argparse
import json
import random


#manualy select 20 random features
choosen = []
# choosen = ["direct_deposit","carry_on","whisper_mode","text","recipe",
#            "smart_home","who_do_you_work_for","rewards_balance","restaurant_reservation","travel_notification",
#            "update_playlist","change_volume","routing","mpg","bill_balance",
#            "do_you_have_pets","cook_time","what_song","new_card","todo_list_update"]

def random_select_labels(Y):
    unique_labels = np.unique(Y)
    global choosen
    choosen = random.sample(list(unique_labels),20)

def accuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy = correct/len(pred)
    print("Accuracy: %i / %i = %.4f " %(correct, len(pred), accuracy))


def read_data(path):
    with open(path +'data_full.json') as f:
        data = json.load(f)
    train_frame = np.array(data['train'])
    train_label = train_frame[:,1]

    #select 20 random labels
    random_select_labels(train_label)
    print("Select labels:{}".format(choosen))

    #filter the input dataset with selected 20 labels
    train_frame = train_frame[np.in1d(train_label,np.array(choosen))]
    test_frame = np.array(data['test'])
    test_label = test_frame[:, 1]
    test_frame = test_frame[np.in1d(test_label,np.array(choosen))]

    return train_frame, test_frame


class NaiveBayesClassifier:
    """Naive Bayes Classifier
    """

    def __init__(self):
        # Add your code here!
        self.features_prob = []
        self.label_prob = []

    def fit(self, X, Y):
        # Add your code here!
        for i in range(len(choosen)):
            features_count = X[np.in1d(Y, choosen[i])]
            features_count = np.sum(features_count,axis=0)
            #add 1 smoothing
            features_count += 1
            self.features_prob.append(np.divide(features_count, np.sum(features_count)))
            self.label_prob.append(np.sum(np.in1d(Y, choosen[i])) / len(Y))

    def predict(self, X):
        # Add your code here!
        pred = []
        for row in X:
            pred_prob = np.log(self.label_prob) + np.sum(np.log(np.power(self.features_prob, row)),axis=1)
            pred.append(choosen[np.argmax(pred_prob)])
        return np.array(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'customized'])
    parser.add_argument('--path', type=str, default = './data/', help='path to datasets')
    args = parser.parse_args()
    print(args)

    train_frame, test_frame = read_data(args.path)

    # Convert text into features
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "customized":
        feat_extractor = CustomFeature()
    else:
        raise Exception("Pass unigram, bigram or customized to --feature")

    # Tokenize text into tokens
    tokenized_text = []
    for i in range(0, len(train_frame)):
        tokenized_text.append(tokenize(train_frame[i][0]))

    feat_extractor.fit(tokenized_text)

    # form train set for training
    X_train = feat_extractor.transform_list(tokenized_text)
    Y_train = train_frame[:,1]


    # form test set for evaluation
    tokenized_text = []
    for i in range(0, len(test_frame)):
        tokenized_text.append(tokenize(test_frame[i][0]))
    X_test = feat_extractor.transform_list(tokenized_text)
    Y_test = test_frame[:,1]


    model = NaiveBayesClassifier()

    start_time = time.time()
    model.fit(X_train,Y_train)
    print("===== Train Accuracy =====")
    accuracy(model.predict(X_train), Y_train)

    print("===== Test Accuracy =====")
    accuracy(model.predict(X_test), Y_test)

    print("Time for training and test: %.2f seconds" % (time.time() - start_time))



if __name__ == '__main__':
    main()
