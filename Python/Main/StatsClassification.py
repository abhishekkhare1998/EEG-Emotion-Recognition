"""
This File is an implementation of Statistical Supervised Methods being run on EEG's extracted features data
"""

import sys
"""
 This Project Works exclusively with Python 3.6.8 and the libraries based on python linux requirements 
 (reqs_linux.txt),
 please use a virtual environment configured with python 3.6.8 and install the libraries in linux-reqs.txt
 This is an essential part of setting the environment, working without setting up the environment properly 
 would lead to numerous environmental errors
 """

if sys.version[0:5] != '3.6.8':
    print("\n you are currently using python version - {},\n\n please use python 3.6.8".format(sys.version[0:5]))
    sys.exit()

import os
import pandas as pd
import datetime
from sys import platform

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from classification.Amigos.AmigosUtil import AmigosUtil


def get_labels_ind(label_dataframe, is_valence):
    """
    |Method Used for creating discrete labels from valence or Arousal values independently
    :param label_dataframe: valence/Arousal labels data frame
    :param is_valence: Boolean value is True if its valence
    :return: labels list
    """
    if is_valence:
        class_data = label_dataframe.iloc[:, 0]
    else:
        class_data = label_dataframe.iloc[:, 1]
    va_list = []
    for i in class_data:
        if i <= 3.5:
            va_list.append(1)
        if (i > 3.5) and (i <= 6.5):
            va_list.append(2)
        if i > 6.5:
            va_list.append(3)
    return va_list


def create_labels(label_dataframe):
    """
    |Method Used for creating discrete labels from both Valence and Arousal taken at once
    :param label_dataframe: Data frame containing labels info for both valence and arousal
    :return: discretized labels list
    """
    valence_data = label_dataframe.iloc[:, 0]
    arousal_data = label_dataframe.iloc[:, 1]
    va_list = []
    aa_list = []
    for i in valence_data:
        if i <= 3.5:
            va_list.append("LV")
        if (i > 3.5) and (i <= 6.5):
            va_list.append("MV")
        if i > 6.5:
            va_list.append("HV")

    for i in arousal_data:
        if i <= 3.5:
            aa_list.append("LA")
        if (i > 3.5) and (i <= 6.5):
            aa_list.append("MA")
        if i > 6.5:
            aa_list.append("HA")

    labels_str_list = ["{}{}".format(va, aa) for va, aa in zip(va_list, aa_list)]

    labels_dict = {"LVLA": 1, "LVMA": 2, "LVHA": 3, "MVLA": 4, "MVMA": 5,
                   "MVHA": 6, "HVLA": 7, "HVMA": 8, "HVHA": 9}
    labels_num = [labels_dict[i] for i in labels_str_list]
    return labels_num


def get_predictions(classification_type, input_data, labels):
    """
    |Trains the respective classifier and fits it
    :param classification_type: type of method used - SVM, KNN, etc
    :param input_data: input data
    :param labels: input labels
    :return: list of classification predictions and the classifier model
    """
    classifiers_dict = {"knn": KNeighborsClassifier(n_neighbors=3),
                        "svm": SVC(kernel='rbf'),
                        "d_tree": DecisionTreeRegressor(random_state=0),
                        "g_bayes": GaussianNB()}
    classifier = classifiers_dict[classification_type]
    classifier.fit(input_data, labels)
    predictions = classifier.predict(input_data)
    return predictions, classifier


def calculate_accuracy(test_labels, test_predictions, method, is_valence):
    """
    | Calculated the accuracy percentage and confusion Matrix
    :param test_labels: test labels actually present
    :param test_predictions: predicted test labels
    :param method: Method used for classification
    :param is_valence: Boolean value, is true if its valence classification
    :return: accuracy percentage and confusion matrix
    """
    acc, cm = AmigosUtil().calculate_accuracy(test_labels,
                                              test_predictions,
                                              method,
                                              is_valence)
    return acc, cm


def run_main():
    """
    |Main method for running statistical supervised classification methods
    :return: None
    """
    dataset_used = "amigos"  # "amigos" or "dreamer"
    if "win" in platform:
        current_path = os.path.realpath(__file__).rsplit("\\", 1)[0]
        dataset_folder_path = os.path.join(current_path.rsplit("\\", 1)[0], "Data", "Extracted_features")
    else:
        current_path = os.path.realpath(__file__).rsplit("/", 1)[0]
        dataset_folder_path = os.path.join(current_path.rsplit(r"/", 1)[0], "Data", "Extracted_features")

    dataset_path = os.path.join(dataset_folder_path, dataset_used + ".csv")
    labels_path = os.path.join(dataset_folder_path, dataset_used + "_labels.csv")

    input_data = pd.read_csv(dataset_path, header=None)
    input_scores_data = pd.read_csv(labels_path, header=None)

    labels_values_valence = get_labels_ind(input_scores_data, True)
    labels_values_arousal = get_labels_ind(input_scores_data, True)
    supervised_methods = ["svm", "knn", "d_tree", "g_bayes"]

    now = datetime.datetime.now()
    current_time = now.strftime("%m_%d_%Y_%H_%M_%S")

    save_folder = os.path.join(current_path, "results", current_time+"stats_learn")
    if not os.path.isdir(os.path.join(current_path, "results")):
        os.mkdir(os.path.join(current_path, "results"))
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    for types in ["valence", "arousal"]:
        if types == "valence":
            input_labels = labels_values_valence
            is_valence = True
        else:
            input_labels = labels_values_arousal
            is_valence = False

        train_data, test_data, training_labels, test_labels = train_test_split(input_data,
                                                                               input_labels,
                                                                               test_size=0.2)

        dataset_dict = {"train_data": train_data, "training_labels": training_labels,
                        "test_labels": test_labels, "test_data": test_data}

        prepare_results(supervised_methods, dataset_dict, is_valence, save_folder)


def prepare_results(supervised_methods, dataset_dict, is_valence, save_folder):
    """
    | Classify based on different techniques, produce and save the accuracy results and the confusion matrices
    :param supervised_methods: list of supervised methods used
    :param dataset_dict: dictionary containing training and test data and labels
    :param is_valence: Boolean, is True if its valence
    :param save_folder: Folder in which the results are to be saved
    :return: None
    """
    if is_valence:
        print_str = "valence"
    else:
        print_str = "arousal"

    for i in supervised_methods:

        predicted_labels, classifier = get_predictions(i, dataset_dict["train_data"], dataset_dict["training_labels"])
        test_predictions = classifier.predict(dataset_dict["test_data"])

        percentage_accuracy_output, confusion_matrix_output = calculate_accuracy(test_labels=dataset_dict["test_labels"],
                                                                                 test_predictions=test_predictions,
                                                                                 method=i,
                                                                                 is_valence=is_valence)

        cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_output,
                                            display_labels=["low", "mid", "high"])
        cm_display.plot()
        plt.title('Method used - {}, tested on rand Test data'.format(i))

        plt.savefig(os.path.join(save_folder, r'{}_on_{}.png'.format(i, print_str)))

        conf_matrix_prnt = "Method used - {}, tested on |{}| \n accuracy percentage - {:.2f} \n true class on y axis \n {} \n\n".format(i, print_str, percentage_accuracy_output, str(confusion_matrix_output))

        with open(os.path.join(save_folder, "results.txt"), 'a', encoding='utf-8') as f:
            f.write(conf_matrix_prnt)

        print("percentage accuracy using [{}] on {} = {:.2f}% ".format(i, print_str, percentage_accuracy_output))


if __name__ == '__main__':
    run_main()