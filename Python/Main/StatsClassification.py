import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np
import datetime
import Utils.classification.confusion_matrix.CM_Util as CM_Util


def get_labels_ind(label_dataframe, is_valence):
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

    labels_dict = {"LVLA": 1, "LVMA": 2, "LVHA": 3, "MVLA": 4, "MVMA": 5, "MVHA": 6, "HVLA": 7, "HVMA": 8, "HVHA": 9}
    labels_num = [labels_dict[i] for i in labels_str_list]
    return labels_num


def get_predictions(classification_type, input_data, labels):
    classifiers_dict = {"knn": KNeighborsClassifier(n_neighbors=3),
                           "svm": SVC(kernel='rbf'),
                           "d_tree": DecisionTreeRegressor(random_state=0),
                           "g_bayes": GaussianNB()}
    classifier = classifiers_dict[classification_type]
    classifier.fit(input_data, labels)
    predictions = classifier.predict(input_data)
    return predictions, classifier

def run_main():
    dataset_used = "amigos"  # "amigos" or "dreamer"
    current_path = os.path.realpath(__file__).rsplit("\\", 1)[0]
    dataset_folder_path = os.path.join(current_path, "..", "Data", "Extracted_features")
    dataset_path = dataset_folder_path + r"\\" + dataset_used + ".csv"
    labels_path = dataset_folder_path + r"\\" + dataset_used + "_labels.csv"
    input_data = pd.read_csv(dataset_path, header=None)
    input_scores_data = pd.read_csv(labels_path, header=None)

    labels_values_valence = get_labels_ind(input_scores_data, True)
    labels_values_arousal = get_labels_ind(input_scores_data, True)
    supervised_methods = ["svm", "knn", "d_tree", "g_bayes"]

    now = datetime.datetime.now()
    current_time = now.strftime("%m_%d_%Y_%H_%M_%S")

    save_folder = os.path.join(current_path, "results", current_time)
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
    if is_valence:
        print_str = "valence"
    else:
        print_str = "arousal"

    for i in supervised_methods:

        predicted_labels, classifier = get_predictions(i, dataset_dict["train_data"], dataset_dict["training_labels"])
        test_predictions = classifier.predict(dataset_dict["test_data"])

        percentage_accuracy_real, confusion_matrix_out_real = CM_Util.CM_Util().calculate_accuracy(dataset_dict["test_labels"],
                                                                                                    test_predictions,
                                                                                                    i,
                                                                                                    is_valence)

        results_string = ""

        cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_out_real,
                                            display_labels=["low", "mid", "high"])
        cm_display.plot()
        plt.title('Method used - {}, tested on rand Test data'.format(i))

        plt.savefig('{}\\\\{}_on_{}.png'.format(save_folder, i, print_str))

        print("percentage accuracy using [{}] on test data = {:.2f}% ".format(i, percentage_accuracy_real))


run_main()