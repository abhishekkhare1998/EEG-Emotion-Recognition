import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dbn.tensorflow import SupervisedDBNClassification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from dbn.tensorflow import SupervisedDBNRegression
import numpy as np
from sys import platform
from classification.Amigos.AmigosUtil import AmigosUtil
import datetime

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


def run_main():
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

    now = datetime.datetime.now()
    current_time = now.strftime("%m_%d_%Y_%H_%M_%S")

    save_folder = os.path.join(current_path, "results", current_time+"DBN_learn")
    if not os.path.isdir(os.path.join(current_path, "results")):
        os.mkdir(os.path.join(current_path, "results"))
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    labels_values_valence = get_labels_ind(input_scores_data, True)
    labels_values_arousal = get_labels_ind(input_scores_data, False)

    is_supervised = True

    if(is_supervised):
        ss = StandardScaler()
        input_data = ss.fit_transform(input_data)

        for types in ["valence", "arousal"]:
            if types == "valence":
                input_labels = labels_values_valence
                is_valence = True
            else:
                input_labels = labels_values_arousal
                is_valence = False

            x_train, x_test, y_train, y_test = train_test_split(input_data, input_labels, test_size=0.2)
            clasifier = SupervisedDBNClassification(hidden_layers_structure=[100, 100, 100, 100], learning_rate_rbm=0.05,
                                                learning_rate=0.05, n_epochs_rbm=5, n_iter_backprop=10, batch_size=8,
                                                activation_function='relu', dropout_p=0.2, verbose=False)
            clasifier.fit(x_train, y_train)
            y_predict = clasifier.predict(x_test)

            percentage_accuracy_output, confusion_matrix_output = calculate_accuracy(test_labels=y_test,
                                                                                 test_predictions=y_predict,
                                                                                 method="DBN",
                                                                                 is_valence=is_valence)

            cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_output,
                                            display_labels=["low", "mid", "high"])
            cm_display.plot()
            plt.title('Method used - {}, tested on training data'.format("dbn"))

            plt.savefig(os.path.join(save_folder, r'DBN_on_{}.png'.format(types)))

            conf_matrix_prnt = "Method used - DBN, tested on |{}| \n accuracy percentage - {:.2f} \n true class on y axis \n {} \n\n".format(types, percentage_accuracy_output, str(confusion_matrix_output))
            with open(os.path.join(save_folder, "results.txt"), 'a', encoding='utf-8') as f:
                f.write(conf_matrix_prnt)
            print("percentage accuracy using [{}] on test data = {:.2f}% ".format("dbn", percentage_accuracy_output))
    else:
        valence_data = input_scores_data.iloc[:, 0]
        arousal_data = input_scores_data.iloc[:, 1]
        min_max_scaler = MinMaxScaler()
        input_data = min_max_scaler.fit_transform(input_data)

        # Training
        regressor = SupervisedDBNRegression(hidden_layers_structure=[100, 100, 100, 100],
                                            learning_rate_rbm=0.01,
                                            learning_rate=0.01,
                                            n_epochs_rbm=20,
                                            n_iter_backprop=200,
                                            batch_size=16,
                                            activation_function='relu',
                                            verbose=False)
        regressor.fit(input_data, np.array(labels_values))

        # Test
        X_test = min_max_scaler.transform(input_data)
        Y_pred = regressor.predict(X_test)


def calculate_accuracy(test_labels, test_predictions, method, is_valence):
    acc, cm = AmigosUtil().calculate_accuracy(test_labels,
                                  test_predictions,
                                  method,
                                  is_valence)
    return acc, cm


if __name__=='__main__':
    run_main()