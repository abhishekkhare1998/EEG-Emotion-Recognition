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

def create_labels_ind(label_dataframe):
    #valence_data = label_dataframe.iloc[:, 0]
    valence_data = label_dataframe.iloc[:, 1]
    va_list = []
    for i in valence_data:
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
    current_path = os.path.dirname(__file__)
    dataset_folder_path = os.path.join(current_path, "..", "Data", "Extracted_features")
    dataset_path = dataset_folder_path + r"\\" + dataset_used + ".csv"
    labels_path = dataset_folder_path + r"\\" + dataset_used + "_labels.csv"
    input_data = pd.read_csv(dataset_path, header=None)
    input_scores_data = pd.read_csv(labels_path, header=None)

    labels_values = create_labels(input_scores_data)

    is_supervised = False

    if(is_supervised):
        ss = StandardScaler()
        input_data = ss.fit_transform(input_data)
        x_train, x_test, y_train, y_test = train_test_split(input_data, labels_values, test_size=0.2)
        clasifier = SupervisedDBNClassification(hidden_layers_structure=[100, 100, 100, 100], learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=5, n_iter_backprop=10, batch_size=8,
                                            activation_function='relu', dropout_p=0.2)
        clasifier.fit(x_train, y_train)
        y_pred = clasifier.predict(x_test)

        accuracy_percentage = accuracy_score(y_test, y_pred)
        confusion_matrix_out = confusion_matrix(y_test, y_pred)

        cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_out,
                                        display_labels=[1, 2, 3])
        cm_display.plot()
        plt.title('Method used - {}, tested on training data'.format("dbn"))
        plt.savefig('{}\\{}_on_training.png'.format(current_path, "dbn"))
        plt.show()

        plt.close('all')
        print("percentage accuracy using [{}] on training data = {:.2f}% ".format("dbn", accuracy_percentage * 100))
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
                                            activation_function='relu')
        regressor.fit(input_data, np.array(labels_values))

        # Test
        X_test = min_max_scaler.transform(input_data)
        Y_pred = regressor.predict(X_test)
        a = 1

    a = 1

if __name__=='__main__':
    run_main()