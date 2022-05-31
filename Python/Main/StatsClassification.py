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

def calculate_accuracy(actual_labels, predicted_labels):
    accuracy_percentage = accuracy_score(actual_labels, predicted_labels)
    confusion_matrix_out = confusion_matrix(actual_labels, predicted_labels)

    return accuracy_percentage, confusion_matrix_out

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
    current_path = os.getcwd()
    dataset_folder_path = os.path.join(current_path, "..", "Data", "Extracted_features")
    dataset_path = dataset_folder_path + r"\\" + dataset_used + ".csv"
    labels_path = dataset_folder_path + r"\\" + dataset_used + "_labels.csv"
    input_data = pd.read_csv(dataset_path, header=None)
    input_scores_data = pd.read_csv(labels_path, header=None)

    labels_values = create_labels(input_scores_data)

    predicted_labels, classifier = get_predictions("svm", input_data, labels_values)
    percentage_accuracy, confusion_matrix_out = calculate_accuracy(labels_values, predicted_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_out,
                                        display_labels=classifier.classes_)
    cm_display.plot()

    X_train, X_test, y_train, y_test = train_test_split(input_data, labels_values, test_size=0.2)

    a = 1

if __name__=='__main__':
    run_main()