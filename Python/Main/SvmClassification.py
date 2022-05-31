import os
import pandas as pd
from sklearn.model_selection import train_test_split

def run_main():
    dataset_used = "amigos"  # "amigos" or "dreamer"
    current_path = os.getcwd()
    dataset_folder_path = os.path.join(current_path, "Data", "Extracted_features")
    dataset_path = dataset_folder_path + r"\\" + dataset_used + ".csv"
    labels_path = dataset_folder_path + r"\\" + dataset_used + "_labels.csv"
    input_data = pd.read_csv(dataset_path)
    input_scores_data = pd.read_csv(labels_path)



if __name__=='__main__':
    run_main()