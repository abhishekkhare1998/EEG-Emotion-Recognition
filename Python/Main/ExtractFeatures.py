"""
This File is used as a part of feature extraction.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

import Utils.GeneralUtils as g_util
import Utils.ComputePsd as psd_util

class ExtractFeatures:
    def __init__(self, signal):
        self.signal = signal

    def use_fft(self, signal, point):
        """
        | set FFT of incoming signal
        :return: None
        """
        self.signal = point


class Data:
    def __init__(self):
        self.eeg_data = {}
        self.gen_util_object = g_util.GeneralUtils()
        self.project_root = os.path.abspath(__file__).rsplit("\\", 2)[0]
        self.baseline_signal_folder = os.path.join(self.project_root, "Data\\Baseline")
        self.stimuli_signal_folder = os.path.join(self.project_root, "Data\\Stimuli")
        self.metadata_folder = os.path.join(self.project_root, "Data\\Metadata")
        self.populate_state = False

    def populate_all_signals(self):
        """
        | populate all EEG signals in eeg_data object
        :return: None
        """
        person_dict = {}
        for i in range(23):
            person_id = i + 1
            base_data = self.extract_signal_for_person(person_id, self.baseline_signal_folder)
            stimuli_data = self.extract_signal_for_person(person_id, self.stimuli_signal_folder)
            person_dict[person_id] = {"baseline": base_data, "stimuli": stimuli_data}

        self.eeg_data = pd.DataFrame(person_dict)
        self.populate_state = True

    def plot_signal(self, sig_type='baseline', person_id=1, emotion_seq=1, electrode_num=1):
        """
        | plot respective signal
        :param sig_type: baseline or stimuli
        :param person_id: which person are we viewing, out of 23 ppl
        :param emotion_seq: which seq is played out of 18 seqs
        :param electrode_num: which electrode are we viewing
        :return: None
        """
        req_signal = self.get_exact_signal(sig_type=sig_type, person_id=person_id,
                                           emotion_seq=emotion_seq, electrode_num=electrode_num)
        self.gen_util_object.plot_signal(req_signal)

    def get_exact_signal(self, sig_type='baseline', person_id=1, emotion_seq=1, electrode_num=1):
        """
        |get respective signal
        :param sig_type: baseline or stimuli
        :param person_id: which person are we viewing, out of 23 ppl
        :param emotion_seq: which seq is played out of 18 seqs
        :param electrode_num: which electrode are we viewing
        :return: required signal
        """
        if self.populate_state:
            return self.eeg_data[person_id][sig_type][emotion_seq].values[0][electrode_num]

        else:
            if sig_type == 'baseline':
                signal_folder = self.baseline_signal_folder
            else:
                signal_folder = self.stimuli_signal_folder

            signal_df = self.extract_signal_for_person(person_id, signal_folder)
            req_signal = signal_df[emotion_seq].values[0][electrode_num]

            return req_signal

    def extract_signal_for_person(self, person_id, path):
        """
        | Method used for extracting signal for 1 person
        :param person_id: unique ID for the test subject
        :param path: folder path for data
        :return: person's signal data
        """
        person_dict = {}
        for file in os.listdir(path):
            if file.endswith("csv"):
                if int(file.rsplit('_', 2)[-2]) == person_id:
                    video_num = int(file.rsplit('_', 2)[-1][:-4])
                    csv_path = os.path.join(path, file)
                    data_frame = pd.read_csv(csv_path, header=None)
                    person_dict[video_num] = data_frame

        person_seq = pd.DataFrame([person_dict])
        return person_seq


def extract_features():
    """
    | Main Function for extracting the EEG signals data and features
    :return: None
    """
    gen_util_object = g_util.GeneralUtils()

    project_root = os.path.abspath(__file__).rsplit("\\", 2)[0]
    baseline_signal_folder = os.path.join(project_root, "Data\\Baseline")
    stimuli_signal_folder = os.path.join(project_root, "Data\\Stimuli")
    metadata_folder = os.path.join(project_root, "Data\\Metadata")
    data_object = Data()

    data_object.plot_signal(sig_type='stimuli', person_id=21, emotion_seq=17, electrode_num=13)
    data_object.plot_signal(sig_type='baseline', person_id=1, emotion_seq=1, electrode_num=1)
    data_object.populate_all_signals()

    input_signal = data_object.get_exact_signal(sig_type='baseline', person_id=22, emotion_seq=17, electrode_num=13)
    new_inp_signal = gen_util_object.remove_dc(input_signal)
    gen_util_object.perform_fft(new_inp_signal)

    psd_util_obj = psd_util.ComputePsd(new_inp_signal)
    psd_util_obj.compute()

    # To be used for plotting all signals

    person_base_data = data_object.extract_signal_for_person(3, baseline_signal_folder)
    person_stimuli_data = data_object.extract_signal_for_person(3, stimuli_signal_folder)
    
    for index, col in person_stimuli_data[0].T.iterrows():
        plt.figure(index)
        gen_util_object.plot_signal(col)
        
    temp = index

    for index, col in person_base_data[0].T.iterrows():
        plt.figure(temp+index+1)
        gen_util_object.plot_signal(col)


if __name__ == '__main__':
    extract_features()
