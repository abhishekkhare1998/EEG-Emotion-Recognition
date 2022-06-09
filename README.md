# EEG-Emotion-Recognition
Human Emotion recognition using EEG Signals

Steps to Reproduce : 

1) login to dsmlp

2) create a project folder - 

3) clone the repository using - 

git clone https://github.com/abhishekkhare1998/EEG-Emotion-Recognition.git

4) go to the directory - 

Project_folder/EEG-Emotion-Recognition/Python/

5)create a virtualenv (make sure you are using python 3.6.8, [python --version])

 python3 -m venv ./venv

6)activate the virtualenv

 source venv/bin/activate

7)install the linux requirements

pip install --upgrade pip

pip install -r reqs_linux.txt


8) go to the directory - 
 
EEG-Emotion-Recognition/Python/Main/


9)run stats learning - 

python3 StatsClassification.py

10) to check results, go to - 

EEG-Emotion-Recognition/Python/Main/Results/<timestamp folder>

11)confusion matrices images are present in this folder, text results stored at - EEG-Emotion-Recognition/Python/Main/Results/<timestamp folder>/results.txt
 
 
