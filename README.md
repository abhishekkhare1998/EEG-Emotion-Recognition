# EEG-Emotion-Recognition
Human Emotion recognition using EEG Signals

## Steps to Reproduce : 

1) login to dsmlp

![dsmlp_login_final1](https://user-images.githubusercontent.com/20601671/172792891-98d5c836-6b40-4def-b3ce-a4e4c37127dc.gif)


2) create a project folder - 

```
mkdir Final
```

3) clone the repository using - 

```
git clone https://github.com/abhishekkhare1998/EEG-Emotion-Recognition.git
```

![git_clone](https://user-images.githubusercontent.com/20601671/172794481-05989539-82fd-4861-bf71-aa6f9c5cd3bd.gif)


4) go to the directory - 

<Project_folder>/EEG-Emotion-Recognition/Python/

5)create a virtualenv (make sure you are using python 3.6.8, [python3 --version])

```
python3 --version
```

```
 python3 -m venv ./venv
```

6)activate the virtualenv

```
 source venv/bin/activate
```

![activate_venv](https://user-images.githubusercontent.com/20601671/172796146-8acc0cf5-fb3a-41e8-a635-1fb476b59a4f.gif)


7)install the linux requirements

```
pip install --upgrade pip
pip install -r reqs_linux.txt
```

![pip_install](https://user-images.githubusercontent.com/20601671/172798509-da39a235-9f0b-41b3-8b63-63863b06a30f.gif)


8) go to the directory - 

EEG-Emotion-Recognition/Python/Main/

9)run stats learning - 

```
python3 StatsClassification.py
```

![run_stats_learning](https://user-images.githubusercontent.com/20601671/172800982-581073de-7dd4-4512-8472-6deb03d6969d.gif)


10) to check results, go to - 

EEG-Emotion-Recognition/Python/Main/Results/<timestamp folder>

11)confusion matrices images are present in this folder, text results stored at - EEG-Emotion-Recognition/Python/Main/Results/<timestamp folder>/results.txt
 
 ![check_stats_learning_results](https://user-images.githubusercontent.com/20601671/172803510-c80c5f34-8c31-4279-a069-33661c916fee.gif)

 
