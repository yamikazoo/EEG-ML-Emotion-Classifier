# EEECDL (Electroencephalo-emotional Classifer via Deep Learning)
## Contributors
- Richard Gao @MrFlyingPizza
- Jooyoung (Julia) Lee @jylee2033
- Calvin Weng @yamikazoo
- Aarham Haider @AarhamH
- Abrar Rahman @abr-rhmn

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/ETNOAUV8d3pLla21L8OfzlkBIbjm6ZrE9dPXMvCE2adCcQ) | [Slack channel](https://cmpt340fall2025.slack.com/archives/C09F0NR7QHZ) | [Project report](https://www.overleaf.com/9314115499gkgmhkrgjdzm#801001) |
|-----------|---------------|-------------------------|

## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

Training Loop
![train_record](https://github.com/user-attachments/assets/61403f00-36dd-461c-959e-88b6de3bbe88)

Sample Charts

<img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/29d4ce4d-3fe1-4455-ba51-9240f4bc2637" />
<img width="300" height="800" alt="image" src="https://github.com/user-attachments/assets/3838322e-397d-4b85-bbfd-6f2ea2f49c93" />
<img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/3520ceb0-cc9b-4eca-bffb-929466721b21" />




### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
    ├── data                     ## code for data preprocessing, feature engineering, and the EEGDataSet class
    ├── model                    ## code for 1-D CNN class and training
    ├── utils                    ## utility
    config.py                    ## contains paths and hyperparameters used by learning model
    main.py                      ## main driver
    run.py                       ## gradio deliverable
├── README.md                    ## You are here
├── requirements.txt             ## If you use conda
```

<a name="installation"></a>

## 2. Installation and Training

Install the project.

1. Create the virtual environment
   ```shell
   python -m venv .venv
   ```
2. Activate the virtual environment

    **macOS / Linux**
   ```shell
   source .venv/bin/activate
   ```
    **Windows**
   ```shell
    .venv\Scripts\activate
   ```
2. Install dependencies using the requirements file.
   ```shell
   pip install -r requirements.txt
   ```
3. If you update the dependencies, remember to freeze it and commit the changes to the requirements.txt everytime.
   ```shell
   pip freeze > requirements.txt
   ```



<a name="repro"></a>
## 3. Reproduction
First enable Github SSH (follow the [guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh))

Then clone the repository
```bash
git clone git@github.com:huytungst/EEGEmotions-27.git
cd EEGEmotions-27
```
Windows Powershell
````powershell
Move-Item -Path "path\to\EEGEmotions-27\training\eeg_features_extracted.csv" -Destination "path\to\2025_3_project_06\src\"
````

Linux/macOS
````bash
mv /path/to/EEGEmotions-27/training/eeg_features_extracted.csv /path/to/2025_3_project_06/src/
````

If you wish the rename the file, or change the path overall, you have to change `CSV_FILE_PATH` parameter under `src/config.py`
````python
class Config:
...
CSV_FILE_PATH = "eeg_features_extracted.csv" <---- CHANGE YOUR .csv PATH HERE
 ...
````

To train the model, simply `cd` into `src/` and run
````bash
python main.py 
````
After training has completed, a set of charts will be created under `plots/` for accuracy and loss for training and validation loops, as well as a confusion matrix.

Data can be found at: [https://github.com/huytungst/EEGEmotions-27](https://github.com/huytungst/EEGEmotions-27)

Output will be saved in: `src/best_cnn_model.pth` (our model with the best model weights)
