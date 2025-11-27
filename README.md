# EEEC (Electroencephalo-emotional Classifer)
## Contributors
- Richard Gao @MrFlyingPizza
- Jooyoung (Julia) Lee @jylee2033
- Calvin Weng @yamikazoo
- Aarham Haider @AarhamH
- Abrar Rahman @abr-rhmn

This repository is a template for your CMPT 340 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

Add a 1-2 line summary of your project here.

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

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.txt             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

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
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
