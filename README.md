# model_performance_app

# Watch the tutorial video

[How to Build a Machine Learning Model Performance Calculator App](https://youtu.be/Ge17mZe54dY)

<a href="https://youtu.be/Ge17mZe54dY"><img src="http://img.youtube.com/vi/Ge17mZe54dY/0.jpg" alt="How to Build a Machine Learning Model Performance Calculator App" title="How to Build a Machine Learning Model Performance Calculator App" width="400" /></a>

# Demo

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/dataprofessor/model_performance_app/main/app.py)

# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a conda environment called *performance*
```
conda create -n performance python=3.7.9
```
Secondly, we will login to the *performance* environement
```
conda activate performance
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://raw.githubusercontent.com/dataprofessor/model_performance_app/main/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```
###  Download and unzip contents from GitHub repo

Download and unzip contents from https://github.com/dataprofessor/model_performance_app/archive/main.zip

###  Launch the app

```
streamlit run app.py
```
