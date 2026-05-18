# FairVision — Age Group Classification

A CNN-based age group classification web application built with fairness detection and bias mitigation at its core.

## Live App

https://fairvision-app.streamlit.app/

## Repository

https://github.com/janith-t/fairvision-app

## About

FairVision classifies a face image into one of 9 age groups using a ResNet-style CNN trained from scratch on the FairFace dataset. The project includes a full fairness audit across 7 race groups and 2 gender groups, with two bias mitigation strategies evaluated alongside the baseline model.

## Run Locally

```bash
git clone https://github.com/janith-t/fairvision-app.git
cd fairvision-app
pip install -r requirements.txt
streamlit run app.py
```

## Developer

Janith Thiwanka  
janith.tw@gmail.com
