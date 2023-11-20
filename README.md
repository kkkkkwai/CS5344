# CS5344 YouTube Video Popularity Analysis

## Dataset
Dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset) and [Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new). Dataset contains more than 4GB data of trending youtube videos in different countries.

## Requirements and Preprocess
The scripts in this repository run with python version 3.9.18. Run `pip install -r requirements.txt` if there is dependency issue.

Download dataset into _data_ directory. Modify _TRENDING_VIDEO_DATA_ in preprocess.py if necessary, then run
```
python preprocess.py
```

## Analysis and Prediction Model
Refer to [this](analysis/README.md) for instructions to run the analysis

Refer to notebooks inside _predict_ directory for details to prediction model implementation