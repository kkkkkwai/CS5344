## Config env
Run on Mac OS or Linux (PySpark have issue on Window)

Before running the program, please make sure java,spark cluster,python3, and Mac OS correctly. 
```
alias python=python3
#java
export JAVA_HOME=/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home
export PATH="$PATH:$JAVA_HOME/bin"
#mvn
export MAVEN_HOME=/Users/jiageliu/dev/maven
export PATH="$PATH:$MAVEN_HOME/bin"
#spark
export SPARK_HOME=/Users/jiageliu/dev/app/spark
export PATH="$PATH:$SPARK_HOME/bin"
#hadoop
export HADOOP_HOME=/Users/jiageliu/dev/app/spark
#python
export PYSPARK_PYTHON=/opt/homebrew/bin/python3
export PATH="$PATH:$PYSPARK_PYTHON"
export PYSPARK_DRIVER_PYTHON=/opt/homebrew/bin/python3
export PATH="$PATH:$PYSPARK_DRIVER_PYTHON"
```

## Generate analysis data 
Input data source:
YouTube Trending Video Dataset:
https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset/data
video catetory data source: 
this project/data/youtube_category.csv

Output file: 
Main function is handle_most_popular.py 
* change the dataset file path in the main function, and then run the main function. 

the program will be excute below steps: 
1. load category data
2. load data and choose the item
3. remove the duplicate reocrds and wrong records (found only one wrong record)
4. handle top rank by value in date 
5. choose top 10 rank records only
6. aggregate top 10 rank per month
7. order by monty and rank_count
8. save the data
9. add title for export data

## Visualization analysis of exported data (Needs install Jupyter)
the export data already is analysis data, the data size is smaller than raw data. 
so we can easly use EDA Visualization analysis data. 

EDA Visualization analysis:
* final-analysis-most-popular.ipynb (analysis the export data) *Mainly use this one to analysis export data. 
* pre-analysis-most-popular.ipynb (pre analysis US raw data)