import csv
import shutil
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import SQLContext, Row
import pyspark.sql.functions as F

#insert int
def add_new_line(_name, _str):
  with open(_name, 'r+') as file:
    lines = file.readlines()
    file.seek(0)
    file.write(_str)
    file.writelines(lines)
      
#load file
def load_file(_sc, _name, _lambda):
  rdd = _sc.textFile(_name)
  header = rdd.first()
  filter = rdd.filter(lambda row: row != header).mapPartitions(lambda x: csv.reader(x)).map(_lambda)
  return filter.toDF()
  
if __name__ == "__main__":
  sc = SparkContext()
  #sqlContext = SQLContext(sc)
  sqlContext = SparkSession.builder.appName("example").getOrCreate()
  #0.load category data
  category_file = "data/youtube-trending-video-dataset/youtube_category.csv"
  category_ds = load_file(sc, category_file, lambda x: Row(id=str(x[0]),name=str(x[1])))
  #1.load data 
  data_file = "data/youtube-trending-video-dataset/US_youtube_trending_data.csv"
  #data_file = "data/youtube-trending-video-dataset/US_test_data.csv"
  #2.choose items and convert the data formate from yyyy-mm-dd hh:mi:ss to yyyy-mm-dd
  data_ds = load_file(sc, data_file, lambda x: Row(id=str(x[0]), date=str(x[2])[0:10],type=str(x[5]),value=int(x[8])))
  #3.remove the duplicate reocrds  
  #videoTable2 = sqlContext.sql("SELECT id,date,type,max(value) as value FROM videoTable group by id,date,type limit 10")
  data_ds = data_ds.groupBy('id','date','type').agg(F.max('value').alias('value'))
  #data_ds.show()
  a2 = data_ds.head(10)
  print(a2)
  #4.handle top rank by value in date 
  #videoTable = sqlContext.sql("select id,date,type,value,ROW_NUMBER() OVER(PARTITION BY date \
  #  ORDER BY value desc) rank_value from videoTable")
  #window_spec = Window.partitionBy("date").orderBy(F.col("value").desc())
  print("==========done============")
  sc.stop()
  