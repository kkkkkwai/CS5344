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

#video_id,title,publishedAt,channelId,channelTitle,categoryId,trending_date,tags,view_count,likes,dislikes,comment_count,thumbnail_link,comments_disabled,ratings_disabled,description
def map_file(_x):
  if(len(_x) != 16):
    return Row(id="1", date="1",type="1",value=1)
  return Row(id=str(_x[0]), date=str(_x[2])[0:10],type=str(_x[5]),value=int(_x[8]))
  
if __name__ == "__main__":
  sc = SparkContext()
  #sqlContext = SQLContext(sc) 42678
  sqlContext = SparkSession.builder.appName("example").getOrCreate()
  #0.load category data
  category_file = "/Users/jiageliu/dev/lab/CS5344/data/youtube-trending-video-dataset/youtube_category.csv"
  category_ds = load_file(sc, category_file, lambda x: Row(id=str(x[0]),name=str(x[1])))
  #1.load data and choose the item
  data_file = "/Users/jiageliu/dev/lab/CS5344/data/youtube-trending-video-dataset/US_youtube_trending_data.csv"
  data_ds = load_file(sc, data_file, map_file)
  #2.remove the duplicate reocrds and wrong records (found only one wrong record)
  data_ds = data_ds.filter(data_ds['id'] != '1')
  data_ds = data_ds.groupBy('id','date','type').agg(F.max('value').alias('value'))
  #3.handle top rank by value in date 
  temp_date_value = Window().partitionBy("date").orderBy(F.desc("value"))
  data_ds = data_ds.select("id", "date", "type", "value", F.row_number().over(temp_date_value).alias("rank_value"))
  #4.choose top 10 rank records only
  data_ds = data_ds.select("id",F.expr("SUBSTRING(date, 0, 7)").alias("date"),"type","value","rank_value").filter(data_ds["rank_value"] <= 10)
  #5.aggregate top 10 rank per month
  temp_joined = data_ds.join(category_ds, data_ds["type"] == category_ds["id"])
  data_ds = temp_joined.groupBy("date", "type", "name").agg(F.count("*").alias("rank_count"))
  #6.order by monty and rank_count
  data_ds = data_ds.orderBy(F.desc("date"), F.desc("rank_count"))
  #data_ds.show()
  #7.save the data
  temp_output = "/Users/jiageliu/dev/lab/CS5344/output/US_youtube_trending_data"
  shutil.rmtree(temp_output, ignore_errors=True)
  data_ds.rdd.map(lambda x: str(x[0])+","+str(x[1])+","+str(x[2])+","+str(x[3])).coalesce(1).saveAsTextFile(temp_output)
  #9.add title 
  add_new_line(temp_output+"/part-00000","date,type,name,rank_count\n")
  print("==========done============")
  sc.stop()
  