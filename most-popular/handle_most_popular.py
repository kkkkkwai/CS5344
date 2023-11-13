import csv
import shutil
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import SQLContext, Row
import pyspark.sql.functions as F

#insert a new line, for insert a row title when export the result to a file
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

'''
Handle the row records and choose the items (only have one worng)
Source file itmes: 
video_id,title,publishedAt,channelId,channelTitle,categoryId,trending_date,tags,
view_count,likes,dislikes,comment_count,thumbnail_link,comments_disabled,ratings_disabled,description
'''
def map_file(_x):
  if(len(_x) != 16):
    return Row(id="1", date="1",type="1",value=1)
  #video_id,publishedAt,categoryId,view_count only need them for top rank analysis 
  return Row(id=str(_x[0]), date=str(_x[2])[0:10],type=str(_x[5]),value=int(_x[8]))

'''
Generate top rank for analysis
#0.load category data
#1.load data and choose the item
#2.remove the duplicate reocrds and wrong records (found only one wrong record)
#3.handle top rank by value in date 
#4.choose top 10 rank records only
#5.aggregate top 10 rank per month
#6.order by monty and rank_count
#7.save the data
#8.add title for export data
'''
def generate_top_rank(sc, category_file_path, data_file_path, output_path):
  #0.load category data
  category_ds = load_file(sc, category_file_path, lambda x: Row(id=str(x[0]),name=str(x[1])))
  #1.load data and choose the item
  data_ds = load_file(sc, data_file_path, map_file)
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
  shutil.rmtree(output_path, ignore_errors=True)
  data_ds.rdd.map(lambda x: str(x[0])+","+str(x[1])+","+str(x[2])+","+str(x[3])).coalesce(1).saveAsTextFile(output_path)
  #8.add title for export data
  add_new_line(output_path+"/part-00000","date,type,name,rank_count\n")
  
'''
Local stand alone spark 
'''
if __name__ == "__main__":
  sc = SparkContext()
  sqlContext = SparkSession.builder.appName("example").getOrCreate()
  category_file = "/Users/jiageliu/dev/lab/CS5344/data/youtube-trending-video-dataset/youtube_category.csv"
  #change the dataset file
  data_file = "/Users/jiageliu/dev/lab/CS5344/data/youtube-trending-video-dataset/FR_youtube_trending_data.csv"
  #change export file name
  output_path = "/Users/jiageliu/dev/lab/CS5344/output/FR_youtube_trending_data"
  generate_top_rank(sc, category_file, data_file, output_path)
  sc.stop()