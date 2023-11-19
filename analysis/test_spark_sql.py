import csv
import shutil
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

#insert int
def add_new_line(_name, _str):
    with open(_name, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        file.write(_str)
        file.writelines(lines)


if __name__ == "__main__":
  sc = SparkContext()
  sqlContext = SQLContext(sc)
  #0.load category data
  category_file = "data/youtube-trending-video-dataset/youtube_category.csv"
  rdd0 = sc.textFile(category_file)
  header = rdd0.first()
  rdd0_filter = rdd0.filter(lambda row: row != header).mapPartitions(lambda x: csv.reader(x)).map(lambda x: Row(id=str(x[0]),name=str(x[1])))
  categoryTable = rdd0_filter.toDF()
  sqlContext.registerDataFrameAsTable(categoryTable, "categoryTable")
  #categoryTable.show()
  #1.load data
  #us_test_file = "data/youtube-trending-video-dataset/US_test_data.csv"   
  us_test_file = "data/youtube-trending-video-dataset/US_youtube_trending_data.csv"
  rdd1 = sc.textFile(us_test_file)
  header = rdd1.first()
  #2.choose items and convert the data formate from yyyy-mm-dd hh:mi:ss to yyyy-mm-dd
  rdd_filter1 = rdd1.filter(lambda row: row != header).mapPartitions(lambda x: csv.reader(x)).map(lambda x: Row(id=str(x[0]),date=str(x[2])[0:10],type=str(x[5]),value=int(x[8])))
  videoTable = rdd_filter1.toDF()
  #myDF.show()
  sqlContext.registerDataFrameAsTable(videoTable, "videoTable")
  #3.remove the duplicate reocrds  
  videoTable = sqlContext.sql("SELECT id,date,type,max(value) as value FROM videoTable group by id,date,type")
  sqlContext.registerDataFrameAsTable(videoTable, "videoTable")
  #4.handle top rank by value in date 
  videoTable = sqlContext.sql("select id,date,type,value,ROW_NUMBER() OVER(PARTITION BY date \
    ORDER BY value desc) rank_value from videoTable")
  sqlContext.registerDataFrameAsTable(videoTable, "videoTable")
  #5.choose top 10 rank records only
  videoTable = sqlContext.sql("select id,SUBSTRING(date,0,7) as date,type,value,rank_value \
                              from videoTable where rank_value <= 5")
  sqlContext.registerDataFrameAsTable(videoTable, "videoTable")
  #6.aggregate top 10 rank per month
  videoTable = sqlContext.sql("select `videotable`.`date`,`videotable`.`type`,`categorytable`.`name`,\
                              count(*) as rank_count from videoTable,categoryTable where \
    videoTable.type == categoryTable.id group by \
    videoTable.date,videoTable.type,categoryTable.name")
  #7.order by monty and rank_count
  sqlContext.registerDataFrameAsTable(videoTable, "videoTable")
  videoTable = sqlContext.sql("select * from videoTable order by date desc,rank_count desc")
  #videoTable.show()
  #videoTable.coalesce(1).write.csv("output/videoTopRank.csv")
  #8.save the data
  us_test_out_file = "output/US_test_data_out"
  shutil.rmtree(us_test_out_file, ignore_errors=True)
  videoTable.rdd.map(lambda x: str(x[0])+","+str(x[1])+","+str(x[2])+","+str(x[3])).coalesce(1).saveAsTextFile(us_test_out_file)
  #9.add title 
  add_new_line(us_test_out_file+"/part-00000","date,type,name,rank_count\n")
  sc.stop()
