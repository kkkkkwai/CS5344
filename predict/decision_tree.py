from d_tokenize import Tokenizer
from pyspark.sql import SparkSession
from pyspark.ml import Transformer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler, HashingTF
import pyspark.pandas as pd

ROOT_PATH = "/home/kw/CS5344/CS5344/data/youtube-trending-video-dataset/"
COUNTRIES = ["CA", "GB", "US", "JP"]


def train(spark: SparkSession) -> (HashingTF, HashingTF, VectorAssembler, Transformer):
    file_list = [f"{ROOT_PATH}{c}_youtube_trending_data_processed.csv" for c in COUNTRIES]
    df = spark.read.csv(file_list, )
    df['trending_date'] = pd.to_datetime(df['trending_date'])
    df = df[df['trending_date'] < "2023-11-01"]
    df = df.sort_values("trending_date", ascending=False).drop_duplicates("video_id")

    tokenizer = Tokenizer()
    df["filtered_words"] = df["description"].apply(lambda x: tokenizer.tokenize_text(x))
    df["filtered_tags"] = df["tags"].apply(lambda x: tokenizer.tokenize_tag(x))
    hashing_text = HashingTF(numFeatures=32, inputCol="filtered_words", outputCol="hashed_words")
    df = hashing_text.transform(df)
    hashing_tags = HashingTF(numFeatures=8, inputCol="filtered_tags", outputCol="hashed_tags")
    df = hashing_tags.transform(df)
    # df = df.drop("video_id", "title", "channelId", "channelTitle", "categoryId", "tags", "thumbnail_link", "comments_disabled", "ratings_disabled", "description")
    assembler = VectorAssembler(inputCols=["publishedAt", "comments_disabled", "category_name", "hashed_words", "hashed_tags"], outputCol="features")
    df = assembler.transform(df)
    regressor = DecisionTreeRegressor(maxDepth=10, maxMemoryInMB=512, input="features", predictionCol="view_count")
    model = regressor.fit(df)
    return hashing_text, hashing_tags, assembler, model
    

if __name__ == "__main__":
    spark: SparkSession = SparkSession.builder.getOrCreate()
    train(spark)