import sys

sys.path.append(".")

from tokenizer_util import tokenize_description, tokenize_tag
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import date_diff, dense_rank
from pyspark.ml import Transformer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import (
    VectorAssembler,
    HashingTF,
    MinHashLSH,
)
from pyspark.mllib.evaluation import RegressionMetrics


ROOT_PATH = "/home/kw/CS5344/CS5344/data/youtube-trending-video-dataset/"
TEST_PATH = "/home/kw/CS5344/CS5344/data/test_dataset/"
PREDICT_OUTPUT = "/home/kw/CS5344/CS5344/data/predict"
COUNTRIES = ["CA", "US"]


def filter_and_cast(df: DataFrame) -> DataFrame:
    # Keep only data before 11-01
    df = df[df["trending_date"] < "2023-11-01"]
    df = df.fillna("")
    df = df.drop_duplicates(["video_id"])

    df = df.withColumn(
        "days_since_published", date_diff(df["trending_date"], df["publishedAt"])
    )
    df = df.withColumns(
        {
            "comments_disabled": df["comments_disabled"].cast("boolean"),
            "categoryId": df["categoryId"].cast("int"),
            "view_count": df["view_count"].cast("int"),
        }
    )
    return df


def construct_hashing_assembler() -> tuple[Transformer, Transformer, Transformer]:
    hashing_text = HashingTF(
        numFeatures=32,
        inputCol="filtered_words",
        outputCol="hashed_words",
    )
    hashing_tags = HashingTF(
        numFeatures=8,
        inputCol="filtered_tags",
        outputCol="hashed_tags",
    )
    assembler = VectorAssembler(
        inputCols=[
            # "days_since_published",
            "comments_disabled",
            "categoryId",
            "hashed_words",
            "hashed_tags",
        ],
        outputCol="features",
        handleInvalid="skip",
    )
    return hashing_text, hashing_tags, assembler


def train(
    spark: SparkSession,
) -> tuple[Transformer, Transformer, Transformer, Transformer]:
    """
    Returns
    -------
    :py:class:`Transformer`
        description transformer
    :py:class:`Transformer`
        tags transformer
    :py:class:`Transformer`
        vector assembler transformer
    :py:class:`Transformer`
        model
    """
    file_list = [
        f"{ROOT_PATH}/{c}_youtube_trending_data_processed.csv" for c in COUNTRIES
    ]
    df = spark.read.option("header", True).csv(file_list[0])
    print(df)

    df = filter_and_cast(df)
    df = tokenize_description(df)
    df = tokenize_tag(df)

    hashing_text, hashing_tags, assembler = construct_hashing_assembler()
    df = hashing_text.transform(df)
    df = hashing_tags.transform(df)
    df = assembler.transform(df)

    regressor = DecisionTreeRegressor(
        maxDepth=10,
        maxMemoryInMB=512,
        featuresCol="features",
        labelCol="view_count",
        predictionCol="prediction",
    )
    model = regressor.fit(df)
    # regressor.save("model/decision_tree")
    return hashing_text, hashing_tags, assembler, model


def evaluate(
    hashing_text: Transformer,
    hashing_tags: Transformer,
    assembler: Transformer,
    model: Transformer,
):
    """
    Evaluation here
    """
    file_list = [
        f"{TEST_PATH}/{c}_youtube_trending_data_processed_test.csv" for c in COUNTRIES
    ]
    df = spark.read.option("header", True).csv(file_list[0])

    df = filter_and_cast(df)
    df = tokenize_description(df)
    df = tokenize_tag(df)

    df = hashing_text.transform(df)
    df = hashing_tags.transform(df)
    df = assembler.transform(df)
    df = model.transform(df)

    # Eval based on rank instead of abs view count
    df = df.withColumn(
        "expected",
        dense_rank().over(Window.partitionBy("channelId").orderBy("view_count")),
    )
    df = df.withColumn(
        "predicted",
        dense_rank().over(Window.partitionBy("channelId").orderBy("prediction")),
    )
    df = df.withColumns(
        {
            "expected": df["expected"].cast("double"),
            "predicted": df["predicted"].cast("double"),
        }
    )

    metrics = RegressionMetrics(df.select("expected", "predicted").rdd.map(tuple))
    print(f"MAE: {metrics.meanAbsoluteError}, MSE: {metrics.meanSquaredError}")
    sub_df = df.limit(100).select(
        ["video_id", "title", "channelId", "channelTitle", "expected", "predicted"]
    )
    sub_df.write.csv(f"{PREDICT_OUTPUT}/decision_tree1.csv", header=True)


if __name__ == "__main__":
    spark: SparkSession = SparkSession.builder.getOrCreate()
    hashing_text, hashing_tags, assembler, model = train(spark)
    print(model.featureImportances)
    evaluate(hashing_text, hashing_tags, assembler, model)
