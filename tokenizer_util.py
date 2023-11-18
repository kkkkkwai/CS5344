from nltk.corpus import stopwords
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer
from pyspark.sql import DataFrame
import nltk


def collect_stopwords():
    nltk.download("punkt")
    nltk.download("stopwords")
    with open("stopwords.txt") as f:
        custom_stopwords = [line.strip() for line in f]
    return list(
        set(stopwords.words("english")).union(set(custom_stopwords))
    )

def tokenize_description(df: DataFrame) -> DataFrame:
    df = RegexTokenizer(inputCol="description", outputCol="tmp_text").transform(df)
    df = StopWordsRemover(
        inputCol="tmp_text", outputCol="filtered_words", stopWords=collect_stopwords()
    ).transform(df)
    df = df.drop("tmp_text")
    return df

def tokenize_tag(df: DataFrame) -> DataFrame:
    df = RegexTokenizer(inputCol="tags", outputCol="tmp_tag", pattern="\\|").transform(
        df
    )
    df = StopWordsRemover(
        inputCol="tmp_tag", outputCol="filtered_tags", stopWords=["[None]"]
    ).transform(df)
    df = df.drop("tmp_tag")
    return df
    def __init__(self) -> None:
        nltk.download("punkt")
        nltk.download("stopwords")
        with open("stopwords.txt") as f:
            custom_stopwords = [line.strip() for line in f]
        self.all_stopwords = list(
            set(stopwords.words("english")).union(set(custom_stopwords))
        )

    def tokenize_text(
        self,
        df: DataFrame,
        input_col: str,
        output_col: str,
    ) -> DataFrame:
        """
        Tokenize title or description and return the new DataFrame
        """
        df = RegexTokenizer(inputCol=input_col, outputCol="tmp_text").transform(df)
        df = StopWordsRemover(
            inputCol="tmp_text", outputCol=output_col, stopWords=self.all_stopwords
        ).transform(df)
        return df.drop("tmp_text")

    def tokenize_tag(
        self,
        df: DataFrame,
        input_col: str,
        output_col: str,
    ):
        """
        Tokenize tags string e.g. "cgpgrey|education|hello internet"
        """
        df = RegexTokenizer(
            inputCol=input_col, outputCol="tmp_tag", pattern="\\|"
        ).transform(df)
        df = StopWordsRemover(
            inputCol="tmp_tag", outputCol=output_col, stopWords=["[None]"]
        ).transform(df)
        return df.drop("tmp_tag")
