from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession


def main(context):
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED
    """Main function takes a Spark SQL context."""

    # CREATES SparkSession

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    # THESE COMMANDS LOAD THE PARQUET FILES INTO DATAFRAMES

    submission_df = spark.read.parquet(
        "/home/cs143/project2/submissions.parquet")
    comments_df = spark.read.parquet("/home/cs143/project2/comments.parquet/")
    submission_df.printSchema()
    comments_df.printSchema()

    # RUN THESE COMMANDS IN PYSPARK TO CREATE PARQUET FILES FROM THE JSON FILES
    # (USED FOR FASTER LOAD TIME DURING DEVELOPMENT)

    # submission_df = spark.read.json("submissions.json")
    # submission_df.write.parquet("submissions_parquet")
    # comments_df = spark.read.json("comments-minimal.json")
    # comments_df.write.parquet("comments_parquet")


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
