from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.ml.feature import CountVectorizer
from cleantext import sanitize


def positive(label):
    if label == '1':
        return 1
    else:
        return 0


def negative(label):
    if label == '-1':
        return 1
    else:
        return 0


def combine(allgrams):
    return allgrams[1].split() + allgrams[2].split() + allgrams[3].split()


def main(context):
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED
    """Main function takes a Spark SQL context."""

    # CREATES SparkSession, copied from Spark documentation

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    # RUN THESE COMMANDS IN PYSPARK TO CREATE PARQUET FILES FROM THE JSON FILES
    # (USED FOR FASTER LOAD TIME DURING DEVELOPMENT)

    # submission_df = spark.read.json("submissions.json.bz2")
    # submission_df.write.parquet("submissions_parquet")
    # comments_df = spark.read.json("comments-minimal.json.bz2")
    # comments_df.write.parquet("comments_parquet")

    # THESE COMMANDS LOAD THE PARQUET FILES INTO DATAFRAMES

    # submission_df = spark.read.parquet(
    # "/home/cs143/project2/submissions.parquet")
    comments_df = spark.read.parquet("/home/cs143/project2/comments.parquet/")
    labeled_data_df = spark.read.format("csv").load("labeled_data.csv")

    useful_df = labeled_data_df.join(
        comments_df, labeled_data_df._c0 == comments_df.id)

    # spark.udf.register("sanitizeComment", sanitize)
    sanitize_udf = udf(sanitize, ArrayType(StringType()))
    combine_udf = udf(combine, ArrayType(StringType()))

    # Copied from Stack Overflow
    allgrams_df = useful_df.withColumn("allgrams", sanitize_udf("body"))
    combinedgrams_df = allgrams_df.withColumn(
        "combinedgrams", combine_udf("allgrams"))

    # submission_df.printSchema()
    # comments_df.printSchema()
    # labeled_data_df.show()
    # allgrams_df.show()
    # combinedgrams_df.show()
    # combinedgrams_df.select("allgrams").show(truncate=False)
    # combinedgrams_df.select("combinedgrams").show(truncate=False)
    # submission_df.show()
    # allgrams_df.select("allgrams").show()

    # Copied from Spark docs
    cv = CountVectorizer(inputCol="combinedgrams",
                         outputCol="features", minDF=10.0, binary=True)
    model = cv.fit(combinedgrams_df)
    result_df = model.transform(combinedgrams_df)
    # result_df.show(truncate=False)

    positive_udf = udf(positive, IntegerType())
    negative_udf = udf(negative, IntegerType())
    results_positive_df = result_df.withColumn("positive", positive_udf("_c3"))
    allresults_df = results_positive_df.withColumn(
        "negative", negative_udf("_c3"))
    allresults_df.show(truncate=False)


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
