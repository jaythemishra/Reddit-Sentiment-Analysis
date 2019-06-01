from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.ml.feature import CountVectorizer
from cleantext import sanitize
from os import path

# ML Stuff
# Bunch of imports (may need more)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def strip_link(link):
    return link[3:]


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

    # READ FROM THE PARQUET FILES IF THEY HAVE ALREADY BEEN CREATED, OTHERWISE CREATE THEM
    if path.exists("/home/cs143/project2/comments.parquet/") and path.exists("/home/cs143/project2/submissions.parquet/"):
        # TASK 1
        submission_df = spark.read.parquet(
            "/home/cs143/project2/submissions.parquet")
        comments_df = spark.read.parquet(
            "/home/cs143/project2/comments.parquet/")
        print()
        print("LOADED PARQUET")
        print()
    else:
        submission_df = spark.read.json("submissions.json.bz2")
        comments_df = spark.read.json("comments-minimal.json.bz2")
        submission_df.write.parquet("submissions_parquet")
        comments_df.write.parquet("comments_parquet")
        print()
        print("WROTE PARQUET")
        print()

    labeled_data_df = spark.read.format("csv").load("labeled_data.csv")

    # TASK 2
    comment_data_df = comments_df.select("id", "body")
    useful_df = labeled_data_df.join(
        comment_data_df, labeled_data_df._c0 == comment_data_df.id).select("_c3", "id", "body")

    sanitize_udf = udf(sanitize, ArrayType(StringType()))
    combine_udf = udf(combine, ArrayType(StringType()))

    # Copied from Stack Overflow
    # TASK 4
    allgrams_df = useful_df.withColumn("allgrams", sanitize_udf("body"))
    # TASK 5
    combinedgrams_df = allgrams_df.withColumn(
        "combinedgrams", combine_udf("allgrams"))

    # Copied from Spark docs
    # TASK 6A
    cv = CountVectorizer(inputCol="combinedgrams",
                         outputCol="features", minDF=10.0, binary=True)
    model = cv.fit(combinedgrams_df)
    result_df = model.transform(combinedgrams_df)

    # TASK 6B
    positive_udf = udf(positive, IntegerType())
    negative_udf = udf(negative, IntegerType())
    results_positive_df = result_df.withColumn("label", positive_udf("_c3"))
    results_negative_df = result_df.withColumn("label", negative_udf("_c3"))

    # TASK 7
    if path.exists("/home/cs143/project2/project2/pos.model") and path.exists("/home/cs143/project2/project2/neg.model"):
        posModel = CrossValidatorModel.load("project2/pos.model")
        negModel = CrossValidatorModel.load("project2/neg.model")
        print()
        print("MODELS WERE LOADED")
        print()
    else:
        # ML Stuff copied from spec
        # Initialize two logistic regression models.
        # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
        poslr = LogisticRegression(
            labelCol="label", featuresCol="features", maxIter=10)
        neglr = LogisticRegression(
            labelCol="label", featuresCol="features", maxIter=10)
        # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
        posEvaluator = BinaryClassificationEvaluator()
        negEvaluator = BinaryClassificationEvaluator()
        # There are a few parameters associated with logistic regression. We do not know what they are a priori.
        # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
        # We will assume the parameter is 1.0. Grid search takes forever.
        posParamGrid = ParamGridBuilder().addGrid(
            poslr.regParam, [1.0]).build()
        negParamGrid = ParamGridBuilder().addGrid(
            neglr.regParam, [1.0]).build()
        # We initialize a 5 fold cross-validation pipeline.
        posCrossval = CrossValidator(
            estimator=poslr,
            evaluator=posEvaluator,
            estimatorParamMaps=posParamGrid,
            numFolds=5)
        negCrossval = CrossValidator(
            estimator=neglr,
            evaluator=negEvaluator,
            estimatorParamMaps=negParamGrid,
            numFolds=5)
        # Although crossvalidation creates its own train/test sets for
        # tuning, we still need a labeled test set, because it is not
        # accessible from the crossvalidator (argh!)
        # Split the data 50/50
        pos = results_positive_df.select("label", "features")
        neg = results_negative_df.select("label", "features")
        posTrain, posTest = pos.randomSplit([0.5, 0.5])
        negTrain, negTest = neg.randomSplit([0.5, 0.5])
        # Train the models
        print()
        print("Training positive classifier...")
        print()
        posModel = posCrossval.fit(posTrain)
        print()
        print("Training negative classifier...")
        print()
        negModel = negCrossval.fit(negTrain)

        # Once we train the models, we don't want to do it again. We can save the models and load them again later.
        posModel.save("project2/pos.model")
        negModel.save("project2/neg.model")

    # TASK 8
    strip_link_udf = udf(strip_link, StringType())
    task8_df = comments_df.select(
        "id", "created_utc", "link_id", "author_flair_text", "body")
    task8_stripped_df = task8_df.withColumn(
        "linkid", strip_link_udf("link_id"))
    task8_stripped_df = task8_stripped_df.select(
        "id", "created_utc", "linkid", "body", "author_flair_text")
    submission_title_df = submission_df.select("id", "title")
    task8_join_df = submission_title_df.alias("b").join(
        task8_stripped_df.alias("a"), task8_stripped_df.linkid == submission_title_df.id).select("a.id", "a.created_utc", "a.linkid", "a.author_flair_text", "b.title", "a.body")
    task8_join_df.show()

    # TASK 9 AND 10 STILL NEED TO BE FINISHED


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
