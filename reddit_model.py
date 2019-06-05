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

def get_pos_score(probability_vec):
    return (0, 1)[float(probability_vec[1]) > 0.20]

def get_neg_score(probability_vec):
    return (0, 1)[float(probability_vec[1]) > 0.25]

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
        submission_df.write.parquet("submissions.parquet")
        comments_df.write.parquet("comments.parquet")
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
        "id", "created_utc", "link_id", "author_flair_text", "body", "score")
    task8_stripped_df = task8_df.withColumn(
        "linkid", strip_link_udf("link_id"))
    task8_stripped_df = task8_stripped_df.select(
        "id", "created_utc", "linkid", "body", "author_flair_text", "score")
    submission_title_df = submission_df.select("id", "title")
    task8_join_df = submission_title_df.alias("b").join(
        task8_stripped_df.alias("a"), task8_stripped_df.linkid == submission_title_df.id).select("a.id", "a.created_utc", "a.linkid", "a.score", "a.author_flair_text", "b.title", "a.body")

    # TASK 9
    # Filter out all comments with "/s" and comments that start with "&gt"
    filtered_sarcasm_comments_df = task8_join_df.filter(
        ~(task8_join_df.body.like("%/s%"))
    )
    filtered_all_comments_df = filtered_sarcasm_comments_df.filter(
        ~(filtered_sarcasm_comments_df.body.like("&gt;%"))
    )
    # Generate all unigrams, bigrams, and trigrams and store into a column -- see TASK 4
    task9_allgrams_df = filtered_all_comments_df.withColumn("allgrams", sanitize_udf("body"))
    # Combine allgrams into one column -- see TASK 5
    task9_combinedgrams_df = task9_allgrams_df.withColumn(
        "combinedgrams", combine_udf("allgrams"))
    # Apply the countervectorizer model on the new data -- see TASK 6A
    task9_df = model.transform(task9_combinedgrams_df)

    # Apply classifier to the dataframes
    get_pos_score_udf = udf(get_pos_score, IntegerType())
    get_neg_score_udf = udf(get_neg_score, IntegerType())

    # Get positive scores
    posResult = posModel.transform(task9_df)
    posResult = posResult.drop('rawPrediction').drop('prediction')
    posResultLabeled = posResult.withColumn("is_positive", get_pos_score_udf("probability"))
    pos_result_final_df = posResultLabeled.drop('probability') # Needed so it doesn't conflict with next step, when probability is added again

    # Combine with negative scores
    withNegResults = negModel.transform(pos_result_final_df)
    withNegResults = withNegResults.drop('rawPrediction').drop('prediction')
    task9_df = withNegResults.withColumn("is_negative", get_neg_score_udf("probability"))
    results = task9_df.drop('probability').drop('allgrams').drop('combinedgrams').drop('features')


    results = results.sample(False, .05, 12345)
    # results.write.parquet("results.parquet")

    # print("Wrote to parquet")

    # Save to CSV for testing
    # results.limit(40).toPandas().to_csv("sample_data.csv", header=True)
    sqlContext.registerDataFrameAsTable(results, "results")

    # TASK 10 HERE
    # Compute the percentage of comments that were positive and the percentage 
    # of comments that were negative across all submissions/posts. 

    # task10_1_total_comments_per_link = results.limit(40).groupBy('linkid').count().show()
    # task10_1_total_pos_per_link = results.limit(40).groupBy('linkid').sum('is_positive').show()
    # task10_1_total_neg_per_link = results.limit(40).groupBy('linkid').sum('is_negative').show()
    task10_1 = sqlContext.sql('SELECT 100 * avg(is_positive) AS avg_pos, 100 * avg(is_negative) AS avg_neg FROM results')
    task10_1.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_1.csv")


        # Compute the percentage of comments that were positive and the percentage of 
        # comments that were negative across all days. Check out from from_unixtime function.
    task10_2 = sqlContext.sql('SELECT avg(is_positive) AS avg_pos, avg(is_negative) AS avg_neg, DATE(FROM_UNIXTIME(created_utc)) AS date FROM results GROUP BY date ORDER BY date')
    task10_2.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_2.csv")

        # Compute the percentage of comments that were positive and the percentage 
        # of comments that were negative across all states. There is a Python list 
        # of US States here. Just copy and paste it.
    task10_3 = sqlContext.sql('SELECT author_flair_text AS state,  100 * avg(is_positive) AS avg_pos, 100 * avg(is_negative) as avg_neg, 100 * avg(is_positive) - 100 * avg(is_negative) AS diff FROM results WHERE author_flair_text IN (\'Alabama\', \'Alaska\', \'Arizona\', \'Arkansas\', \'California\', \'Colorado\', \'Connecticut\', \'Delaware\', \'District of Columbia\', \'Florida\', \'Georgia\', \'Hawaii\', \'Idaho\' ,\'Illinois\', \'Indiana\', \'Iowa\', \'Kansas\', \'Kentucky\', \'Louisiana\', \'Maine\', \'Maryland\',\'Massachusetts\', \'Michigan\', \'Minnesota\', \'Mississippi\', \'Missouri\', \'Montana\', \'Nebraska\', \'Nevada\', \'New Hampshire\', \'New Jersey\', \'New Mexico\', \'New York\', \'North Carolina\', \'North Dakota\', \'Ohio\', \'Oklahoma\', \'Oregon\', \'Pennsylvania\', \'Rhode Island\',\'South Carolina\', \'South Dakota\', \'Tennessee\', \'Texas\', \'Utah\', \'Vermont\', \'Virginia\', \'Washington\', \'West Virginia\', \'Wisconsin\', \'Wyoming\') GROUP BY author_flair_text ORDER BY author_flair_text')
    task10_3.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_3.csv")

        # Compute the percentage of comments that were positive and the percentage of 
        # comments that were negative by comment and story score, independently. You will 
        # want to be careful about quotes. Check out the quoteAll option.
    task10_4 = sqlContext.sql('SELECT 100 * avg(is_poasitive) AS avg_pos, 100 * avg(is_negative) AS avg_neg, score FROM results GROUP BY score')
    task10_4.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("task10_4.csv")

    # Any other dimensions you compute will receive extra credit if they make sense based on the data you have.





if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
