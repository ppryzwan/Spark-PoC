from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import regexp_extract
import matplotlib.pyplot as plt
import seaborn as sns
import re


def read_file(spark, file_path):
    first_line = spark.read.text(file_path).first().value
    pattern = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<ip_address>\d+\.\d+\.\d+\.\d+) (?P<http_method>\w+) (?P<url>\S+) (?P<status_code>\d+) (?P<response_size>\d+) (?P<processing_time>\d+\.\d+) "(?P<user_agent>.*)"'
    custom_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\d+\.\d+\.\d+\.\d+) (\w+) (\S+) (\d+) (\d+) (\d+\.\d+) "(.*)"'
    match = re.match(pattern, first_line)
    if match:
        captured_groups = match.groupdict()
        if len(captured_groups) == 8:
            log_df = spark.read.text(file_path)
            log_df = log_df.withColumn("timestamp",  regexp_extract("value", custom_pattern, 1)) \
                           .withColumn("ip_address", regexp_extract("value", custom_pattern, 2)) \
                           .withColumn("http_method", regexp_extract("value", custom_pattern, 3)) \
                           .withColumn("url", regexp_extract("value", custom_pattern, 4)) \
                           .withColumn("status_code", regexp_extract("value", custom_pattern, 5).cast(IntegerType()))  \
                           .withColumn("response_size", regexp_extract("value", custom_pattern, 6).cast(IntegerType())) \
                           .withColumn("processing_time", regexp_extract("value", custom_pattern, 7).cast(FloatType())) \
                           .withColumn("user_agent", regexp_extract("value", custom_pattern, 8))
            log_df = log_df.drop("value")
            return log_df
    return None


def detect_anomalies(log_df):
    assembler = VectorAssembler(
        inputCols=["status_code", "response_size", "processing_time"],
        outputCol="features"
    )

    indexer = StringIndexer(inputCol="user_agent", outputCol="user_agent_index")
    log_df = indexer.fit(log_df).transform(log_df)

    feature_vector_df = assembler.transform(log_df)

    kmeans = KMeans(k=3, seed=1)
    model = kmeans.fit(feature_vector_df)
    predictions = model.transform(feature_vector_df)

    anomalies = predictions.filter("prediction != 0")

    return anomalies


def analyze_logs_with_anomalies(log_df, anomalies_df, output_dir):
    box_plot_cols = ["status_code", "response_size", "processing_time"]

    for col_val in box_plot_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="prediction", y=col_val, data=anomalies_df.toPandas())
        plt.title(f"Box Plot for {col_val} by Anomaly Prediction")
        plt.savefig(output_dir + f"box_plot_{col_val}_anomalies.png")
        plt.close()

    anomalies_per_method = anomalies_df.groupBy("http_method").count().toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="http_method", y="count", data=anomalies_per_method)
    plt.title("Anomalies per HTTP Method")
    plt.xlabel("HTTP Method")
    plt.ylabel("Number of Anomalies")
    plt.savefig(output_dir + "anomalies_per_http_method.png")
    plt.close()
