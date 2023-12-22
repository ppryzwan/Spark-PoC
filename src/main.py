from pyspark.sql import SparkSession
from functions.functions import read_file, detect_anomalies, analyze_logs_with_anomalies
import argparse


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Server_Log_Analysis").getOrCreate()
    parser = argparse.ArgumentParser(description="PySpark Log Analysis")
    parser.add_argument("--output_dir", type=str, help="Output directory path")
    parser.add_argument("--server_logs_path", type=str, help="Server Log Path")
    parser.add_argument("--usage", type=int, default=1, help="Server Log Path")
    args = parser.parse_args()

    output_dir = args.output_dir
    server_logs = args.server_logs_path
    usage = args.usage

    try:
        if usage == 1:
            log_df = read_file(spark, server_logs)
            log_df.show()
        elif usage == 2:
            log_df = read_file(spark, server_logs)
            log_df.show()
            anomalies_df = detect_anomalies(log_df)
            analyze_logs_with_anomalies(log_df, anomalies_df, output_dir)
        else:
            log_df = read_file(spark, server_logs)
            anomalies_df = detect_anomalies(log_df)
            analyze_logs_with_anomalies(log_df, anomalies_df, output_dir)
    except Exception as e:
        print(f"Error with reading file {e}")
