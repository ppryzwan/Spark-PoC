# Spark-PoC
The Spark-PoC analyse the server-log file as they should be standarized. There is example log of http response that it reads. Then you can read 20 rows or/and save examplery plots for it by providing parameter --usage

For starter there is only basic server logs but then it can be expanded by other server logs. The most important thing is that the server logs are always very big and spark is very good in terms of very big data.

# Usage
In order to use the Apache-Spark need to be installed. 
One of initializing script in spark is:

- Only show plot

spark-submit <main.py path> --usage 1 --server_logs_path <absolute_path_to_log_file> --output_dir <path_to_directory_to_output_plots>

- Show plot and save plots

spark-submit <main.py path> --usage 2 --server_logs_path <absolute_path_to_log_file> --output_dir <path_to_directory_to_output_plots>

- Only save plots

spark-submit <main.py path> --usage 3 --server_logs_path <absolute_path_to_log_file> --output_dir <path_to_directory_to_output_plots>