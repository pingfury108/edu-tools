import os
import influxdb_client
import logging
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

bucket = os.getenv("LOG_DB_BUCKET", "edu-tools")
org = os.getenv("LOG_DB_ORG", "docs")
token = os.getenv("LOG_DB_TOKEN", "")
# Store the URL of your InfluxDB instance
url = os.getenv("LOG_DB_API", "http://192.168.31.95:8086")


def write_log(p, key, process_time):
    client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api()
    p = (
        influxdb_client.Point("llm")
        .tag("key", key)
        .tag("url", p)
        .field("process_time", process_time)
    )
    write_api.write(bucket=bucket, org=org, record=p)

    return
