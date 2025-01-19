import os
import opendal
from dotenv import load_dotenv

load_dotenv()

# MinIO 配置
endpoint = "https://s3.pingfury.top"
access_key = "vhKnSzy2AWFj98neIUlG"
secret_key = "6zVZMQu6CEajQ1x17ESIwMk1jDQfQU1hru3D20VP"
bucket_name = "edu-tools"


def upload_file(file):
    file_name = os.path.basename(file)
    op = opendal.Operator(
        "s3",
        bucket=bucket_name,
        endpoint=endpoint,
        access_key_id=access_key,
        secret_access_key=secret_key,
        region="us-east-1",
        # 如果是自建 MinIO，需要设置 allow_anonymous
        allow_anonymous="False",
    )
    # 打开本地文件
    with open(file, "rb") as f:
        # 上传文件到 MinIO
        op.write(f"{file_name}", f.read())

    return f"{endpoint}/{bucket_name}/{file_name}"
