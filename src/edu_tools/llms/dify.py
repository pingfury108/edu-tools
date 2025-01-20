import os
from dotenv import load_dotenv
from dify_client import CompletionClient
from dify_client.client import WorkflowClient

load_dotenv()

BASE_API = os.getenv("DIFY_BASE_API")

API_KEY = os.getenv("DIFY_OCR_APP_ID")


def dify_ocr(file, uid: str):
    workflow_client = WorkflowClient(
        API_KEY,
    )
    workflow_client.base_url = BASE_API
    workflow_response = workflow_client.run(
        inputs={
            "img": {
                "transfer_method": "local_file",
                "upload_file_id": file_upload(file_2_md(file), uid),
                "type": "image",
            }
        },
        response_mode="blocking",
        user=uid,
    )

    try:
        workflow_response.raise_for_status()
        result = workflow_response.json()
        return result.get("data", {}).get("outputs", {}).get("text")
    except Exception as e:
        print(workflow_response.text)
        raise e


def file_upload(files, uid: str):
    completion_client = CompletionClient(API_KEY)
    completion_client.base_url = BASE_API

    completion_response = completion_client.file_upload(user="uid", files=files)
    completion_response.raise_for_status()

    result = completion_response.json()
    return result.get("id")


def file_2_md(file):
    md = {"file": (os.path.basename(file), open(file, "rb"), "image/png")}
    return md
