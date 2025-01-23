import os
from dotenv import load_dotenv
from dify_client import CompletionClient
from dify_client.client import WorkflowClient

from edu_tools.llms.context import LLMContext
from edu_tools.utils import save_base64_image
from edu_tools.llms.prompts import exp_con_kw

load_dotenv()

BASE_API = os.getenv("DIFY_BASE_API")

OCR_API_KEY = os.getenv("DIFY_OCR_APP_ID")
MATH_API_KEY = os.getenv("DIFY_MATH_APP_ID")


def dify_ocr(file, uid: str):
    workflow_client = WorkflowClient(
        OCR_API_KEY,
    )
    workflow_client.base_url = BASE_API
    workflow_response = workflow_client.run(
        inputs={
            "img": {
                "transfer_method": "local_file",
                "upload_file_id": file_upload(file_2_md(file), uid, OCR_API_KEY),
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


def dify_math_run(ctx: LLMContext, uid: str):
    file = save_base64_image(ctx.image_data)
    workflow_client = WorkflowClient(
        MATH_API_KEY,
    )
    workflow_client.base_url = BASE_API
    workflow_response = workflow_client.run(
        inputs={
            "topic": ctx.topic,
            "answer": ctx.answer,
            "exp_con": exp_con_kw.get(ctx.topic_type or "问答") or "",
            "img": {
                "transfer_method": "local_file",
                "upload_file_id": file_upload(file_2_md(file), uid, MATH_API_KEY),
                "type": "image",
            },
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
    finally:
        os.remove(file)


def file_upload(files, uid: str, api_key):
    completion_client = CompletionClient(api_key)
    completion_client.base_url = BASE_API

    completion_response = completion_client.file_upload(user=uid, files=files)
    completion_response.raise_for_status()

    result = completion_response.json()
    return result.get("id")


def file_2_md(file):
    md = {"file": (os.path.basename(file), open(file, "rb"), "image/png")}
    return md
