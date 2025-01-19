import os
from dotenv import load_dotenv
from pathlib import Path
from cozepy import (
    COZE_CN_BASE_URL,
    Coze,
    TokenAuth,
    Stream,
    WorkflowEvent,
    WorkflowEventType,
)

load_dotenv()
coze = Coze(auth=TokenAuth(os.getenv("COZE_API_TOKEN", "")), base_url=COZE_CN_BASE_URL)

ocr_workflow = "7452887140226662454"


def coze_ocr(file):
    f = file = coze.files.upload(file=Path(file))
    print(f)
    result = coze.workflows.runs.create(
        # id of workflow
        workflow_id=ocr_workflow,
        # params
        parameters={
            "input": f.id,
        },
    )
    print(result)
    return result
