import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

from edu_tools.llms.gemini import (
    text_format as llm_text_format,
    gemini_run,
)
from edu_tools.llms.context import LLMContext
from edu_tools.llms.prompts import prompt_templates, gen_prompt
from edu_tools.pb import auth_key_is_ok

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)

app = FastAPI()

origins = ["*"]  # 允许所有来源，仅限开发环境

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def replace_text(text):
    text_list = [("乘以", "乘"), ("除以", "除")]
    txt = text
    for t in text_list:
        txt = txt.replace(t[0], t[1])
    return txt


def remove_empty_lines_from_string(text):
    lines = [replace_text(line) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


class Topic(BaseModel):
    text: str
    image_url: Optional[str] = None


@app.post("/text/format")
def text_format(topic: Topic):
    text = llm_text_format(topic.text)
    return {"topic": text}


@app.post("/llm/run/{item}")
def llm_run(item: str, ctx: LLMContext, req: Request):
    key = req.headers.get("X-Pfy-Key")
    if key:
        ok = auth_key_is_ok(key)
        if not ok:
            return {"topic": "账户已过期"}
    else:
        return {"topic": "无权访问"}
    prompt = prompt_templates.get(item)
    if prompt:
        run_prompt = gen_prompt(ctx, prompt)
        text = gemini_run(run_prompt)
        return {"topic": remove_empty_lines_from_string(text)}
    else:
        return {"msg": "Unsupported parameters"}
