import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

from edu_tools.llms.gemini import gemini_run, gemini_ocr, PROVIDE_NAME as gemini_provide
from edu_tools.llms.deepseek import (
    deepseek_run,
    PROVIDE_NAME as deepseek_provide,
    deepseek_math_fromat,
)
from edu_tools.llms.context import LLMContext, OCRContext
from edu_tools.llms.prompts import prompt_templates, gen_prompt
from edu_tools.pb import auth_key_is_ok

from edu_tools.utils import save_base64_image

load_dotenv()

logging.basicConfig(level=int(os.getenv("LOG_LEVEL", 20)))

log = logging.getLogger(__name__)

llm_provide = os.getenv("LLM_PROVIDE", gemini_provide)

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
        llm_fun = gemini_run
        if llm_provide == deepseek_provide:
            llm_fun = deepseek_run
        text = llm_fun(run_prompt)
        text = deepseek_math_fromat(text)
        return {"topic": remove_empty_lines_from_string(text)}
    else:
        return {"msg": "Unsupported parameters"}


@app.post("/llm/ocr")
def ocr(ctx: OCRContext, req: Request):
    key = req.headers.get("X-Pfy-Key")
    if key:
        ok = auth_key_is_ok(key)
        if not ok:
            return {"topic": "账户已过期"}
    else:
        return {"topic": "无权访问"}
    tf = save_base64_image(ctx.image_data)
    text = gemini_ocr(tf)
    os.remove(tf)
    return {"text": text}
