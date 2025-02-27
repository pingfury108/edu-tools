import os
import re
import time
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from collections import defaultdict, deque

from edu_tools.llms.gemini import gemini_run, gemini_ocr, PROVIDE_NAME as gemini_provide
from edu_tools.llms.deepseek import (
    deepseek_run,
    PROVIDE_NAME as deepseek_provide,
)
from edu_tools.llms.context import LLMContext, OCRContext
from edu_tools.llms.prompts.math import prompt_templates, gen_prompt, exp_con_kw
from edu_tools.llms.prompts.yuwen import prompt_templates as yuwen_prompt_templates

from edu_tools.pb import auth_key_is_ok, fetch_key_info
from edu_tools.utils import save_base64_image
from edu_tools.llms.ark import ark_run, ark_ocr, PROVIDE_NAME as ark_provide

from edu_tools.influxdb import write_log

load_dotenv()

logging.basicConfig(level=int(os.getenv("LOG_LEVEL", 20)))

log = logging.getLogger(__name__)

llm_provide = os.getenv("LLM_PROVIDE", gemini_provide)

app = FastAPI(docs_url=None, redoc_url=None)

origins = ["*"]  # 允许所有来源，仅限开发环境

# 简单的请求状态管理
_request_states = {}
_locks = {}


def cleanup_expired_locks():
    """清理过期的锁和请求状态"""
    current_time = time.time()
    expired_keys = []

    for key, timestamp in _request_states.items():
        if current_time - timestamp >= 60:
            expired_keys.append(key)

    for key in expired_keys:
        if key in _request_states:
            del _request_states[key]
        if key in _locks and not _locks[key].locked():
            del _locks[key]


# Rate limiting configuration
RATE_LIMIT_REQUESTS = 4  # Number of requests allowed
RATE_LIMIT_WINDOW = 10  # Time window in seconds
RATE_LIMIT_EXEMPT_ENDPOINTS = ["/docs", "/llm/topic_type_list", "/user/info"]

# Rate limit storage
_rate_limits = defaultdict(lambda: deque(maxlen=RATE_LIMIT_REQUESTS))


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    key = request.headers.get("x-pfy-key")

    # Skip rate limiting for certain endpoints
    if request.url.path in RATE_LIMIT_EXEMPT_ENDPOINTS:
        return await call_next(request)

    if not key:
        return JSONResponse(content={"topic": "无权访问"})

    # Get the current timestamp
    now = time.time()

    # Remove requests older than the window
    while _rate_limits[key] and _rate_limits[key][0] < now - RATE_LIMIT_WINDOW:
        _rate_limits[key].popleft()

    # Check if rate limit is exceeded
    if len(_rate_limits[key]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=200,
            content={
                "topic": "请求过于频繁，请稍后再试",
                "retry_after": int(RATE_LIMIT_WINDOW - (now - _rate_limits[key][0])),
            },
        )

    # Add current request timestamp
    _rate_limits[key].append(now)

    # Process the request
    return await call_next(request)


@app.middleware("http")
async def auth(request: Request, call_next):
    # 检查是否跳过 middleware
    if request.url.path in RATE_LIMIT_EXEMPT_ENDPOINTS:
        return await call_next(request)

    key = request.headers.get("X-Pfy-Key")
    if key:
        ok = auth_key_is_ok(key)
        if not ok:
            return JSONResponse(content={"topic": "账户已过期"})
    else:
        return JSONResponse(content={"topic": "无权访问"})

    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    log.info(
        f"Request processed - path: {request.url.path}, method: {request.method}, uid: {key}, time: {process_time:.3f}s"
    )
    try:
        write_log(p=request.url.path, key=key, process_time=process_time)
    except Exception as e:
        log.error(f"write log err: {e}")
        pass
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def replace_text(text):
    text_list = [("乘以", "乘"), ("\\(", "$"), ("\\)", "$")]
    txt = text
    for t in text_list:
        txt = txt.replace(t[0], t[1])
    return txt


def remove_empty_lines_from_string(text):
    text = convert_to_plain_text(text)
    lines = [replace_text(line) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def convert_to_plain_text(text):
    # 移除标题
    text = re.sub(r"#+\s", "", text)
    # 移除粗体和斜体
    text = re.sub(r"\*\*?(.*?)\*\*?", r"\1", text)
    # 移除代码块
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # 移除行内代码
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # 移除链接
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # 移除列表标记
    text = re.sub(r"^\s*[-*+]\s", "", text, flags=re.MULTILINE)
    return text.strip()


class Topic(BaseModel):
    text: str
    image_url: Optional[str] = None


@app.post("/llm/run/{item}")
async def llm_run(item: str, ctx: LLMContext, req: Request):
    try:
        prompt = prompt_templates.get(item)
        if ctx.discipline and (ctx.discipline == "" or ctx.discipline == "yuwen"):
            prompt = yuwen_prompt_templates.get(item)

        if prompt:
            if (
                item in ["topic_answer", "topic_analysis"]
                and ctx.image_data
                and ctx.image_data.strip() != ""
                and ctx.discipline
                and (ctx.discipline == "" or ctx.discipline != "yuwen")
            ):
                run_prompt = gen_prompt(ctx, prompt)
                text = ark_run(run_prompt, ctx)
                return {"topic": remove_empty_lines_from_string(text)}

            run_prompt = gen_prompt(ctx, prompt)
            llm_fun = gemini_run
            if llm_provide == deepseek_provide:
                llm_fun = deepseek_run
            if llm_provide == ark_provide:
                run_prompt = gen_prompt(ctx, prompt)
                text = ark_run(run_prompt, ctx)
                return {"topic": remove_empty_lines_from_string(text)}

            log.debug(run_prompt)
            text = llm_fun(run_prompt, ctx)
            return {"topic": remove_empty_lines_from_string(text)}
        else:
            return {"msg": "Unsupported parameters"}
    except Exception as e:
        log.error(f"Error in llm_run: {str(e)}")
        return {"topic": "处理请求时发生错误，请稍后重试"}


@app.post("/llm/ocr")
async def ocr(ctx: OCRContext, req: Request):
    try:
        text = ""
        if llm_provide == ark_provide:
            text = ark_ocr(ctx)
        else:
            tf = save_base64_image(ctx.image_data)
            text = gemini_ocr(tf)
            os.remove(tf)
        return {"text": text}
    except Exception as e:
        log.error(f"Error in ocr: {str(e)}")
        return {"text": "处理请求时发生错误，请稍后重试"}


@app.get("/llm/topic_type_list")
def topic_type_list(req: Request):
    return [i for i in exp_con_kw.keys()]


@app.get("/user/info")
def key_info(req: Request):
    key = req.headers.get("X-Pfy-Key")
    if key:
        data = fetch_key_info(key)
        if data:
            return {
                "text": f"账户是否过期({auth_key_is_ok(key)})",
                "exp_time": data.get("exp_time", ""),
            }
        else:
            return {"text": "账户不存在", "exp_time": ""}
    return {"exp_time": ""}
