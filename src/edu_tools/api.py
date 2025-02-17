import os
import time
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

from edu_tools.llms.gemini import gemini_run, gemini_ocr, PROVIDE_NAME as gemini_provide
from edu_tools.llms.deepseek import (
    deepseek_run,
    PROVIDE_NAME as deepseek_provide,
)
from edu_tools.llms.context import LLMContext, OCRContext
from edu_tools.llms.prompts.math import prompt_templates, gen_prompt, exp_con_kw
from edu_tools.llms.prompts.yuwen import (
    prompt_templates as yuwen_prompt_templates,
    gen_prompt as yuwen_gen_prompt,
    exp_con_kw as yunwen_exp_con_kw,
)

from edu_tools.pb import auth_key_is_ok, fetch_key_info
from edu_tools.utils import save_base64_image
from edu_tools.llms.dify import dify_ocr, dify_math_run
from edu_tools.llms.ark import ark_run, ark_ocr, PROVIDE_NAME as ark_provide

load_dotenv()

logging.basicConfig(level=int(os.getenv("LOG_LEVEL", 20)))

log = logging.getLogger(__name__)

llm_provide = os.getenv("LLM_PROVIDE", gemini_provide)

app = FastAPI()

origins = ["*"]  # 允许所有来源，仅限开发环境


@app.middleware("http")
async def auth(request: Request, call_next):
    # 检查是否跳过 middleware
    if request.url.path in ["/llm/topic_type_list", "/user/info"]:
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
    lines = [replace_text(line) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


class Topic(BaseModel):
    text: str
    image_url: Optional[str] = None


@app.post("/llm/run/{item}")
def llm_run(item: str, ctx: LLMContext, req: Request):
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
            # text = dify_math_run(ctx, key)
            run_prompt = gen_prompt(ctx, prompt)
            text = ark_run(run_prompt, ctx)

            return {"topic": remove_empty_lines_from_string(text)}
        run_prompt = gen_prompt(ctx, prompt)
        llm_fun = gemini_run
        if llm_provide == deepseek_provide:
            llm_fun = deepseek_run
        if llm_provide == ark_provide:
            run_prompt = gen_prompt(ctx, prompt)
            # text = ark_run(run_prompt, ctx)
            # return {"topic": remove_empty_lines_from_string(text)}
        log.debug(run_prompt)
        text = llm_fun(run_prompt, ctx)
        # log.info(deepseek_math_fromat(text))
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
    text = ""
    try:
        if llm_provide == ark_provide:
            text = ark_ocr(ctx)
        else:
            tf = save_base64_image(ctx.image_data)
            text = gemini_ocr(tf)
            os.remove(tf)
    except Exception as e:
        log.error(f"ocr: {e}")

    return {"text": text}


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
