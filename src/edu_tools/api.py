from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

from edu_tools.llms.gemini import (
    topic_formt as llm_topic_formt,
    topic_answer as llm_topic_answer,
    topic_analysis as llm_topic_analysis,
    text_format as llm_text_format,
)
from edu_tools.llms.context import LLMContext
from edu_tools.llms.prompts import prompt_templates, gen_prompt

load_dotenv()

app = FastAPI()

origins = ["*"]  # 允许所有来源，仅限开发环境

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def remove_empty_lines_from_string(text):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Topic(BaseModel):
    text: str
    image_url: Optional[str] = None


@app.post("/topic/formt")
def topic_fromt(topic: Topic):
    text = llm_topic_formt(topic.text)
    print(text)
    return {"topic": remove_empty_lines_from_string(text)}


@app.post("/topic/answer")
def topic_answer(topic: Topic):
    text = llm_topic_answer(topic.text, topic.image_url or "")
    print(text)
    return {"topic": remove_empty_lines_from_string(text)}


@app.post("/topic/analysis")
def topic_analysis(topic: Topic):
    text = llm_topic_analysis(topic.text, topic.image_url or "")
    print(text)
    return {"topic": remove_empty_lines_from_string(text)}


@app.post("/text/format")
def text_format(topic: Topic):
    text = llm_text_format(topic.text)
    print(text)
    return {"topic": text}


@app.post("/llm/run/{item}")
def llm_run(item: str, ctx: LLMContext):
    prompt = prompt_templates.get(item)
    if prompt:
        text = gen_prompt(ctx, prompt)
        print(text)
        return {"topic": remove_empty_lines_from_string(text)}
    else:
        return {"msg": "Unsupported parameters"}
