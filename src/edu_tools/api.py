import os

from dotenv import load_dotenv
from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from edu_tools.llms.gemini import (
    topic_formt as llm_topic_formt,
    topic_answer as llm_topic_answer,
    topic_analysis as llm_topic_analysis,
)

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


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Topic(BaseModel):
    text: str


@app.post("/topic/formt")
def topic_fromt(topic: Topic):
    text = llm_topic_formt(topic.text)
    print(text)
    return {"topic": text}


@app.post("/topic/answer")
def topic_answer(topic: Topic):
    text = llm_topic_answer(topic.text)
    print(text)
    return {"topic": text}


@app.post("/topic/analysis")
def topic_answer(topic: Topic):
    text = llm_topic_analysis(topic.text)
    print(text)
    return {"topic": text}
