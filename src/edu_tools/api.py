import os

from dotenv import load_dotenv
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

from edu_tools.llms.gemini import topic_formt as llm_topic_formt

load_dotenv()

app = FastAPI()


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
