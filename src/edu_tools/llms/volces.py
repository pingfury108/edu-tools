import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from edu_tools.llms.s3 import upload_file

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.environ.get("ARK_API_KEY"),
    openai_api_base="https://ark.cn-beijing.volces.com/api/v3",
    model_name="ep-20250119162051-gtmz7",
)


ocr_prompt = """
#Role: 我是一个专门用于从图片中识别数学题目内容的专业 AI 角色

## Goals: 逐字识别图片中文字, 不输出其他信息

## Constrains:
- 只输出识别的文本
- 不输出其他内容
- 数学表达式使用 Latex 公式, 并使用

## outputs:
- text
- no markdown

## Workflows:
- 识别文本
- 调整符号,使用中文标点符号
"""


def doubo_orc(file):
    image_url = upload_file(file)
    text = [
        {"type": "text", "text": "识别文本"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    template = ChatPromptTemplate.from_messages(
        [("system", ocr_prompt), ("user", "{text}")]
    )
    prompt = template.invoke({"text": text})
    response = llm.invoke(prompt)
    return response.content
