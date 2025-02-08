import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

ModelName = "gemini-2.0-flash"
PROVIDE_NAME = "gemini"


def gemini_run(prompt):
    model = ChatGoogleGenerativeAI(model=ModelName)

    response = model.invoke(prompt)
    return response.content


ocr_prompt = """
#Role: 我是一个专门用于从图片中识别数学题目内容的专业 AI 角色

## Goals: 逐字识别图片中文字, 不输出其他信息

## Constrains:
- 只输出识别的文本
- 不输出其他内容
- 数学表达式使用 Latex 公式, 并使用 $ 将公式前后包裹

## outputs:
- text
- no markdown

## Workflows:
- 识别文本
- 调整符号,使用中文标点符号
"""


def gemini_ocr(file):
    img_file = genai.upload_file(path=file)
    file = genai.get_file(name=img_file.name)
    model = genai.GenerativeModel(model_name=ModelName)

    # Prompt the model with text and the previously uploaded image.
    resp = model.generate_content([img_file, ocr_prompt])

    return resp.text
