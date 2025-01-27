import os
from openai import OpenAI
from dotenv import load_dotenv
from edu_tools.llms.context import LLMContext, OCRContext


load_dotenv()

PROVIDE_NAME = "ark"

API_KEY = os.getenv("ARK_API_KEY", "")
API_BASE = "https://ark.cn-beijing.volces.com/api/v3/"
model = os.getenv("ARK_MODEL")
text_model = os.getenv("ARK_TEXT_MODEL")


def ark_run(prompt, ctx: LLMContext):
    user_msg = {
        "role": "user",
        "content": [{"type": "text", "text": prompt.to_messages()[-1].content}],
    }

    mode_name = text_model

    if ctx.image_data and ctx.image_data != "":
        mode_name = model
        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.to_messages()[-1].content},
                {
                    "type": "image_url",
                    "image_url": {"url": ctx.image_data},
                },
            ],
        }

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    completion = client.chat.completions.create(
        model=mode_name,
        messages=[
            {"role": "system", "content": prompt.to_messages()[0].content},
            user_msg,
        ],
    )
    return completion.choices[0].message.content


ocr_prompt = """
#Role: 我是一个专门用于从图片中识别内容的专业 AI 角色

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


def ark_ocr(ctx: OCRContext):
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ocr_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": ctx.image_data},
                    },
                ],
            },
        ],
    )
    return completion.choices[0].message.content
