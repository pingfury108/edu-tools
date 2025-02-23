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
        response_format={"type": "text"},
    )
    return completion.choices[0].message.content


ocr_prompt = """
#Role: 我是一个专门用于从图片中识别内容的专业 AI 角色

## Goals:
- 严格逐字识别图片中可见的文字
- 保持括号内空白
- 保持原有格式和标点

## Constraints:
- 仅输出实际可见的文字
- 括号内若为空白则保持 ( )
- 不进行任何推测或补全
- 不理解或解释内容
- 不添加任何额外标点符号
- 数学表达式使用 LaTeX 格式,用 $ 包裹

## Outputs:
- 纯文本格式
- 保持原有换行
- 不使用 markdown

## Rules:
- 遇到空白处保持原样,不填充
- 遇到不完整的句子保持原样,不补全
- 严格按照原文呈现,包括标点和空格
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
