import os
from openai import OpenAI
from dotenv import load_dotenv
from edu_tools.llms.context import LLMContext


load_dotenv()

API_KEY = os.getenv("ARK_API_KEY", "")
API_BASE = "https://ark.cn-beijing.volces.com/api/v3/"
model = os.getenv("ARK_MODEL")


def ark_run(prompt, ctx: LLMContext):
    user_msg = {
        "role": "user",
        "content": [{"type": "text", "text": prompt.to_messages()[-1].content}],
    }

    if ctx.image_data != "":
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
        model=model,
        messages=[
            {"role": "system", "content": prompt.to_messages()[0].content},
            user_msg,
        ],
    )
    return completion.choices[0].message.content
