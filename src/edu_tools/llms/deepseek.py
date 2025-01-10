import os
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

ModelName = "deepseek-chat"

PROVIDE_NAME = "deepseek"
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_BASE = "https://api.deepseek.com/v1"


def deepseek_run(prompt):
    model = ChatOpenAI(
        model_name=ModelName,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.7,
        max_tokens=4096,
    )
    response = model.invoke(prompt)
    return response.content
