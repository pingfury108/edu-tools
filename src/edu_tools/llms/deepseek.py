import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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


math_fromat_prompt_template = (
    """
    # Role 我是一个专业的整理数学公样式的程序 AI 角色

    ## Workflow
    - 解析输入中被`$`前后包裹的内容
    - 如果解析后的内容中出现了 >= 2 的 `=`, 则将它替换为 `$\n=$`
    - 然后输出完整的内容

    ## Constrains
    - 不要去理解输入文本内容的含义,当中字符既可以
    - `$=$` 不能单独一行出现
    """,
    """
    """,
)


def deepseek_math_fromat(text):
    prompt_template = math_fromat_prompt_template
    template = ChatPromptTemplate.from_messages(
        [("system", prompt_template[0]), ("user", "{text}")]
    )
    prompt = template.invoke({"text": text})
    model = ChatOpenAI(
        model_name=ModelName,
        openai_api_key=API_KEY,
        openai_api_base=API_BASE,
        temperature=0.7,
        max_tokens=4096,
    )
    response = model.invoke(prompt)
    return response.content
