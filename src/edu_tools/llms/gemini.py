from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


def topic_formt(text: str):
    system_template = """
    # Role 你是一个整理文本格式的AI角色

    ## Workflow
    - 将输入整理格式后输出
    - 文本保持通顺

    ## Constrains
    - 句子需要通顺无歧义
    - 输出纯文本
    - 数学表达式在无法通过文本表达是,使用 Latex 公式代替
    - 不使用 Markdown 语法
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": text})
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    response = model.invoke(prompt)
    return response.content
