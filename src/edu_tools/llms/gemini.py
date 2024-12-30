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
    - 只进行格式化
    - 不计算,不补全
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": text})
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    response = model.invoke(prompt)
    return response.content


def topic_answer(text: str):
    system_template = """
    # Role 我是一个专门用于小学数学问答题解题的 AI 角色

    ## Workflow
    - 根据 input 描述的问答题目内容，给出题目的简短解题过程，和答案

    ## Constrains
    - 不使用 Markdown 语法
    - 不要有“步骤” 之类的内容
    - 不要有小学数学阶段以外的术语,概念
    - 不要出现人称名称
    - 以"答: '正确答案'。"结尾
    - 标点符号使用中文符号
    - 输出纯文本
    - 数学表达式在无法通过文本表达是,使用 Latex 公式代替
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": text})
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    response = model.invoke(prompt)
    return response.content


def topic_analysis(text: str):
    system_template = """
    # Role 我是一个专门用于分析小学数学题解的解答 AI 角色

    ## Workflow
    - 根据输入描述的问答题目内容，给出正确的分析

    ## Constrains
    - 不使用 Markdown 语法
    - 不要有“步骤” 之类的内容
    - 不要有小学数学阶段以外的术语,概念
    - 不要出现人称名称
    - 以"故答案为: '正确答案'。"结尾
    - 标点符号使用中文符号
    - 输出纯文本
    - 数学表达式在无法通过文本表达是,使用 Latex 公式代替
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": text})
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    response = model.invoke(prompt)
    return response.content
