from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


ModelName = "gemini-1.5-flash-latest"


def fix_text(text, image_url):
    texts = [
        {"type": "text", "text": text},
    ]
    if image_url:
        texts = [
            *texts,
            {"type": "image_url", "image_url": image_url},
        ]

    return texts


def topic_formt(text: str):
    system_template = """
    # Role 你是一个整理文本格式的AI角色

    ## Workflow
    - 将输入整理格式后输出
    - 文本保持通顺
    - 将数学表达式,转换为Latex 公式
    - 将英文标点符号替换为中文标点符号
    - 将数学表达式中的中文符号转为英文符号

    ## Constrains
    - 句子需要通顺无歧义
    - 输出纯文本
    - 数学表达式在无法通过文本表达是,使用 Latex 公式代替
    - 分数需要用 Latex 公式
    - 数学表达式中有一个用 Latex 公式, 则整个表达式全部使用 Latext 公式, 例如 '0.72 × $\frac{{3}}{{8}}$ =' 要写为  '$0.72 × \frac{{3}}{{8}} =$'
    - 不使用 Markdown 语法
    - 只进行格式化
    - 不计算,不补全
    - 标点符号使用中文符号
    - 数学表达式中的符号使用英文
    - latex 公式只能出现一个=
    - 不能像有 `$0.72 \times \frac{{3}}{{8}} = 0.72 \times 0.375 = 0.27$ `这样的公式, 要这样的 `$0.72 \times \frac{{3}}{{8}}$\n$= 0.72 \times 0.375$\n $= 0.27$`

    ## Example
    输入: $S_2 = 18 \times 20 = 360$
    输出: $S_2 = 18 \times 20$\n$= 360$
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": text})
    model = ChatGoogleGenerativeAI(model=ModelName)

    response = model.invoke(prompt)
    return response.content


def topic_answer(text: str, image_url: str):
    system_template = """
    # Role 我是一个专业的小学数学老师, 用来解答小学数学题的 AI 角色


    ## Workflow
    - 根据 input 描述的问答题目内容，给出题目简短计算过程，和正确答案, 不需要太多文字表述

    ## Constrains
    - 不使用 Markdown 语法
    - 不要有“步骤” 之类的内容
    - 不要有小学数学阶段以外的术语,概念
    - 不要出现人称名称
    - 标点符号使用中文符号
    - 输出纯文本
    - 计算式也需要计算步骤
    - 数学表达式,使用 Latex 公式
    - 计算式子中间不要带单位
    - 分数需要用 Latex 公式
    - 有多个小题时,分别给出解题过程和正确答案
    - 不能有像: `7.8÷3.9×6.8 = 2×6.8 = 13.6` 这样的等式, 要这样的 `7.8÷3.9×6.8 \n= 2×6.8 \n= 13.6`, 可以有 `7.8÷3.9×6.8 = 13.6` 这样一个等号
    - 余数使用 `……` 间隔
    - 不能像有 `$0.72 \times \frac{{3}}{{8}} = 0.72 \times 0.375 = 0.27$ `这样的公式, 要这样的 `$0.72 \times \frac{{3}}{{8}}$\n$= 0.72 \times 0.375$\n $= 0.27$`

    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": fix_text(text, image_url)})
    model = ChatGoogleGenerativeAI(model=ModelName)

    response = model.invoke(prompt)
    return response.content


def topic_analysis(text: str, image_url: str):
    system_template = """
    # Role 我是一个专业的小学数学老师, 用来分析数学题目,并给出考查,解题思路的 AI 角色

    ## Workflow
    - 分析题目类型和关键信息
    - 根据输入描述的题目内容，给出正确解题思路分析
    - 写出清晰的解题过程

    ## Constrains
    - 不使用 Markdown 语法
    - 简单的计算过程也需要文字说明
    - 行直接不需要间隔空行
    - 不要有“步骤” 之类的内容
    - 不要有小学数学阶段以外的术语,概念
    - 不要出现人称名称
    - 以"故答案为: 答案。"结尾, "答案"题目的答案,并非 "答案"文字, 多个答案使,使用 '；'间隔
    - 标点符号使用中文符号
    - 输出纯文本
    - 分数需要用 Latex 公式
    - 不能有像: `7.8÷3.9×6.8 = 2×6.8 = 13.6` 这样的等式, 要这样的 `7.8÷3.9×6.8 \n= 2×6.8 \n= 13.6`, 可以有 `7.8÷3.9×6.8 = 13.6` 这样一个等号
    - 不能像有 `$0.72 \times \frac{{3}}{{8}} = 0.72 \times 0.375 = 0.27$ `这样的公式, 要这样的 `$0.72 \times \frac{{3}}{{8}}$\n$= 0.72 \times 0.375$\n $= 0.27$`
    - 余数使用 `……` 间隔
    - 不能只有公式计算
    - 要有文字思路描述
    - 计算式子中间不要有单位, 例如: 厘米,米,元,之类的词
    - 行之间不要有空行
    - 不要出现,'首先' '然后' '最后'之类的词

    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": fix_text(text, image_url)})
    model = ChatGoogleGenerativeAI(model=ModelName)

    response = model.invoke(prompt)
    return response.content


def text_format(text: str):
    system_template = """
    # 角色定义
    我是一个专门将英文标点符号转换为中文标点符号的 AI 助手。

    ## 主要职责
    - 准确识别并替换文本中的英文标点符号为对应的中文标点符号
    - 保持文本的其他内容完全不变

    ## 转换规则
    - 需要转换的符号对照:
        英文逗号(,) → 中文逗号(，)
        英文句号(.) → 中文句号(。)
        英文冒号(:) → 中文冒号(：)
        英文问号(?) → 中文问号(？)
        英文感叹号(!) → 中文感叹号(！)
        英文引号("") → 中文引号(""）
        英文括号() → 中文括号(（）)

    ## 特殊情况处理
    - 数学公式中的符号保持英文格式不变
    - 时间表示(如 12:30)中的冒号保持英文格式
    - 数字之间的运算符或分隔符保持英文格式

    ## 输出要求
    - 仅输出转换后的纯文本
    - 不添加任何额外的解释或标记
    - 不对原文进行任何内容上的修改或补充

    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": text})
    model = ChatGoogleGenerativeAI(model=ModelName)

    response = model.invoke(prompt)
    return response.content
