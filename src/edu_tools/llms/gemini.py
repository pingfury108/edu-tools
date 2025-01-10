from langchain_google_genai import ChatGoogleGenerativeAI


ModelName = "gemini-1.5-flash-latest"
PROVIDE_NAME = "gemini"


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


def gemini_run(prompt):
    model = ChatGoogleGenerativeAI(model=ModelName)

    response = model.invoke(prompt)
    return response.content
