from typing import Optional
from pydantic import BaseModel


class LLMContext(BaseModel):
    topic: str
    topic_type: Optional[str] = None
    image_url: Optional[str] = None
    answer: Optional[str] = None
    analysis: Optional[str] = None
    image_data: Optional[str] = None


class OCRContext(BaseModel):
    image_data: str
