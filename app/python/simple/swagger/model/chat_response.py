from pydantic import BaseModel


class ChatResponse(BaseModel):
    answer: dict


    class Config:
        json_schema_extra = {
            "example": {
                "answer": {
                    "response": "こんにちは！今日はどんなお手伝いができますか？😊",
                    "user_happiness": 4
                }
            }
        }