from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import infer


# fastapi run api.py --port 5000 --reload
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class QuestionInput(BaseModel):
    question: str


@app.post("/eval/")
async def _eval(data: QuestionInput):
    try:
        res = await infer(data.question)
        return res
    except Exception as e:
        return {"answer": "", "reason": e}
