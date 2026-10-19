from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="本地文件助手 Agent API")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def root():
    return {"status": "ok", "message": "Agent API 正在运行"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = run_agent(req.message, verbose=False)
    return ChatResponse(reply=reply)