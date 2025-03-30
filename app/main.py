from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from app.graph.workflow import build_graph

# 環境変数を読み込む
load_dotenv()

app = FastAPI(title="LangManus API", description="LangGraph AIエージェントAPI")

class Message(BaseModel):
    role: str
    content: str

class AgentRequest(BaseModel):
    messages: List[Message]

class AgentResponse(BaseModel):
    result: Dict[str, Any]

@app.post("/agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    try:
        # グラフの構築
        graph = build_graph()
        
        # リクエストメッセージをdict形式に変換
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # グラフを実行
        result = graph.invoke({
            "messages": messages
        })
        
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "LangManus AIエージェントAPIへようこそ！"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)