import json
from functools import lru_cache
from typing import Generator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_rag_tool_project.agent.react_agent import ReactAgent


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户输入的问题")


class ChatResponse(BaseModel):
    answer: str


app = FastAPI(
    title="Agent RAG API",
    description="将 Agent RAG 能力包装为 HTTP API，供前端页面调用。",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_agent() -> ReactAgent:
    return ReactAgent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    answer = get_agent().execute(request.query)
    return ChatResponse(answer=answer)


def stream_chat_chunks(query: str) -> Generator[str, None, None]:
    for chunk in get_agent().execute_token_stream(query):
        payload = json.dumps({"content": chunk}, ensure_ascii=False)
        yield f"data: {payload}\n\n"

    yield "event: done\ndata: {}\n\n"


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_chat_chunks(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "agent_rag_tool_project.app2:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
    )


if __name__ == "__main__":
    main()
