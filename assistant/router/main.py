import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage

from contextlib import asynccontextmanager

from assistant.workflow.agentic_rag_workflow_websearch import AgenticRAG

rag_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_agent
    rag_agent = await AgenticRAG.create() # await async factory
    print("AgenticRAG initialized")
    yield


app=FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates=Jinja2Templates(directory="templates")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- FastAPI Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get", response_class=HTMLResponse)
async def chat(msg: str=Form(...)):
    """ Call the Agentic RAG workflow. """
    answer = await rag_agent.run(msg)
    print(f"Agentic Response: {answer}")
    return answer
