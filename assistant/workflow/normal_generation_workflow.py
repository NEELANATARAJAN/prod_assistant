from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader

class AgenticRAG:
    """ Agentic RAG Pipeline using LangGraph. """
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

        def __init__(self):
            self.retriever_obj = Retriever()
            self.model_loader = ModelLoader()
            self.llm = self.model_loader.load_llm()
            self.workflow = self._build_workflow()
            self.app = self.workflow.compile()
        
        # -------------------- Helpers ----------------------
        def format_docs(self, docs) -> str:
            if not docs:
                return "No relevant documents found."
            formatted_chunks = []
            for d in docs:
                meta = d.metadata or {}
                formatted = (
                    f"Title: {meta.get('product_title', 'N/A')}\n"
                    f"Price: {meta.get('price', 'N/A')}\n"
                    f"Rating: {meta.get('rating', 'N/A')}\n"
                    f"Reviews:\n{d.page_content.strip()}"
                )
                formatted_chunks.append(formatted)
            return "\n\n---\n\n".join(formatted_chunks)
        
        # ---------------------- Nodes ------------------------
        def _ai_assistant(self, state: AgentState):
            print("--- CALL ASSISTANT ---")
            messages = state["messages"]
            last_message = messages[-1].content

            if any(word in last_message.lower() for word in ['price', 'review', 'product']):
                return {"messages": [HumanMessage(content="TOOL: retriever")]}
            else:
                prompt = ChatPromptTemplate.from_template(
                    "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
                )
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke({"question": last_message})
                return {"messages": [HumanMessage(content=response)]}
        
        def _vector_retriever(self, state:AgentState):
            print("--- RETRIEVER ---")
            query = state["messages"][-1].content
            retriever=self.retriever_obj.load_retriever()
            docs = retriever.invoke(query)
            context = self._format_docs(docs)
            return {"messages": [HumanMessage(content=context)]}