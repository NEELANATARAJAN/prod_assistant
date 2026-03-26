from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import os
from langchain_community.tools.tavily_search import TavilySearchResults
# from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy

retriever_obj=Retriever()
model_loader=ModelLoader()
llm=model_loader.load_llm()

class AgenticRAG:
    """Agentic RAG pipeline using LangGraph"""

    # ---------- State Definition ----------
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
    
    def __init__(self):
        self.retriver_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.workflow = self.build_workflow()
        self.web_search_tool = TavilySearchResults(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        #self.checkpointer = MemorySaver()
        
        
        #self.app = self.workflow.compile(checkpointer=self.checkpointer)


    # ---------- Helper for formatting ----------
    def format_docs(docs) -> str:
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

    # ---------- Nodes ----------
    def ai_assistant(state: AgentState):
        """Decide whether to call retriever or just answer directly."""
        print("--- CALL ASSITANT ---")
        query = state["messages"][-1].content

        # Simple routing: if query mentions product -> go retriever
        if any(word in query.lower() for word in ['price', 'review', 'product']):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            # Direct answer without retriever
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestions: {question}\nAnswer:"
            )
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"question": query})
            return {"messages": [HumanMessage(content=response)]}

    def vector_retriever(self, state: AgentState):
        """Fetch product info from vector DB."""
        print("--- RETRIEVER ---")
        query = state["messages"][-1].content
        retriever = retriever_obj.load_retriever()
        docs = retriever.invoke(query)
        context = self.format_docs(docs)
        return {"messages": [HumanMessage(content=context)]}
    
    def web_search(self, state:AgentState):
        print("--- WEB SEARCH ---")
        question = state["messages"][0].content

        # Run Web search
        results = self.web_search_tool.invoke({"query":question})

        # Format results into a single string
        web_context = "\n\n".join(
            f"Source: {r['url']}\n{r['content']}"
            for r in results
        )
        return {"messages":[HumanMessage(content=web_context)]}

    def grade_documents(state: AgentState) -> Literal["generator", "rewriter", "web_search"]:
        """Grade docs relevance."""
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer 'yes', 'no', or 'web_search' if the question needs current information. """,
            input_variables=["question", "docs"],
        )
        chain = prompt | llm | StrOutputParser()
        score = chain.invoke({"question":question, "docs": docs})
        
        if "yes" in score.lower():
            return "generator"
        elif "web_search" in score.lower():
            return "web_search"
        else:
            return "rewriter"

    def generate(state: AgentState):
        """Generate final answer with docs."""
        print("--- GENERATE ---")
        question=state["messages"][0].content
        docs=state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT]
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [HumanMessage(content=response)]}

    def rewrite(state: AgentState):
        """Rewrite bad query."""
        print("--- REWRITE ---")
        question = state["messages"][0].content
        new_q = llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)]}

    # ---------- Build workflow ----------
    def build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self.ai_assistant)
        workflow.add_node("Retriever", self.vector_retriever)
        workflow.add_node("Generator", self.generate)
        workflow.add_node("Rewriter", self.rewrite)
        workflow.add_node("WebSearch", self.web_search)

        # edges
        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END:END}
        )
        workflow.add_conditional_edges(
            "Retriever",
            self.grade_documents,
            {
                "generator": "Generator", 
                "rewriter": "Rewriter",
                "web_search": "WebSearch",
            }
        )
        workflow.add_edge("WebSearch", "Generator")
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        return workflow

        #app = workflow.compile()
    
    def run(self, query: str) -> str:
            """Run the workflow for a given query and return the final answer."""
            result = self.app.invoke({"messages": [HumanMessage(content=query)] },
                                     config={"configurable": {"thread":thread_id}}
                                     )
            return result["messages"][-1].content

# ---------- Run ----------
if __name__ == "__main__":
    rag=AgenticRAG()
    result = rag.run("What is the price of iphone 15?")
    print("\nFinal Answer:\n", result)