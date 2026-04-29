from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from assistant.retriever.retrieval import Retriever
from assistant.utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from assistant.evaluation.ragas_evaluation import evaluate_context_precision, evaluate_response_relevancy
from IPython.display import Image


retriever_obj=Retriever()
model_loader=ModelLoader()
llm=model_loader.load_llm()

class AgenticRAG:
    """Agentic RAG pipeline using LangGraph"""

    # ---------- State Definition ----------
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        documents: list
    
    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.workflow = self.build_workflow()
        self.web_search_tool = TavilySearchResults(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        #self.checkpointer = MemorySaver()
        
        
        #self.app = self.workflow.compile(checkpointer=self.checkpointer)


    # ---------- Helper for formatting ----------
    def _format_docs(self, docs) -> str:
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
    def _ai_assistant(self, state: AgentState):
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
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": query})
            return {"messages": [HumanMessage(content=response)]}

    def _vector_retriever(self, state: AgentState):
        """Fetch product info from vector DB."""
        print("--- RETRIEVER ---")
        query = state["messages"][0].content
        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)
        # context = self._format_docs(docs)
        return {"documents": docs}
    
    def _web_search(self, state:AgentState):
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

    # def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter", "web_search"]:
    #     """Grade docs relevance."""
    #     print("--- GRADER ---")
    #     question = state["messages"][0].content
    #     docs = state["messages"][-1].content

    #     prompt = PromptTemplate(
    #         template="""You are a grader. Question: {question}\nDocs: {docs}\n
    #         Are docs relevant to the question? Answer 'yes', 'no', or 'web_search' if the question needs current information. """,
    #         input_variables=["question", "docs"],
    #     )
    #     chain = prompt | self.llm | StrOutputParser()
    #     score = chain.invoke({"question":question, "docs": docs})
        
    #     if "yes" in score.lower():
    #         return "generator"
    #     elif "web_search" in score.lower():
    #         return "web_search"
    #     else:
    #         return "rewriter"

    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter", "web_search"]:
        """Grade docs relevance."""
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state.get("documents", [])

        if not docs:
            print("No documents retrieved, defaulting to web search.")
            return "web_search"

        prompt = PromptTemplate(
            template=""" You are a document relevance grader.
            Question: {question}
            Document: {docs}
            Does this document contain information relevant to the question?
            Reply with ONE word only: yes or no
            """, input_variables=["question", "docs"],
        )    
        chain = prompt | self.llm | StrOutputParser()
        relevant_docs = []
        for doc in docs:
            score = chain.invoke({
                "question": question,
                "docs": doc.page_content
            })
            print(f" score='{score.strip().lower()}' doc='{doc.page_content[:100]}...'")
            if "yes" in score.strip().lower():
                relevant_docs.append(doc)
        if relevant_docs:
            # Update documents with only relevant ones, format for generator
            state["documents"] = relevant_docs
            print(f"{len(relevant_docs)} relevant docs -> generator")
            return "generator"
        else:
            # Docs were retrieved but not relevant -> web search
            print("Docs irrelevant to question -> web search")
            return "web_search"

    def _generate(self, state: AgentState):
        """Generate final answer with docs."""
        print("--- GENERATE ---")
        question=state["messages"][0].content
        docs=state.get("documents", [])

        # Use formatted docs if available, else fallback to last message (web search results)
        if docs:
            context = self._format_docs(docs)
        else:
            context = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        return {"messages": [HumanMessage(content=response)]}

    def _rewrite(self, state: AgentState):
        """Rewrite bad query."""
        print("--- REWRITE ---")
        question = state["messages"][0].content
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)]}

    # ---------- Build workflow ----------
    def build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("WebSearch", self._web_search)

        # edges
        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END:END}
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
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
    
    def run(self, query: str, thread_id: str = "default_thread") -> str:
            """Run the workflow for a given query and return the final answer."""
            result = self.app.invoke({"messages": [HumanMessage(content=query)] },
                                     config={"configurable": {"thread_id":thread_id}}
                                     )
            print(f"Full conversation history for thread {thread_id}:")
            for msg in result["messages"]:
                print(f"{msg.__class__.__name__}: {msg.content}")
            print(self.app.get_graph().draw_mermaid())
            return result["messages"][-1].content
    
    def run_with_evaluation(self, query: str, thread_id: str = "default_thread") -> tuple[str, list[str]]:
        """Run the workflow and return both the final answer and retrieved context for evaluation.

           Returns:
                response (str): Final answer generated by the model.
                retrieved_context (list[str]): page_content of each retrieved doc
                                               (web_search content if DB missed) 
        """
        result = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )

        response = result["messages"][-1].content
        docs = result.get("documents", [])

        if docs:
            retrieved_context = [doc.page_content for doc in docs]
            print(f"Length of Retrieved context from vector DB: {len(retrieved_context)}")
        else:
            retrieved_context = (
                [result["messages"][-2].content] if len(result["messages"]) >= 2 else []
            )
            print(f"Length of Retrieved context from web search: {len(retrieved_context)}")
        
        return response, retrieved_context

# ---------- Run ----------
if __name__ == "__main__":
    rag=AgenticRAG()
    result = rag.run("What is the review of samsung s25?")
    print("\nFinal Answer:\n", result)
    user_query = "What is the review of samsung s25?"
    response, retrieved_context = rag.run_with_evaluation(user_query)

    context_score=evaluate_context_precision(query=user_query, response=response, retrieved_context=retrieved_context)
    relevancy_score=evaluate_response_relevancy(query=user_query, response=response, retrieved_context=retrieved_context)

    print("\n----- Evaluatio Metrics -----")
    print(f"Context Precision Score: {context_score}")
    print(f"Response Relevancy Score: {relevancy_score}")
