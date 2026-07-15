from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from assistant.retriever.retrieval import Retriever
from assistant.utils.model_loader import ModelLoader
from assistant.evaluation.ragas_evaluation import evaluate_context_precision, evaluate_response_relevancy
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

class AgenticRAG:
    """ Agentic RAG pipeline using LangGraph + MCP Tools (Retriever + Web Search)"""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
    
    def __init__(self):
        """ Initialize components and build workflow graph."""
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        
        # Tells the compiled workflow graph to save the state after every node execution in memory.
        self.checkpointer = MemorySaver()

        # Initialize MCP tools
        self.mcp_tools = []

        # Initialize workflow graph 
        self.workflow = None

        # Initialize CompiledStateGraph app
        self.app = None

        # Initialize MCP client
        self.mcp_client = MultiServerMCPClient({
            "product_retriever": {
                "command": "python",
                "transport": "stdio",
                "args": ["/Users/neeladnatarajan/DSProjects/LLMOps/hw/prod_assistant/assistant/mcp_server/product_search_server.py"],
            }

        })
    
    @classmethod
    async def create(cls):
        """ Async factory method to create an instance of AgenticRAG() directly.
            Guarantees MCP tools are loaded before the workflow graph is built and app is compiled.
        """
        instance = cls() # runs __init__ MCP tools not loaded yet

        # Step 1: Load MCP tools asynchronously 
        try:

            instance.mcp_tools = await instance.mcp_client.get_tools()
            print("MCP Tools loaded: ", [t.name for t in instance.mcp_tools])
        except Exception as e:
            raise RuntimeError(f"Error loading MCP tools, workflow cannot be built. Error: {str(e)}")

        # Step 2: Build workflow graph only after MCP tools are loaded
        instance.workflow = instance._build_workflow()

        # Step 3: Compile the workflow graph into an executable CompiledStateGraph which validates graph structure and 
        # exposes invoke, ainvoke and stream functions
        instance.app = instance.workflow.compile(checkpointer=instance.checkpointer)

        return instance
    
    # ----- Nodes definition ----- #
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""

        if any(word in last_message.lower() for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user's query directly.\n\nQuestion: {question}\n\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message}) or "Sorry, I don't have an answer for that."
            return {"messages": [HumanMessage(content=response)]}
    
    async def _vector_retriever(self, state: AgentState):
        print("--- CALL RETRIEVER (MCP) ---")
        query = state["messages"][-1].content

        tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
        if not tool:
            return {"messages": [HumanMessage(content="Retriever tool not available in MCP tools.")]}
        
        try:
            result = await tool.ainvoke({"query": query})
            context = result or "No relevant product data found."
        except Exception as e:
            context = f"Error invoking retriever tool: {str(e)}"

        return {"messages": [HumanMessage(content=context)]}
    

    async def _web_search(self, state: AgentState):
        """ Fallback web search using MCP Tool."""

        print("--- CALL WEB SEARCH (MCP) ---")
        query = state["messages"][-1].content
        tool = next((t for t in self.mcp_tools if t.name == "web_search"), None)
        context = await tool.ainvoke({"query": query}) if tool else "Web search tool not available."
        return {"messages" : [HumanMessage(content=context)]}
    
    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        """ Grade documents based on relevancy else rewrite the query for better retrieval."""

        print("--- CALL GRADER ---")
        question = state["messages"][0].content
        retrieved_context = state["messages"][-1].content

        prompt = PromptTemplate(
            template = """ You are a grader. Question: {question} \n Docs: {docs}\n Are docs relevant to the question?
            Answer with "Yes" or "No".""",
            input_variables=["question", "docs"],
        )

        prompt = PromptTemplate(
            template="""You are a strict relevance grader.
            Question: {question}
            Retrieved Context: {context}
            Is the retrieved context SPECIFICALLY about the product mentioned in the question?
            - If the question asks about "Windows Surface" but context is about "Dell Inspiron", answer "no"
            
            - Only answer "yes" if the context directly answers the question asked
            Answer with only 'yes' or 'no'.""",
            input_variables=["question", "context"]
            )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": retrieved_context}) or "no"
        return "generator" if "yes" in score.lower() else "rewriter"
    
    def _route_after_retrieval(self, state:AgentState) -> Literal["generator", "rewriter", "web_search"]:
        """ Route after retrieval - skip grader if retrieval clearly failed. """
        context = state["messages"][-1].content

        # ByPass grader entirely if retrieval returned nothing useful
        if not context.strip() or "No local results found." in context or "Error" in context:
            print("Retrieval failed -> routing directly to web search")
            return "web_search"
        
        # Grade retrieved docs for relevance
        question = state["messages"][0].content
        prompt = PromptTemplate(
            template="""You are a strict relevance grader.

            Question: {question}
            Retrieved Context: {context}

            Is the retrieved context SPECIFICALLY about the product mentioned in the question?
            Only answer 'yes' if the context directly answers the question asked.
            If the question asks about a different product than what is in the context, answer 'no'.

            Answer with only 'yes' or 'no'.""",
        input_variables=["question", "context"]
    )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "context": context}).strip().lower()
        print(f"Relevance Score : {score}")
        return "generator" if "yes" in score else "rewriter"



    def _generate(self, state: AgentState):
        """ Generate final answer using retrieved context."""

        print("--- CALL GENERATOR ---")
        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({"question": question, "context": context}) or "Sorry, I couldn't generate an answer."
        except Exception as e:
            answer = f"Error during generation: {str(e)}"
        
        return {"messages": [HumanMessage(content=answer)]}
    
    def _rewrite(self, state: AgentState) -> str:
        """ Rewrite the query for better retrieval."""
        
        print("--- CALL REWRITER ---")
        question = state["messages"][0].content

        prompt = ChatPromptTemplate.from_template(
            "Rewrite this user query to make it more clear and specific for a search engine."
            "Do NOT answer the query. Only rewrite it.\n\nOriginal Query: {question}\nRewritten Query:"
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            new_q = chain.invoke({"question": question}).strip()
        except Exception as e:
            new_q = f"Error during query rewriting: {str(e)}"
        
        return {"messages": [HumanMessage(content=new_q)]}
    
    # ----- Workflow graph ----- #
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("WebSearch", self._web_search)

        # Workflow edges
        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant", 
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._route_after_retrieval,
            {"generator": "Generator", 
             "rewriter": "Rewriter", 
             "web_search":"WebSearch"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "WebSearch")
        workflow.add_edge("WebSearch", "Generator")

        return workflow
    
    # ----- Public Run ----- #
    async def run(self, query: str, thread_id: str="default_thread") -> str:
        """ Run the Agentic RAG workflow with MCP tools. """
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content if result and "messages" in result else "No response generated."
        
# ----- Standalone Test ----- #
if __name__ == "__main__":
    
    async def main():
        rag_agent = await AgenticRAG.create() # use async factory method to ensure MCP tools are loaded before workflow is built
        answer = await rag_agent.run("What is the reviews of iphone 17?")
        print("\nFinal Answer:", answer)
    
    asyncio.run(main())

# Final Answer: The provided text doesn't specifically mention the "iPhone 17" or its reviews. It appears to be a general discussion about iPhones, their features, and the abundance of reviews available online. If you're looking for reviews of a specific iPhone model, I'd be happy to help you find them. However, I couldn't find any information on an "iPhone 17" in the given context.
