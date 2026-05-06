from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from assistant.retriever.retrieval import Retriever
from assistant.utils.model_loader import ModelLoader
from assistant.evaluation.ragas_evaluation import evaluate_context_precision, evaluate_response_relevancy
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

class AgenticRAG:
    """ Agentic RAG pipeline using LangGraph + MCP (Retriever + Web Search)"""

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
    

    # ----- Helper functions ----- #
    def _format_docs(self, docs):
        """ Format retrieved documents into readable context."""
        if not docs:
            return ""
        formatted_chunks = []
        for d in docs:
            meta = d.metadata
            if meta:
                formatted = (
                    f"Title: {meta.get('title','N/A')}\n"
                    f"Price: {meta.get('price','N/A')}\n"
                    f"Rating: {meta.get('rating','N/A')}\n"
                    f"Review: {d.page_content.strip()}\n"
                )
                formatted_chunks.append(formatted)
        return "\n---\n".join(formatted_chunks)
    
    # ------- Nodes ------- #
    # --- AGENT NODE --- #
    def _ai_assistant(self, state:AgentState):
        """ Initiate the AI Agent """
        print("--- CALL AI ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content

        if any(word in last_message.lower() for word in ['price', 'review', 'product']):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the question directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}
    

    # --- RETRIEVER NODE --- #
    def _vector_retriever(self, state: AgentState) -> str:
        """ Retrieve product information for a given query from vector database using MCP tool."""
        print("--- CALL VECTOR RETRIEVER ---")
        query = state["messages"][-1].content

        # Find tool by name
        retriever_tool = next(t for t in self.mcp_tools if t.name == "get_product_info")

        # Call the tool by name
        result = asyncio.run(retriever_tool.ainvoke({'query': query}))
        
        context = result if result else "No retriever data"
        return {"messages": [HumanMessage(content=context)]}
    
    # --- GENERATOR NODE --- #
    def _generate(self, state: AgentState) -> str:
        """ Generator Node will take the retrieved context and query and generate the final answer. """
        print("--- CALL GENERATOR ---")
        question = state["messages"][0].content
        retrieved_context = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question, "retrieved_context": retrieved_context})
        return {"messages": [HumanMessage(content=response)]}
    
    # --- REWRITER NODE --- #
    def _rewrite(self, state: AgentState) -> str:
        """ Rewriter Node will rewrite the user query to be more specific for better retrieval."""
        print("--- CALL REWRITER ---")
        question = state["messages"][0].content

        prompt = ChatPromptTemplate.from_template(
            """Rewrite the user query to make it more clear and specific for the search engine.
            Do Not answer the query. Only rewrite the query.\n\nQuery: {question}\n\nRewritten Query:""",
            input_variables=["question"]
        )
        chain = prompt | self.llm | StrOutputParser()
        rewritten_query = chain.invoke({"question": question})
        return {"messages": [HumanMessage(content=rewritten_query)]}
    
    # --- GRADER CONDITIONAL EDGE --- #
    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        """ Grader Node will evaluate the response from the retriever and if documents are not relevant the query is rewritten."""
        print("--- GRADER ---")
        question = state["messages"][0].content
        retrieved_context = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(
            """ You are a grader. Question: {question}\nDocs: {retrieved_context}\nAre the docs relevant to the question?" 
            Answer with 'Yes' or 'No'.""",
            input_variables=["question", "retrieved_context"]
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "retrieved_context": retrieved_context})
        return "generator" if "yes" in score.lower() else "rewriter"
    

    # ----- Build Workflow Graph ----- #
    def _build_workflow(self):
        """ Build the workflow graph by connecting the nodes with edges or conditional edges based on logic. """
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        # Add edges from START node to AI ASSISTANT node (Agent)
        workflow.add_edge(START, "Assistant")

        # Add conditional edge from AI ASSISTANT to RETRIEVER if assistant decides to use tool else use LLM response
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )

        # Add conditional edge from RETRIEVER to GENERATOR if GRADER (Conditional edge) decides retrieved docs are relevant else go to REWRITER
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )

        # Add edge to GENERATOR from RETRIEVER if retriever is successful
        workflow.add_edge("Generator", END)

        # Add edge from REWRITER back to AI ASSISTANT if retriever is not successful and query is rewritten
        workflow.add_edge("Rewriter", "Assistant")

        return workflow
    
    # ----- Run the workflow ----- #
    def run(self, query: str, thread_id: str = "default_thread") -> str:
        """ Run the workflow and return the final answer."""

        result = self.app.invoke({"messages": [HumanMessage(content=query)]}, config={"configurable": {"thread_id": thread_id}})

        return result["messages"][-1].content

# ----- Example usage ----- #
if __name__ == "__main__":
    async def main():
        rag_agent = AgenticRAG.create() # use async factory method to ensure MCP tools are loaded before workflow is built

        query = "What is the price of iphone 15?"

        answer = await rag_agent.run(query)
        print("\nFinal Answer:", answer)

    asyncio.run(main())


