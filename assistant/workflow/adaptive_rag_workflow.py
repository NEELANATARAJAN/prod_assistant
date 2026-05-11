from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

from assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from assistant.retriever.retrieval import Retriever
from assistant.utils.model_loader import ModelLoader
from assistant.evaluation.ragas_evaluation import evaluate_context_precision, evaluate_response_relevancy
import dotenv
import os

dotenv.load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


class AdaptiveRAG:
    """
    Adaptive RAG pipeline that classifies query complexity and routes accordingly.
    Simple -> LLM answers directly (No retrieval)
    Single -> Vecto DB retrieval -> Grade -> Generate
    Complex -> Multihop Retrieval -> Vector DB -> Web Search -> Synthesize -> Generate 
    """

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        question: str
        complexity: Literal["simple", "single", "complex"]
        retrieval_attempts: int

    def __init__(self):
        """ Initialize components and workflow graph. """
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.retriever_obj = Retriever()
        self.checkpointer = MemorySaver()
        self.mcp_tools = []
        self.workflow = None
        self.app = None

        self.mcp_client = MultiServerMCPClient({
            "product_retriever": {
                "command": "python",
                "transport": "stdio",
                "args": ["/Users/neeladnatarajan/DSProjects/LLMOps/hw/prod_assistant/assistant/mcp_server/product_search_server.py"],
            }
        })
    
    # ----- Async factory method to ensure MCP tools are loaded before graph construction ----- #
    @classmethod
    async def create(cls):
        instance = cls()  # __init__ runs, but MCP tools not loaded yet
        try:
            instance.mcp_tools = await instance.mcp_client.get_tools()  # Load MCP tools asynchronously
            print("MCP Tools loaded successfully:", [t.name for t in instance.mcp_tools])
        except Exception as e:
            raise RuntimeError(f"Failed to load MCP tools: {str(e)}")
        instance.workflow = instance._build_workflow()  # Now safe to build workflow with MCP tools
        instance.app = instance.workflow.compile(checkpointer=instance.checkpointer)  # Compile the workflow graph
        return instance
    
    # ----- Nodes ------ #
    def _classify_query(self, state:AgentState):
        """
        Classifies query into:
        - Simple: Answer directly with LLM (no retrieval)
        - Single: Single product lookup -> One vector DB retrieval -> Grade -> Generate
        - Complex: Comparison, multi-product, time-sensitive -> Multihop Retrieval -> Vector DB -> Web Search -> Synthesize -> Generate
        """
        print("--- NODE: CLASSIFY QUERY ---")
        question = state["messages"][-1].content 

        prompt = PromptTemplate(
            template=""" You are a query complexity classifier for a product assistant. 
            Classify the user's question into one of three categories: 
            - Simple: General knowledge, greetings, questions the LLM can answer without product data.
                      Example: "What is RAM?", "Hello", "What is 5G?"
            - Single: Questions about a specific product that can be answered with one retrieval. 
                      Questions about price, rating, review of a single product. Example: "What is the price of iPhone 15?", "What are the reviews for Samsung Galaxy S23?"
            - Complex: Comparison questions, multi-product questions, budget-based recommendations, or time-sensitive queries that require web search.
                        Example: "Which is better, iPhone 15 or Samsung S23?", "What is the best smartphone under $500?", "What are the latest reviews for iPhone 17?"
            
            Query: {question}

            Respond with ONLY one word: simple, single or complex.
            """, input_variables=["question"]
        )
        chain = prompt | self.llm | StrOutputParser()
        complexity = chain.invoke({"question": question}).strip().lower()

        # Sanitize output in case LLM adds extra text
        if "simple" in complexity:
            complexity = "simple"
        elif "complex" in complexity:
            complexity = "complex"
        else:
            complexity = "single"
        
        print(f"Query complexity: {complexity}")
        return {
            "question": question,
            "complexity": complexity,
            "retrieval_attempts": 0,
        }
    
    # ----- ROUTING EDGE after classifier ----- #
    def _route_by_complexity(self, state: AgentState) -> Literal["direct_answer", "vector_retriever", "multi_hop_retriever"]:
        """ Route to appropriate node based on a complexity."""
        complexity = state.get("complexity", "single")
        routes = {
            "simple": "direct_answer",
            "single": "vector_retriever",
            "complex": "multi_hop_retriever"
        }
        route = routes.get(complexity, "vector_retriever")
        print(f"Routing to: {route}")
        return route
    
    # ----- NODE 2: Direct Answer (SIMPLE) ----- #
    def _direct_answer(self, state: AgentState):
        """ Answer simple queries directly without retrieval."""
        print("--- DIRECT ANSWER (no retrieval) ---")
        question = state["question"]

        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the question directly and concisely.\n\n" \
            "Question: {question}\n\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question}) or "Sorry, I couldn't answer that."
        return {"messages": [HumanMessage(content=answer)]}

    # ----- NODE 3: Vector Retriever (SINGLE) ----- #
    async def _vector_retriever(self, state: AgentState):
        """ Single product lookup via MCP vector DB tool. """
        print(f"--- SINGLE HOP VECTOR RETRIEVER ---") 
        question = state["question"]

        tool = next((t for t in self.mcp_tools if t.name == "product_retriever"), None)
        if not tool:
            return {"messages": [HumanMessage(content="Retriever tool not available.")]}
        try:
            result = await tool.ainvoke({"query": questino})
            context = result or "No relevant information found."
        except Exception as e:
            context = f"Error invoking retriever: {str(e)}"

        return {
            "messages": [HumanMessage(content=context)],
            "retrieval_attempts": state.get("retrieval_attempts",0) + 1
        }

    # ----- NODE 4: Multi-hop Retriever {COMPLEX} ----- #
    async def _multi_hop_retriever(self, state: AgentState):
        """
        Multi-hop retrieval for complex queries:
        Step 1 -> Vector DB for product data
        Step 2 -> Web Search for live/current data
        Synthesize both results.
        """
        print("--- MULTI-HOP RETRIEVER ---")
        question = state["question"]

        # Step 1: Vector DB 
        vector_context = "" 
        vector_tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
        if vector_tool:
            try:
                vector_context = await vector_tool.ainvoke({"query": question}) or ""
                print("Vector DB context retrieved.")
            except Exception as e:
                vector_context = f"Error invoking vector retriever: {str(e)}"
        
        # Step 2: Web Search 
        web_context = ""
        web_tool = next(( t for t in self.mcp_tools if t.name == "web_search"), None)
        if web_tool:
            try:
                web_context = await web_tool.ainvoke({"query": question}) or ""
                print("Web search result obtained.")
            except Exception as e:
                web_context = f"Error invoking web search: {str(e)}"
        
        # Synthesize contexts
        combined_context = f"---- Product Database ---\n{vector_context}\n\n---- Web Search -----\n{web_context}"
        return {
            "messages": [HumanMessage(content=combined_context)],
            "retrieval_attempts": state.get("retrieval_attempts",0) + 1
        }
    
    # ----- NODE 5: Retrieval Grader ----- #
    def _grade_retrieval(self, state: AgentState):
        """ Grade retrieved docs. If irrelevant and under retry limit, rewrite query."""
        print("--- GRADING RETRIEVAL ---")
        question = state["question"]
        context = state["messages"][-1].content
        attempts = state.get("retrieval_attempts", 0)

        # Avoid infinite rewrite loops
        if attempts >= 2:
            print("Max retrieval attempts reached. Proceeding with current context.")
            return "generate"

        prompt = PromptTemplate(
            template=""" You are a grader. Is the retrieved context relevant and sufficient to answer the question?
            Question: {question}
            Context: {context}
            Answer with 'yes' or 'no'. """, input_variables=["question", "context"] 
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "context": context}).strip().lower()
        result = "generate" if "yes" in score else "rewrite"
        print(f"Grade: {score} -> {result}")
        return result

    # ----- NODE 6: Query Rewriter ----- #
    def _rewrite_query(self, state: AgentState):
        """ Rewrite query to be more specific based on retrieved context."""
        print("--- REWRITE QUERY ---")
        question = state["question"]

        prompt = ChatPromptTemplate.from_template(
            "Rewrite this query to be more specific for a product search engine.\n"
            "Do NOT answer. Only rewrite.\n\n"
            "Original Query: {question}\n\nRewritten Query:"
        )
        chain = prompt | self.llm | StrOutputParser()
        rewritten = chain.invoke({"question": question}).strip()
        print(f"Rewritten query: {rewritten}")
        return {"question": rewritten}
    
    # ----- NODE 7: Final Answer Generation ----- #
    def _generate(self, state: AgentState):
        """ Generate final answer using retrieved context if available."""
        print("--- GENERATE FINAL ANSWER ---")
        question = state["question"]
        context = state["messages"][-1].content if state["messages"] else ""

        prompt = ChatPromptTemplate.from_template(
            """ You are a helpful assistant. \n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            answer = chain.invoke({"question": question, "context": context}) or "Sorry, I couldn't generate an answer on that."
        except Exception as e:
            answer = f"Generation error: {str(e)}"
        return {"messages": [HumanMessage(content=answer)]}
    
    # ----- Workflow Graph Construction ----- #
    def _build_workflow(self):
        """ Construct the workflow graph with nodes and routing logic. """
        workflow = StateGraph(self.AgentState)

        # Register nodes
        workflow.add_node("Classifier", self._classify_query)
        workflow.add_node("DirectAnswer", self._direct_answer)
        workflow.add_node("VectorRetriever", self._vector_retriever)
        workflow.add_node("MultiHopRetriever", self._multi_hop_retriever)
        workflow.add_node("RewriteQuery", self._rewrite_query)
        workflow.add_node("Generator", self._generate)

        # START -> Classifier
        workflow.add_edge(START, "Classifier")

        # Classifier -> Routing based on complexity
        workflow.add_conditional_edges(
            "Classifier",
            self._route_by_complexity,
            {
                "direct_answer": "DirectAnswer",
                "vector_retriever": "VectorRetriever",
                "multi_hop_retriever": "MultiHopRetriever",
            }
        )

        # Simple path: DirectAnswer -> END
        workflow.add_edge("DirectAnswer", END)

        # Single Hop path: VectorRetriever -> Grade -> (RewriteQuery or Generator)
        workflow.add_edge("VectorRetriever", self._grade_retrieval, {"generate": "Generator", "rewrite": "RewriteQuery"})
        
        # Complex path: MultiHopRetriever -> Grade -> (RewriteQuery or Generator)
        workflow.add_conditional_edges(
            "MultiHopRetriever",
            self._grade_retrieval,
            {"generate": "Generator", "rewrite": "RewriteQuery"}
        )

        # Rewriter -> retrievers (back to same retriever for another attempt)
        workflow.add_conditional_edges(
            "Rewriter",
            lambda state: "MultiHopRetriever" if state["complexity"] == "complex" else "VectorRetriever",
            {"VectorRetriever": "VectorRetriever", "MultiHopRetriever": "MultiHopRetriever"}
        )

        # Generator -> END
        workflow.add_edge("Generator", END)

        return workflow
    
    # ----- Public run ----- #
    async def run(self, query: str, thread_id: str = "default_thread") -> str:
        """ Run the workflow with the given query and return the final answer. """
        result = await self.app.ainvoke({"messages": [HumanMessage(content=query)]}, config={"configurable" : {"thread_id"=thread_id}})
        return result["messages"][-1].content if result else "No response generated."

# ----- Example usage ----- #
if __name__ == "__main__":
    async def main():
        rag = await AdaptiveRAG.create()

        queries = [
            "What is RAM?",
            "What is the price of iphone 15?",
            "Compare iphone 15 and samsung s23.",
        ]

        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            answer = await rag.run(query, thread_id=query[:20])
            print(f"Answer: {answer}")
        
    asyncio.run(main())
