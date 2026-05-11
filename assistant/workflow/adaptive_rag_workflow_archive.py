from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

class AdaptiveRAG:
    """
    Adaptive RAG pipeline that classifies query complexity and routes accordingly:
    
    Simple   → LLM answers directly (no retrieval)
    Single   → Vector DB retrieval → Grade → Generate
    Complex  → Multi-hop: Vector DB + Web Search → Synthesize → Generate
    """

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        question: str           # original user query, never overwritten
        complexity: str         # "simple" | "single" | "complex"
        retrieval_attempts: int # track how many times retrieval has been attempted

    def __init__(self):
        from assistant.utils.model_loader import ModelLoader
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
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

    # ----- Async Factory ----- #
    @classmethod
    async def create(cls) -> "AdaptiveRAG":
        instance = cls()
        try:
            instance.mcp_tools = await instance.mcp_client.get_tools()
            print("MCP Tools loaded:", [t.name for t in instance.mcp_tools])
        except Exception as e:
            raise RuntimeError(f"Failed to load MCP tools: {str(e)}")
        instance.workflow = instance._build_workflow()
        instance.app = instance.workflow.compile(checkpointer=instance.checkpointer)
        return instance

    # ============================================================
    # NODES
    # ============================================================

    # --- NODE 1: Query Classifier --- #
    def _classify_query(self, state: AgentState):
        """
        Classifies query into:
        - simple  : general knowledge, greetings, math → LLM answers directly
        - single  : single product lookup → one vector DB retrieval
        - complex : comparison, multi-product, time-sensitive → multi-hop retrieval
        """
        print("--- CLASSIFY QUERY ---")
        question = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a query complexity classifier for a product assistant.

Classify the query into exactly one of these categories:

- simple  : General knowledge, greetings, or questions the LLM can answer without product data.
            Examples: "What is RAM?", "Hello", "What is 5G?"

- single  : Questions about a single product's price, specs, or reviews.
            Examples: "What is the price of iPhone 15?", "Samsung S25 reviews"

- complex : Comparisons, multi-product queries, budget-based recommendations, or time-sensitive queries.
            Examples: "Compare iPhone 15 vs Samsung S25", 
                      "Best phone under ₹80,000 for photography",
                      "Latest offers on Samsung today"

Query: {question}

Respond with ONLY one word: simple, single, or complex.""",
            input_variables=["question"]
        )
        chain = prompt | self.llm | StrOutputParser()
        complexity = chain.invoke({"question": question}).strip().lower()

        # Sanitize output in case LLM adds extra text
        if "simple" in complexity:
            complexity = "simple"
        elif "complex" in complexity:
            complexity = "complex"
        else:
            complexity = "single"  # default fallback

        print(f"Query complexity: {complexity}")
        return {
            "question": question,
            "complexity": complexity,
            "retrieval_attempts": 0
        }

    # --- ROUTING EDGE after classifier --- #
    def _route_by_complexity(self, state: AgentState) -> Literal["direct_answer", "vector_retriever", "multi_hop_retriever"]:
        """Route to appropriate node based on complexity."""
        complexity = state.get("complexity", "single")
        routes = {
            "simple": "direct_answer",
            "single": "vector_retriever",
            "complex": "multi_hop_retriever"
        }
        route = routes.get(complexity, "vector_retriever")
        print(f"Routing to: {route}")
        return route

    # --- NODE 2: Direct Answer (Simple queries) --- #
    def _direct_answer(self, state: AgentState):
        """Answer simple queries directly without retrieval."""
        print("--- DIRECT ANSWER (no retrieval) ---")
        question = state["question"]

        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the question directly and concisely.\n\n"
            "Question: {question}\n\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question}) or "Sorry, I couldn't answer that."
        return {"messages": [HumanMessage(content=answer)]}

    # --- NODE 3: Single-hop Vector Retriever --- #
    async def _vector_retriever(self, state: AgentState):
        """Single product lookup via MCP vector DB tool."""
        print("--- SINGLE-HOP VECTOR RETRIEVER ---")
        question = state["question"]

        tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
        if not tool:
            return {"messages": [HumanMessage(content="Retriever tool not available.")]}
        try:
            result = await tool.ainvoke({"query": question})
            context = result or "No relevant product data found."
        except Exception as e:
            context = f"Error invoking retriever: {str(e)}"

        return {
            "messages": [HumanMessage(content=context)],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
        }

    # --- NODE 4: Multi-hop Retriever (Complex queries) --- #
    async def _multi_hop_retriever(self, state: AgentState):
        """
        Multi-hop retrieval for complex queries:
        Step 1 → Vector DB for product data
        Step 2 → Web Search for live/current data
        Synthesizes both results.
        """
        print("--- MULTI-HOP RETRIEVER ---")
        question = state["question"]

        # Step 1: Vector DB
        vector_context = ""
        vector_tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
        if vector_tool:
            try:
                vector_context = await vector_tool.ainvoke({"query": question}) or ""
                print("Vector DB result obtained.")
            except Exception as e:
                vector_context = f"Vector DB error: {str(e)}"

        # Step 2: Web Search
        web_context = ""
        web_tool = next((t for t in self.mcp_tools if t.name == "web_search"), None)
        if web_tool:
            try:
                web_context = await web_tool.ainvoke({"query": question}) or ""
                print("Web search result obtained.")
            except Exception as e:
                web_context = f"Web search error: {str(e)}"

        # Synthesize both
        combined_context = f"--- Product Database ---\n{vector_context}\n\n--- Web Search ---\n{web_context}"
        return {
            "messages": [HumanMessage(content=combined_context)],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
        }

    # --- NODE 5: Retrieval Grader --- #
    def _grade_documents(self, state: AgentState) -> Literal["generate", "rewrite"]:
        """Grade retrieved docs. If irrelevant and under retry limit, rewrite query."""
        print("--- GRADE DOCUMENTS ---")
        question = state["question"]
        context = state["messages"][-1].content
        attempts = state.get("retrieval_attempts", 0)

        # Avoid infinite rewrite loops
        if attempts >= 2:
            print("Max retrieval attempts reached, proceeding to generate.")
            return "generate"

        prompt = PromptTemplate(
            template="""You are a grader. Is the retrieved context relevant to the question?
Question: {question}
Context: {context}
Answer with only 'yes' or 'no'.""",
            input_variables=["question", "context"]
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "context": context}).strip().lower()
        result = "generate" if "yes" in score else "rewrite"
        print(f"Grade: {score} → {result}")
        return result

    # --- NODE 6: Query Rewriter --- #
    def _rewrite(self, state: AgentState):
        """Rewrite query for better retrieval on retry."""
        print("--- REWRITE QUERY ---")
        question = state["question"]

        prompt = ChatPromptTemplate.from_template(
            "Rewrite this query to be more specific for a product search engine.\n"
            "Do NOT answer. Only rewrite.\n\n"
            "Original: {question}\nRewritten:"
        )
        chain = prompt | self.llm | StrOutputParser()
        rewritten = chain.invoke({"question": question}).strip()
        print(f"Rewritten query: {rewritten}")
        # Update question with rewritten version for next retrieval attempt
        return {"question": rewritten}

    # --- NODE 7: Generator --- #
    def _generate(self, state: AgentState):
        """Generate final answer using retrieved context."""
        print("--- GENERATE ANSWER ---")
        question = state["question"]
        context = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(
            "You are a helpful product assistant.\n\n"
            "Question: {question}\n\n"
            "Context: {context}\n\n"
            "Answer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            answer = chain.invoke({"question": question, "context": context}) or "Sorry, I couldn't generate an answer."
        except Exception as e:
            answer = f"Generation error: {str(e)}"
        return {"messages": [HumanMessage(content=answer)]}

    # ============================================================
    # WORKFLOW GRAPH
    # ============================================================
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)

        # Register nodes
        workflow.add_node("Classifier",         self._classify_query)
        workflow.add_node("DirectAnswer",        self._direct_answer)
        workflow.add_node("VectorRetriever",     self._vector_retriever)
        workflow.add_node("MultiHopRetriever",   self._multi_hop_retriever)
        workflow.add_node("Rewriter",            self._rewrite)
        workflow.add_node("Generator",           self._generate)

        # START → Classifier
        workflow.add_edge(START, "Classifier")

        # Classifier → route by complexity
        workflow.add_conditional_edges(
            "Classifier",
            self._route_by_complexity,
            {
                "direct_answer":        "DirectAnswer",
                "vector_retriever":     "VectorRetriever",
                "multi_hop_retriever":  "MultiHopRetriever",
            }
        )

        # Simple path
        workflow.add_edge("DirectAnswer", END)

        # Single-hop path: VectorRetriever → Grade → Generate or Rewrite
        workflow.add_conditional_edges(
            "VectorRetriever",
            self._grade_documents,
            {"generate": "Generator", "rewrite": "Rewriter"}
        )

        # Complex path: MultiHopRetriever → Grade → Generate or Rewrite
        workflow.add_conditional_edges(
            "MultiHopRetriever",
            self._grade_documents,
            {"generate": "Generator", "rewrite": "Rewriter"}
        )

        # Rewriter → retry retrieval based on original complexity
        workflow.add_conditional_edges(
            "Rewriter",
            lambda state: "MultiHopRetriever" if state.get("complexity") == "complex" else "VectorRetriever",
            {"VectorRetriever": "VectorRetriever", "MultiHopRetriever": "MultiHopRetriever"}
        )

        # Generator → END
        workflow.add_edge("Generator", END)

        return workflow

    # ============================================================
    # PUBLIC RUN
    # ============================================================
    async def run(self, query: str, thread_id: str = "default_thread") -> str:
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content if result else "No response generated."


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    async def main():
        rag = await AdaptiveRAG.create()

        queries = [
            "What is RAM?",                                          # simple
            "What is the price of iPhone 15?",                       # single
            "Compare iPhone 15 vs Samsung S25 under ₹80,000",        # complex
        ]
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            answer = await rag.run(query, thread_id=query[:20])
            print(f"Answer: {answer}")

    asyncio.run(main())