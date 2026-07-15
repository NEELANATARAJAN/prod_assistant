from mcp.server.fastmcp import FastMCP
from assistant.retriever.retrieval import Retriever
from langchain_community.tools import DuckDuckGoSearchRun
import os
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
print(f"\n\nTAVILY API KEY: {os.getenv('TAVILY_API_KEY')}\n\n")

# Initialize MCP server
mcp = FastMCP("hybrid_search")

# Load retriever
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# LangChain DuckDuckGo Search Tool
# duckduckgo = DuckDuckGoSearchRun()
# tavilysearch = TavilySearchResults(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
web_search_tool = TavilySearch(max_results=5, tavily_api_key=os.getenv("TAVILY_API_KEY"))

# ----- Helper functions ----- #
def format_docs(docs):
    """Format retrieved documents into readable context."""
    if not docs:
        return ""
    formatted_chunks = []
    for d in docs:
        meta = d.metadata
        if meta:
            formatted = (
               f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Review: {d.page_content.strip()}\n"
            )
            formatted_chunks.append(formatted)
    return "\n---\n".join(formatted_chunks)

# ----- MCP Endpoints ----- #

@mcp.tool()
async def get_product_info(query: str) -> str:
    """ Retrieve product information for a given query from local retriever."""
    try:
        docs = retriever_obj.call_retriever(query)
        context = format_docs(docs)
        if not context.strip():
            return "No local results found."
        return context.strip()
    except Exception as e:
        return f"Error retrieving product info: {str(e)}"
    
@mcp.tool()
async def web_search(query: str) -> str:
    """ Search the web for additional product information using TavilySearch."""
    try:
        # results =  duckduckgo.run(query)
        # return results
        results =  web_search_tool.invoke({"query": query})
        print(f"[WEB SEARCH] Raw Type: {type(results)}, count: {len(results) if isinstance(results, list) else 'N/A'}")
        
        if not results:
            return "No web results found."
        
        if isinstance(results, dict):
            items = results.get("results", [])
            if not items:
                return "No web results found."
            return "\n\n".join(
                f"Source: {r.get('url','N/A')}\nContent: {r.get('content', 'N/A')}"
                for r in items
                if isinstance(r, dict) and r.get("content")
            )

        if isinstance(results, str):
            return results
        
        if isinstance(results, list):
            web_context = "\n\n".join(
                f"Source: {res.get('url','N/A')}\nContent: {res.get('content', 'N/A')}"
                for res in results
                if isinstance(res, dict) and res.get('content')
            )
            return web_context or "No web results found."
        
        return "No web results found."
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# ----- Run MCP Server ----- #
if __name__ == "__main__":
    mcp.run(transport="stdio")