from mcp.server.fastmcp import FastMCP
from assistant.retriever.retrieval import Retriever
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize MCP server
mcp = FastMCP("hybrid_search")

# Load retriever
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# LangChain DuckDuckGo Search Tool
duckduckgo = DuckDuckGoSearchRun()

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
               f"Title: {meta.get('title', 'N/A')}\n"
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
        docs = retriever.invoke(query)
        context = format_docs(docs)
        if context.strip() or "N/A" in context:
            return "No local results found."
        return context
    except Exception as e:
        return f"Error retrieving product info: {str(e)}"
    
@mcp.tool()
async def web_search(query: str) -> str:
    """ Search the web for additional product information using DuckDuckGo."""
    try:
        return duckduckgo.run(query)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# ----- Run MCP Server ----- #
if __name__ == "__main__":
    mcp.run(transport="stdio")
