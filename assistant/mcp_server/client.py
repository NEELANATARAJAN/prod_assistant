import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    # Initialize MCP client for the "hybrid_search" server
    client = MultiServerMCPClient({
        "hybrid_search": {
            "command": "python",
            "transport": "stdio",
            "args": ["/Users/neeladnatarajan/DSProjects/LLMOps/hw/prod_assistant/assistant/mcp_server/product_search_server.py"]
        }
    })

    # Discover tools
    tools = await client.get_tools()
    print("Available tools: ", [t.name for t in tools])

    # Pick tools by name
    retriever_tool = next(t for t in tools if t.name == "get_product_info")
    web_search_tool = next(t for t in tools if t.name == "web_search")

    # --- Step 1: Try Vector Search --- #
    query = "what is iPhone 17 reviews and price?"


    retriever_result = await retriever_tool.ainvoke({"query": query})
    retriever_text = "\n".join(
        block["text"] for block in retriever_result if block.get("type") == "text"
        )
    print("\n--- Retriever Result ---\n", retriever_text)

    # --- Step 2: Fallback to Web Search if Retriever fails --- #
    if not retriever_text.strip() or "No local results found." in retriever_text:
        print("\nNo local results, falling back to web search...")
        web_result = await web_search_tool.ainvoke({"query": query})
        web_result = "\n".join(
            block["text"] for block in web_result if block.get("type") == "text"
        )
        print(f"\n--- Web Search Result --- {web_search_tool.name}\n\n", web_result)

if __name__ == "__main__":
    asyncio.run(main())
