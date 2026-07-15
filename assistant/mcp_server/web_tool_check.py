import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

tool = TavilySearch(max_results=5)
result = tool.invoke({"query": "Apple iPhone 17 reviews"})
print(type(result))
print(result)