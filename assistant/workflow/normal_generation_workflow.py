from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnablePassthrough

from assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from assistant.retriever.retrieval import Retriever
from assistant.utils.model_loader import ModelLoader
from assistant.evaluation.ragas_evaluation import evaluate_context_precision, evaluate_response_relevancy

retriever_obj = Retriever()
model_loader = ModelLoader()

def format_docs(docs) -> str:
    """ Format retrieved documents into a string for LLM input. """
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

def build_chain(query):
    """ Build the RAG pipeline chain with Retriever, prompt, LLM and parser."""
    retriever=retriever_obj.load_retriever()
    retrieved_docs=retriever.invoke(query)

    # retrieved_context=[format_docs(retrieved_docs)]

    retrieved_contexts = [format_docs(retrieved_docs)]

    llm = model_loader.load_llm()
    prompt = ChatPromptTemplate.from_template(
        PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # retrieved_context = retriever.invoke(query)
    return chain, retrieved_contexts

def invoke_chain(query: str, debug: bool = False):
    """ Run the chain with a user query."""
    chain, retrieved_context = build_chain(query)
    response = chain.invoke(query)
    return retrieved_context, response

    if debug:
        # For debugging, show docs retrieved before passing to LLM
        docs = retriever_obj.load_retriever().invoke(query)
        print(f"Retrieved documents:")
        print(format_docs(docs))
        print("\n---\n")
    
    response = chain.invoke(query)
    return retrieved_context, response

# class AgenticRAG:
#     """ Agentic RAG Pipeline using LangGraph. """
#     class AgentState(TypedDict):
#         messages: Annotated[Sequence[BaseMessage], add_messages]

#         def __init__(self):
#             self.retriever_obj = Retriever()
#             self.model_loader = ModelLoader()
#             self.llm = self.model_loader.load_llm()
#             self.workflow = self._build_workflow()
#             self.app = self.workflow.compile()
        
#         # -------------------- Helpers ----------------------
#         def format_docs(self, docs) -> str:
#             if not docs:
#                 return "No relevant documents found."
#             formatted_chunks = []
#             for d in docs:
#                 meta = d.metadata or {}
#                 formatted = (
#                     f"Title: {meta.get('product_title', 'N/A')}\n"
#                     f"Price: {meta.get('price', 'N/A')}\n"
#                     f"Rating: {meta.get('rating', 'N/A')}\n"
#                     f"Reviews:\n{d.page_content.strip()}"
#                 )
#                 formatted_chunks.append(formatted)
#             return "\n\n---\n\n".join(formatted_chunks)
        
#         # ---------------------- Nodes ------------------------
#         def _ai_assistant(self, state: AgentState):
#             print("--- CALL ASSISTANT ---")
#             messages = state["messages"]
#             last_message = messages[-1].content

#             if any(word in last_message.lower() for word in ['price', 'review', 'product']):
#                 return {"messages": [HumanMessage(content="TOOL: retriever")]}
#             else:
#                 prompt = ChatPromptTemplate.from_template(
#                     "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
#                 )
#                 chain = prompt | self.llm | StrOutputParser()
#                 response = chain.invoke({"question": last_message})
#                 return {"messages": [HumanMessage(content=response)]}
        
#         def _vector_retriever(self, state:AgentState):
#             print("--- RETRIEVER ---")
#             query = state["messages"][-1].content
#             retriever=self.retriever_obj.load_retriever()
#             docs = retriever.invoke(query)
#             context = self._format_docs(docs)
#             return {"messages": [HumanMessage(content=context)]}

if __name__ == "__main__":
    user_query = "can you suggest good budget laptop?"
    # retriever_obj = Retriever()
    # retrieved_docs = retriever_obj.call_retriever(user_query)
    
    # def format_docs(docs) -> str:
    #     if not docs:
    #         return "No relevant documents found."
    #     formatted_chunks = []
    #     for d in docs:
    #         meta = d.metadata or {}
    #         formatted = (
    #             f"Title: {meta.get('product_title', 'N/A')}\n"
    #             f"Price: {meta.get('price', 'N/A')}\n"
    #             f"Rating: {meta.get('rating', 'N/A')}\n"
    #             f"Reviews:\n{d.page_content.strip()}"
    #         )
    #         formatted_chunks.append(formatted)
    #     return "\n\n---\n\n".join(formatted_chunks)
    
    # response="Dell Inspiron 15 Intel Core I5 13th production laptop with 16GB RAM and 512GB SSD is a good budget option with positive reviews highlighting its performance and value for money."
    # print(f"Retrieved_context: {retrieved_docs}")
    # retrieved_context = [format_docs([docs]) for docs in retrieved_docs]

    # for idx, doc in enumerate(retrieved_docs, 1):
    #     print(f"Result: {idx} : {doc.page_content}\nMetadata: {doc.metadata}\n")

    retrieved_context, response = invoke_chain(user_query)
    
    context_score=evaluate_context_precision(query=user_query, response=response, retrieved_context=retrieved_context)
    relevancy_score=evaluate_response_relevancy(query=user_query, response=response, retrieved_context=retrieved_context)

    print("\n----- Evaluatio Metrics -----")
    print(f"Context Precision Score: {context_score}")
    print(f"Response Relevancy Score: {relevancy_score}")
