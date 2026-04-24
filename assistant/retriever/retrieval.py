import os
import sys
from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from assistant.utils.config_loader import load_config
from assistant.utils.model_loader import ModelLoader
from dotenv import load_dotenv
from pathlib import Path
import math
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from assistant.evaluation.ragas_evaluation import evaluate_context_precision, evaluate_response_relevancy


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class Retriever:
    def __init__(self):
        """ Initialize Retriever class """
        self.model_loader=ModelLoader()
        self.config=load_config()
        self._load_env_variables()
        self.vstore=None
        self.retriever=None

    def _load_env_variables(self):
        """ Load env variables """
        load_dotenv()
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]

        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
    
    def _sanitize_value(self, value):
        """ Replace nan/inf floats with None - JSON does not support them."""
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    
    def _sanitize_metadata(self, metadata: dict) -> dict:
        """ Sanitize all metadata values recursively. """
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                cleaned[key] = self._sanitize_metadata(value)
            elif isinstance(value, list):
                cleaned[key] = [self._sanitize_value(v) for v in value]
            else:
                cleaned[key] = self._sanitize_value(value)
        return cleaned
    
    def _sanitize_docs(self, docs: list[Document]) -> list[Document]:
        """Sanitize metadata for every retrieved document. """
        for doc in docs:
            doc.metadata = self._sanitize_metadata(doc.metadata or {})
        return docs

    def load_retriever(self):
        """ Load Retriever for the Astra DB VectorStore """
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]

            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
            )
        
        if not self.retriever:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            #retriever=self.vstore.as_retriever(search_kwargs={"k": top_k})

            mmr_retriever = self.vstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k,
                               "fetch_k": 20,
                               "lambda_mult": 0.7,
                               "score_threshold": 0.6
                               })
            print(f"Retriever loaded successfully")

            llm = self.model_loader.load_llm()

            compressor = LLMChainFilter.from_llm(llm)

            self.retriever_instance = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=mmr_retriever
            )

            print(f"[load_retriever] Retriever loaded successfully")
            return self.retriever_instance

    def call_retriever(self, query):
        """ Call retriever and invoke the query """
        retriever=self.load_retriever()
        output=retriever.invoke(query)
        return output

if __name__ == "__main__":
    user_query = "can you suggest good budget laptop?"
    retriever_obj = Retriever()
    retrieved_docs = retriever_obj.call_retriever(user_query)
    
    def format_docs(docs) -> str:
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
    
    response="Dell Inspiron 15 Intel Core I5 13th production laptop with 16GB RAM and 512GB SSD is a good budget option with positive reviews highlighting its performance and value for money."
    print(f"Retrieved_context: {retrieved_docs}")
    retrieved_context = [format_docs([docs]) for docs in retrieved_docs]

    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"Result: {idx} : {doc.page_content}\nMetadata: {doc.metadata}\n")
    
    context_score=evaluate_context_precision(query=user_query, response=response, retrieved_context=retrieved_context)
    relevancy_score=evaluate_response_relevancy(query=user_query, response=response, retrieved_context=retrieved_context)

    print("\n----- Evaluatio Metrics -----")
    print(f"Context Precision Score: {context_score}")
    print(f"Response Relevancy Score: {relevancy_score}")

