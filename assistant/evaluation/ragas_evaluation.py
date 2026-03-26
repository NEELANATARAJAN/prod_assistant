from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy
import asyncio
import grpc.experimental.aio as grpc_aio
grpc_aio.init_grpc_aio()
model_loader=ModelLoader()

def evaluate_context_precision():
    """
    Evaluate the precision of the retrieved context - Are the docs actually useful?
    """
    pass

def evaluate_response_relevancy(query):
    """
    Evaluate the relevancy of the answer. Does the answer actually address the question?
    Args:
        query (str): Question raised by the user
    """
    pass