import json
import asyncio

from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy


# Load real RAG output
with open(
    "evaluation_sample.json",
    "r",
    encoding="utf-8"
) as file:
    data = json.load(file)


# Local evaluation LLM
ollama_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

evaluator_llm = LangchainLLMWrapper(
    ollama_llm
)


# Local embeddings
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    ollama_embeddings
)


# Evaluation sample
sample = SingleTurnSample(
    user_input=data["question"],
    response=data["answer"],
    retrieved_contexts=data["contexts"]
)


async def main():

    faithfulness = Faithfulness(
        llm=evaluator_llm
    )

    relevancy = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )


    print("\nRunning RAG evaluation...\n")


    faithfulness_score = await faithfulness.single_turn_ascore(
        sample
    )

    relevancy_score = await relevancy.single_turn_ascore(
        sample
    )


    print("Question:")
    print(data["question"])

    print("\nRAG Answer:")
    print(data["answer"])

    print("\nEvaluation Results:")
    print(f"Faithfulness: {faithfulness_score:.4f}")
    print(f"Response Relevancy: {relevancy_score:.4f}")


asyncio.run(main())