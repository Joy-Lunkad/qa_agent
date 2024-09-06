import time
import dotenv
import os

dotenv.load_dotenv()

import numpy as np
from tqdm import tqdm


from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import BaseNode
from llama_index.core.storage.index_store.types import DEFAULT_PERSIST_DIR

from openai import OpenAI
from dataclasses import dataclass


@dataclass
class DocStore:
    data_dir: str
    chunk_size: int
    chunk_overlap: int

    def load_data(self):
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.nodes = parser.get_nodes_from_documents(documents)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def distances_from_embeddings(
    query_embedding: list[float],
    embeddings: list[list[float]],
) -> list[float]:
    """Return the distances between a query embedding and a list of embeddings."""
    distances = [
        cosine_similarity(query_embedding, embedding) for embedding in embeddings
    ]
    return distances


@dataclass
class VectorStore:
    nodes: list[BaseNode]
    OAI_client: OpenAI
    embedding_model: str
    batch_n_texts: int = 8
    persist_dir: str = DEFAULT_PERSIST_DIR

    @classmethod
    def from_existing_embeddings(
        cls,
        OAI_client: OpenAI,
        embedding_model: str,
        batch_n_texts: int = 8,
        persist_dir: str = DEFAULT_PERSIST_DIR,
    ):
        docstore = SimpleDocumentStore.from_persist_dir()
        nodes = list(docstore.docs.values())
        vector_db = VectorStore(
            nodes=nodes,
            OAI_client=OAI_client,
            embedding_model=embedding_model,
            batch_n_texts=batch_n_texts,
            persist_dir=persist_dir,
        )
        return vector_db

    def get_embeddings(
        self,
        texts: list[str],
        **kwargs,
    ):
        formatted_texts = [text.replace("\n", " ") for text in texts]
        response = self.OAI_client.embeddings.create(
            input=formatted_texts, model=self.embedding_model, **kwargs
        )
        return [data.embedding for data in response.data]

    def create_embeddings(self):
        embeds = []
        texts = []
        for i, node in enumerate(tqdm(self.nodes)):
            texts.append(node.text)  # type: ignore
            if i % self.batch_n_texts == 0 and i:
                embeds.extend(self.get_embeddings(texts))
                time.sleep(5)  # Rate Limit Workaround
                texts = []

        if texts:
            embeds.extend(self.get_embeddings(texts))

        for node, embed in zip(self.nodes, embeds):
            node.embedding = embed

        docstore = SimpleDocumentStore()
        docstore.add_documents(self.nodes)
        docstore.persist()

    def query_db(
        self,
        query_embedding: list[float],
        top_k: int,
    ):
        embeddings = [
            node.embedding for node in self.nodes if node.embedding is not None
        ]
        distances = np.array(distances_from_embeddings(query_embedding, embeddings))
        id_top_k = np.argsort(-distances)[:top_k]
        retrieved_nodes = np.array(self.nodes)[id_top_k]
        s_scores = distances[id_top_k]
        return retrieved_nodes, s_scores

    def query(self, text: str, top_k: int):
        query_embedding = self.get_embeddings(texts=[text])[0]
        retrieved_nodes, s_scores = self.query_db(query_embedding, top_k)
        return retrieved_nodes, s_scores
