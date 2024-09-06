import dotenv
import os

dotenv.load_dotenv()

import numpy as np
from etils import etree
import functools
from tqdm import tqdm


from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import BaseNode

from openai import OpenAI


def load_data(data_dir, chunk_size, chunk_overlap):
    documents = SimpleDirectoryReader(data_dir).load_data()
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(documents)
    return nodes


def load_existing_embeddings() -> list[BaseNode]:
    docstore = SimpleDocumentStore.from_persist_dir()
    return list(docstore.docs.values())


def get_embeddings(
    texts: list[str],
    OAI_client: OpenAI,
    model: str,
    **kwargs,
):
    formatted_texts = [text.replace("\n", " ") for text in texts]
    response = OAI_client.embeddings.create(
        input=formatted_texts, model=model, **kwargs
    )
    return [data.embedding for data in response.data]


def create_embeddings(
    nodes: list[BaseNode],
    OAI_client: OpenAI,
    model: str,
    batch_n_texts: int = 1,
):

    get_emb_fn = functools.partial(
        get_embeddings,
        OAI_client=OAI_client,
        model=model,
    )

    # TODO: Increase rate limits for parallel_map
    # embeds = etree.parallel_map(
    #     get_emb_fn,
    #     [node.text for node in nodes],
    #     progress_bar=True,
    # )

    embeds = []
    texts = []
    for i, node in enumerate(tqdm(nodes)):
        texts.append(node.text)  # type: ignore
        if i % batch_n_texts == 0 and i:
            embeds.extend(get_emb_fn(texts))
            texts = []
    if texts:
        embeds.extend(get_emb_fn(texts))

    for node, embed in zip(nodes, embeds):
        node.embedding = embed

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist()

    return nodes, embeds


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


def query_db(
    query_embedding: list[float],
    nodes: list[BaseNode],
    top_k: int,
):
    embeddings = [node.embedding for node in nodes if node.embedding is not None]
    distances = np.array(distances_from_embeddings(query_embedding, embeddings))
    id_top_k = np.argsort(-distances)[:top_k]
    retrieved = np.array(nodes)[id_top_k]
    s_scores = distances[id_top_k]
    return retrieved, s_scores


def query(
    text: str,
    nodes: list[BaseNode],
    top_k: int,
    OAI_client: OpenAI,
    model: str,
):
    query_embedding = get_embeddings(
        texts=[text],
        OAI_client=OAI_client,
        model=model,
    )[0]
    retrieved, s_scores = query_db(query_embedding, nodes, top_k)
    return retrieved, s_scores
