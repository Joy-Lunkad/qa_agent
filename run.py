import functools
import json
import os
from typing import Any, Type
import dotenv
import rich

dotenv.load_dotenv()

from openai import OpenAI
from dataclasses import dataclass, field
from llama_index.core.storage.index_store.types import DEFAULT_PERSIST_DIR

import tools
import build_db


DEFAULT_SYSTEM_MSG = (
    "Use the provided articles delimited by triple quotes to answer questions."
    'If the answer cannot be found in the articles, write "Data Not Available"'
)

DEFAULT_PROMPT_TEMPLATE = "{retrieved_articles}" "Question: {question}"


def apply_article_template(retrieved_articles: list[str]):
    applied = ""
    for article in retrieved_articles:
        applied += f"```{article}```\n\n"
    return applied


def apply_message_template(sys_msg: str, usr_msg: str) -> Any:
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": usr_msg},
    ]


@dataclass
class QA_Agent:

    data_dir: str = "./docs"
    embedding_model: str = "text-embedding-3-large"
    model: str = "gpt-4o-mini"
    node_chunk_size = 256
    node_chunk_overlap = 20
    retrieve_top_k: int = 15
    system_message: str = DEFAULT_SYSTEM_MSG
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    LLM_tools: list[Type[tools.Tool]] = field(default_factory=lambda: [tools.SlackTool])
    persist_dir: str = DEFAULT_PERSIST_DIR

    def __post_init__(self):

        self.tools = []
        self.tool_funcs = {}
        for tool_cls in self.LLM_tools:
            tool = tool_cls()
            tool.initialize()
            self.tool_funcs[tool.name] = tool
            self.tools.append(tool.schema)

        self.OAI_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        if os.path.exists(self.persist_dir):

            print(f"Loading embedding from {self.persist_dir}")
            self.vector_db = build_db.VectorStore.from_existing_embeddings(
                OAI_client=self.OAI_client,
                embedding_model=self.embedding_model,
            )

        else:
            docstore = build_db.DocStore(
                data_dir=self.data_dir,
                chunk_size=self.node_chunk_size,
                chunk_overlap=self.node_chunk_overlap,
            )
            docstore.load_data()

            self.vector_db = build_db.VectorStore(
                nodes=docstore.nodes,
                OAI_client=self.OAI_client,
                embedding_model=self.embedding_model,
            )

            print(f"Creating and storing embeddings ...")
            self.vector_db.create_embeddings()

    def query(self, input_text: str):

        retrieved_nodes, _ = self.vector_db.query(
            text=input_text,
            top_k=self.retrieve_top_k,
        )

        retrieved_articles = apply_article_template(
            [node.text for node in retrieved_nodes]
        )

        user_message = self.prompt_template.format(
            retrieved_articles=retrieved_articles,
            question=input_text,
        )

        response = self.OAI_client.chat.completions.create(
            model=self.model,
            messages=apply_message_template(self.system_message, user_message),
            tools=self.tools,
            temperature=0.0,
            seed=42,
        )

        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls

        if tool_calls is not None:
            fn_name = tool_calls[0].function.name
            kwargs = tool_calls[0].function.arguments
            kwargs = json.loads(kwargs)
            content = kwargs["message"]

            func = self.tool_funcs.get(fn_name)

            if func is not None:
                rich.print(f"Calling {func} with kwargs: {kwargs}")
                func(**kwargs)

        return {
            "Question": input_text,
            "Answer": content,
        }

    def batch_queries(self, input_texts):

        outputs = []
        for text in input_texts:
            outputs.append(self.query(text))
        return outputs


def test():

    test_questions = [
        "What is the name of the company? Send the answer to my slack.",
        "Who is the CEO of the company?",
        "What is their vacation policy?",
        "What is the termination policy? Send the answer to my slack.",
    ]

    qa_agent = QA_Agent(data_dir="./docs")
    outputs = qa_agent.batch_queries(test_questions)

    rich.print("Outputs -> \n\n", outputs)


if __name__ == "__main__":
    test()
