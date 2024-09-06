import functools
import json
import os
from typing import Any
import dotenv
import rich

dotenv.load_dotenv()

from openai import OpenAI
from dataclasses import dataclass

import slack_tool
import build_db

from llama_index.core.storage.index_store.types import DEFAULT_PERSIST_DIR
from llama_index.core.schema import BaseNode


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
    embedding_model: str = "text-embedding-3-small"
    model: str = "gpt-4o-mini"
    node_chunk_size = 256
    node_chunk_overlap = 20
    retrieve_top_k: int = 15
    system_message: str = DEFAULT_SYSTEM_MSG
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE

    def __post_init__(self):

        slack_app, slack_channel_id = slack_tool.init_slack_app()
        self.post_message_to_slack = functools.partial(
            slack_tool.send_slack_message,
            slack_app=slack_app,
            channel_id=slack_channel_id,
        )
        self.tools: Any = [slack_tool.SLACK_TOOL_SCHEMA]

        self.OAI_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.nodes = self.load_embeddings()

    def load_embeddings(self) -> list[BaseNode]:

        if os.path.exists(DEFAULT_PERSIST_DIR):
            print(f"Loading embedding from {DEFAULT_PERSIST_DIR}")
            nodes = build_db.load_existing_embeddings()

        else:
            print(f"Loading Documents ...")
            nodes = build_db.load_data(
                self.data_dir,
                self.node_chunk_size,
                self.node_chunk_overlap,
            )

            print(f"Creating and storing embeddings ...")
            nodes, _ = build_db.create_embeddings(
                nodes=nodes,
                OAI_client=self.OAI_client,
                model=self.embedding_model,
                batch_n_texts=8,
            )

        return nodes

    def query(self, input_text: str):

        retrieved_nodes, _ = build_db.query(
            text=input_text,
            nodes=self.nodes,
            top_k=self.retrieve_top_k,
            OAI_client=self.OAI_client,
            model=self.embedding_model,
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

            func = getattr(self, fn_name, None)

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
