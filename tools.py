import dotenv
import os

dotenv.load_dotenv()

from abc import ABC, abstractmethod, abstractproperty
from typing import Any
import slack_bolt
from dataclasses import dataclass


@dataclass
class Tool(ABC):

    @property
    @abstractmethod
    def name(self):
        """name for function call"""
    
    @property
    @abstractmethod
    def schema(self):
        """tool schema for OpenAI api"""
        
    @abstractmethod
    def initialize(self, *args: Any, **kwargs: Any):
        """Initialize the Tool if needed"""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any):
        """Use the tool"""


@dataclass
class SlackTool(Tool):

    @property
    def name(self):
        return "post_message_to_slack"

    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": "post_message_to_slack",
                "description": "Call this whenever you need to send a message to the user's slack channel",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to be sent",
                        },
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        }

    def initialize(self):
        self.slack_app = slack_bolt.App(
            token=os.environ.get("SLACK_BOT_TOKEN"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
        )

        channel_list = self.slack_app.client.conversations_list().data

        channel = next(
            (
                channel
                for channel in channel_list.get("channels")  # type: ignore
                if channel.get("name") == "qa_agent"
            ),
            None,
        )
        self.channel_id = channel.get("id")  # type: ignore

    def __call__(self, message):
        self.slack_app.client.chat_postMessage(
            channel=self.channel_id,
            text=message,
        )


if __name__ == "__main__":

    slack_tool = SlackTool()
    slack_tool.initialize()
    
    message = "Hello, world"
    slack_tool(message)