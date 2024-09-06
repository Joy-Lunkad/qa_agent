import dotenv
import os

dotenv.load_dotenv()

from typing import Any
import slack_bolt


def init_slack_app():
    slack_app = slack_bolt.App(
        token=os.environ.get("SLACK_BOT_TOKEN"),
        signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    )

    channel_list = slack_app.client.conversations_list().data

    channel = next(
        (
            channel
            for channel in channel_list.get("channels")  # type: ignore
            if channel.get("name") == "qa_agent"
        ),
        None,
    )
    channel_id = channel.get("id")  # type: ignore

    return slack_app, channel_id


def send_slack_message(message: str, slack_app: slack_bolt.App, channel_id: Any):
    slack_app.client.chat_postMessage(channel=channel_id, text=message)


SLACK_TOOL_SCHEMA = {
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

if __name__ == "__main__":

    slack_app, slack_channel_id = init_slack_app()
    message = "Hello, world"
    send_slack_message(message, slack_app, slack_channel_id)
