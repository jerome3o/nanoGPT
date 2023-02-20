from pathlib import Path
from pydantic import BaseModel

_FB_DATA_ROOT = "~/source/bigdata/data/facebook/"
_BOT_NAME = "Jerome Swannack"
_END_TOKEN = ""

class Messages(BaseModel):
    sender_name: str
    timestamp_ms: int
    content: str
    type: str
    is_unsent: bool


class Conversation(BaseModel):
    participants: list[str]
    messages: list[dict]


def _get_all_message_files(root: Path):
    root = Path(root).expanduser()
    return root.glob("**/message_*.json")


def _read_conversation_from_file(file: Path):
    with open(file) as f:
        return Conversation.parse_raw(f.read())

def _get_all_conversations(root: Path):
    for file in _get_all_message_files(root):
        yield _read_conversation_from_file(file)

def _convert_conversation_to_training_text(conversation: Conversation):
    ##  construct a string like:
    # <SENDERS_NAME>: <MESSAGE_CONTENT>
    # JEROME SWANNACK: <MESSAGE_CONTENT>
    # END TOKEN

    s = ""
    for message in sorted(conversation.messages, key=lambda m: m.timestamp_ms):
        # only accept generic type messages:
        if message.type != "Generic":
            continue

        s += f"{message.sender_name}: {message.content}\n"
        if message.sender_name == _BOT_NAME:
            s += f"{_END_TOKEN}\n"


def main():
    for conversation in _get_all_conversations(_FB_DATA_ROOT):
        print(_convert_conversation_to_training_text(conversation))
        break


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()

