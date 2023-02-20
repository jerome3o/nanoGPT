from pathlib import Path
from pydantic import BaseModel

_FB_DATA_ROOT = "~/source/bigdata/data/facebook/"
_BOT_NAME = "Jerome Swannack"
_END_TOKEN = ""
_BLOCKED_USER = "Facebook User" 
_BLOCKED_USER_IN_MESSAGE = "Other user"

class Message(BaseModel):
    sender_name: str
    timestamp_ms: int
    content: str = None
    type: str
    is_unsent: bool


class User(BaseModel):
    name: str

class Conversation(BaseModel):
    participants: list[User]
    messages: list[Message]

    def contains_any_blocked_user(self):
        # check if any participant is "Facebook User"
        return any(p.name == _BLOCKED_USER for p in self.participants)


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

        # skip messages with no content
        if not message.content:
            continue

        # skip messages from blocked users
        if message.sender_name == _BLOCKED_USER_IN_MESSAGE:
            continue

        s += f"{message.sender_name}: {message.content}\n"
        if message.sender_name == _BOT_NAME:
            s += f"{_END_TOKEN}\n"

    return s


def main():
    # get the first 10 conversations and print the training output

    i = 0
    for conversation in _get_all_conversations(_FB_DATA_ROOT):
        if conversation.contains_any_blocked_user():
            continue

        print(_convert_conversation_to_training_text(conversation))
        i += 1
        if i == 10:
            break



if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main()

