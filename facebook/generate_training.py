import os
import requests
import tiktoken
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from pydantic import BaseModel

_FB_DATA_ROOT = "~/source/bigdata/data/facebook/"
_BOT_NAME = "Jerome Swannack"
_END_TOKEN = "<|endoftext|>"
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
    for file in tqdm(list(_get_all_message_files(root))):
        yield _read_conversation_from_file(file)


def _convert_conversation_to_training_text(conversation: Conversation):
    ##  construct a string like:
    # <SENDERS_NAME>: <MESSAGE_CONTENT>
    # JEROME SWANNACK: <MESSAGE_CONTENT>
    # END TOKEN

    s = ""
    messages = sorted(conversation.messages, key=lambda m: m.timestamp_ms)
    for i in range(len(messages)):
        message = messages[i]
        # only accept generic type messages:
        if message.type != "Generic":
            continue

        # skip messages with no content
        if not message.content:
            continue

        # skip messages from blocked users
        if message.sender_name == _BLOCKED_USER_IN_MESSAGE:
            continue

        # construction logic
        # we want consecutive messages from the same sender to be on the same line
        # check previous message to see if it's from the same sender
        if i > 0 and messages[i - 1].sender_name == message.sender_name:
            s += f"{message.content}\n"
        else:
            # check if previous sender was bot name
            if i > 0 and messages[i - 1].sender_name == _BOT_NAME:
                # add end token to string
                s += _END_TOKEN

            s += f"\n{message.sender_name}: {message.content}\n"

    return s


def main():
    # get the first 10 conversations and print the training output

    data = ""
    for conversation in _get_all_conversations(_FB_DATA_ROOT):
        if conversation.contains_any_blocked_user():
            continue

        data += _convert_conversation_to_training_text(conversation)

    (Path(__file__).parent / "data.txt").write_text(data)

    # adapted from shakespear example 
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(Path(__file__).parent / "train.bin")
    val_ids.tofile(Path(__file__).parent / "val.bin")

    # train.bin has 301,966 tokens
    # val.bin has 36,059 tokens

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    import ipdb

    with ipdb.launch_ipdb_on_exception():
        main()
