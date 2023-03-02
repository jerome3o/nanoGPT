"""
Sample from a trained model
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
server = False
bot = False
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

print("Start server: ", server)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

if not server and not bot:
    # run generation
    print(start, end='')
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, decode=decode)
                print(decode(y.tolist()[0]))
                print('---------------')

    sys.exit(0)

if server:
    import fastapi as fa
    import pydantic as pyd
    import uvicorn

    class GenerateRequest(pyd.BaseModel):
        prompt: str

    class GenerateResponse(pyd.BaseModel):
        output: str

    app = fa.FastAPI()

    @app.post("/generate")
    def _generate(config: GenerateRequest):
        print(config)
        start_ids = encode(config.prompt)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, decode=decode)
                return GenerateResponse( output=decode(y.tolist()[0]))
        
    uvicorn.run(app, host="0.0.0.0", port=8004)
    sys.exit(0)


# copy over of disc bot

from discord import Message, Client
import discord
from pydantic import BaseModel

API_KEY = os.environ.get("DISCORD_API_KEY")


class PromptConfig(BaseModel):
    prompt: str
    ai_name: str
    human_name: str

config = PromptConfig(prompt="", ai_name="Jerome Swannack", human_name="FLYNN")

import discord
client = Client(intents=discord.Intents.all())

MESSAGE_BUFFER_SIZE = 5
CLEAR = "$clear"

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))


@client.event
async def on_message(message: Message):
    if message.author == client.user:
        return

    if not message.channel.name.startswith("jeromebot-"):
        return

    if message.content == CLEAR:
        return

    

    raw_messages = await message.channel.history(limit=MESSAGE_BUFFER_SIZE).flatten()
    messages = []
    for m in raw_messages:
        if m.content == CLEAR:
            break
        messages.append(m)

    s = "\n\n".join(
        f"{config.ai_name if m.author == client.user else config.human_name}: {m.content}"
        # f"{config.ai_name if m.author == client.user else m.author}: {m.content}"
        for m in messages[::-1]
    )

    prompt = f"{s}\n\n{config.ai_name}: "

    if message.content == "$prompt":
        await message.channel.send(prompt)
        return

    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


    with torch.no_grad():
        with ctx:
            resp_list = [
                model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, decode=decode)
                for k in range(10)
            ]


    resp_list = [
        remove_prefix(decode(y.tolist()[0]), prompt).strip()
        for y in resp_list
    ]

    # response = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt=prompt,
    #     temperature=0.9,
    #     max_tokens=500,
    #     presence_penalty=2,
    #     frequency_penalty=2,
    # )


    resp_list = sorted(resp_list, key=lambda x: -len(x))
    print(prompt)
    print(resp_list[0])
    for resp in resp_list:
        if resp and len(resp) < 2000:
            await message.channel.send(resp)
            return
    await message.channel.send(resp_list[:2000])


client.run(API_KEY)
