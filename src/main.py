import asyncio
import logging
import os
import sys

import openai
import promptlayer
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI

from src.routers import Router_Query, Router_Embedding, Router_Chat

setattr(asyncio.sslproto._SSLProtocolTransport, "_start_tls_compatible", True)
print(tiktoken.list_encoding_names())
load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
promptlayer.api_key = os.environ.get("PROMPTLAYER_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

app = FastAPI()
app.include_router(Router_Query.router)
app.include_router(Router_Chat.router)
app.include_router(Router_Embedding.router)
