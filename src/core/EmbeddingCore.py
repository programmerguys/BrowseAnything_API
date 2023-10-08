import asyncio
import os

import openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus

from src.core.Tools import load_embed_model

load_dotenv()
_openai_api_key = os.environ.get('OPENAI_API_KEY')

os.environ["OPENAI_API_KEY"] = _openai_api_key
openai.api_key = _openai_api_key

_embed_model_name = os.environ.get('EMBED_MODEL_NAME')
_embed_mode = os.environ.get('EMBED_MODE')
_embed_batch_size = int(os.environ.get('EMBED_BATCH_SIZE'))
embed_model = load_embed_model(_embed_model_name, _embed_mode, _embed_batch_size)
milvus_collection_name = os.environ.get('MILVUS_COLLECTION_NAME')
milvus_host = os.environ.get('MILVUS_HOST')
milvus_port = os.environ.get('MILVUS_PORT')
milvus_user = os.environ.get('MILVUS_USER')
milvus_password = os.environ.get('MILVUS_PASSWORD')
milvus_db_name = os.environ.get('MILVUS_DB_NAME')


class EmbeddingCore:
    def __init__(
            self,
            _openai_api_key: str,
            _embed_model: OpenAIEmbeddings,
            _milvus_collection_name: str,
            _milvus_host: str,
            _milvus_port: str,
            _milvus_user: str,
            _milvus_password: str,
            _milvus_db_name: str
    ):
        # 读取embedding模型配置

        async def similarity_search(query, db, expr_text):
            expr = f'text=="{expr_text}"'
            return await db.similarity_search_with_score(query, k=4)

        async def main():
            db = Milvus(
                embedding_function=_embed_model,
                collection_name="beauty",
                connection_args={
                    'host': "192.168.1.11",
                    'port': "19530",
                    'user': "username",
                    'password': "password",
                    'db_name': "beauty"
                }
            )
            query = "蜡笔小新"
            expr_text = ""
            expr = f'text=="{expr_text}"'
            tasks = [similarity_search(query, db, expr_text) for _ in range(100)]

            results = await asyncio.gather(*tasks)

        asyncio.run(main())


embedding_core = EmbeddingCore(
    _openai_api_key,
    embed_model,
    _milvus_collection_name=milvus_collection_name,
    _milvus_host=milvus_host,
    _milvus_port=milvus_port,
    _milvus_user=milvus_user,
    _milvus_password=milvus_password,
    _milvus_db_name=milvus_db_name
)
