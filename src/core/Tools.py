# 文件名转时间戳
import os
import re
from typing import Union, List

import openai
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, \
    AIMessagePromptTemplate
from llama_index import Prompt
from llama_index.callbacks import CallbackManager, TokenCountingHandler, LlamaDebugHandler
from llama_index.embeddings import OpenAIEmbedding
from llama_index.node_parser import SimpleNodeParser

from src.entity.QueryForm import RoleMessage


def get_unix_timestamp_from_filename(filepath):
    # 1. 使用os.path.basename获取文件名
    filename = os.path.basename(filepath)

    if 'public' not in filepath:
        return {"date": "2023-07-14"}

    # 2. 检查文件名是否包含时间戳
    parts = filename.split('_')
    if len(parts) < 3:
        return {"date": "2023-07-14"}

    # 3. 分割文件名，获取时间戳部分
    timestamp = parts[0]

    # 4. 将时间戳格式化为所需的格式
    formatted_timestamp = f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}'
    return {"date": formatted_timestamp}


def load_embed_model(embed_model_name, embed_mode, embed_batch_size):
    return OpenAIEmbedding(
        model=embed_model_name,
        mode=embed_mode,
        embed_batch_size=embed_batch_size,
    )


def load_llm(temperature, model_name, openai_api_key, openai_proxy, openai_api_base, max_tokens, callbacks=None):
    if callbacks is None:
        callbacks = []
    openai.api_key = os.environ["OPENAI_API_KEY"]

    return ChatOpenAI(
        temperature=float(temperature),
        model_name=model_name,
        streaming=True,
        openai_api_key=openai_api_key,
        openai_proxy=openai_proxy,
        openai_api_base=openai_api_base,
        max_tokens=int(max_tokens),
        callbacks=callbacks,
    )


def load_NodeParsers(chunk_size, chunk_overlap):
    return SimpleNodeParser.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def load_CallbackManager(embed_model_name):
    # callback_manager 回调管理器
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(embed_model_name).encode
    )
    return CallbackManager([token_counter, llama_debug])


def get_qa_template():
    with open("./src/prompt/QA_TEMPLATE_SYS.txt", "r") as f:
        TXT_SYSTEM_MESSAGE = f.read()
    with open("./src/prompt/QA_TEMPLATE_USER.txt", "r") as f:
        TXT_USER_MESSAGE = f.read()
    chat_text_qa_msgs = [
        SystemMessagePromptTemplate.from_template(
            TXT_SYSTEM_MESSAGE
        ),
        HumanMessagePromptTemplate.from_template(
            TXT_USER_MESSAGE
        ),
    ]
    chat_text_qa_msgs_lc = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    return Prompt.from_langchain_prompt(chat_text_qa_msgs_lc)


def extract_variable(s):
    pattern = re.compile(r'{{embedding:([^}]*)}}')
    return pattern.findall(s)


def parse_roleMessages_to_prompts(roleMessages: Union[List[RoleMessage], None], with_chat_history: bool = False,
                                  input_variables: List[str] = None, questionTemplate: str | None = None):
    # sourcery skip: raise-specific-error
    if roleMessages is None:
        return None

    chat_text_qa_msgs = []

    for roleMessage in roleMessages:
        _promptTemplate = None
        if roleMessage.role == "user":
            _promptTemplate = HumanMessagePromptTemplate.from_template(
                roleMessage.message
            )
        elif roleMessage.role == "system":
            _promptTemplate = SystemMessagePromptTemplate.from_template(
                roleMessage.message
            )
        elif roleMessage.role == "ai":
            _promptTemplate = AIMessagePromptTemplate.from_template(
                roleMessage.message
            )
        else:
            raise Exception("roleMessage.role must be one of 'user', 'system', 'ai'")

        chat_text_qa_msgs.append(_promptTemplate)

    if with_chat_history:
        chat_text_qa_msgs.append(HumanMessagePromptTemplate.from_template("""
以下通过三引号(```)划分的内容为聊天历史内容:
```
{chat_history}
```
请你根据聊天历史内容，回复用户以下问题：
Question：{input}
"""))
    else:
        chat_text_qa_msgs.append(
            HumanMessagePromptTemplate.from_template(
                questionTemplate,
                input_variables=input_variables,
                template_format="jinja2"
            )
        )
    return ChatPromptTemplate.from_messages(chat_text_qa_msgs)
