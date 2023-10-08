import os
from typing import Union, List

import graphsignal
from dotenv import load_dotenv
from llama_index import ServiceContext, VectorStoreIndex

from src.core.Tools import load_embed_model, load_llm, load_NodeParsers, \
    load_CallbackManager, parse_roleMessages_to_prompts
from src.entity.QueryForm import RoleMessage

load_dotenv()


class LLMs(object):
    def __init__(self):
        # 读取embedding模型配置
        self._embed_model_name = os.environ.get('EMBED_MODEL_NAME')
        self._embed_mode = os.environ.get('EMBED_MODE')
        self._embed_batch_size = int(os.environ.get('EMBED_BATCH_SIZE'))
        self.embed_model = load_embed_model(self._embed_model_name, self._embed_mode, self._embed_batch_size)

        # 配置llm模型
        self._temperature = float(os.environ.get('OPENAI_TEMPERATURE'))
        self._model_name = os.environ.get('OPENAI_MODEL_NAME')
        self._openai_api_key = os.environ.get('OPENAI_API_KEY')
        self._openai_proxy = os.environ.get('OPENAI_PROXY')
        self._openai_api_base = os.environ.get('OPENAI_API_BASE')
        self._max_tokens = int(os.environ.get('OPENAI_MAX_TOKENS'))
        self.llm = load_llm(
            self._temperature,
            self._model_name,
            self._openai_api_key,
            self._openai_proxy,
            self._openai_api_base,
            self._max_tokens
        )

        # NodeParsers 节点解析器
        self._chunk_size = int(os.environ.get('NODE_PARSER_CHUNK_SIZE'))
        self._chunk_overlap = int(os.environ.get('NODE_PARSER_CHUNK_OVERLAP'))
        self.node_parser = load_NodeParsers(
            self._chunk_size,
            self._chunk_overlap
        )

        # callback_manager 回调管理器
        self._embed_model_name = os.environ.get('EMBED_MODEL_NAME')
        self.callback_manager = load_CallbackManager(self._embed_model_name)

        # ServiceContext 服务上下文
        self._num_output = int(os.environ.get('SERVICE_CONTEXT_NUM_OUTPUT'))
        self._context_window = int(os.environ.get('SERVICE_CONTEXT_CONTEXT_WINDOW'))

        self.graphsignal_name = os.environ.get('GRAPHSIGNAL_NAME')

        graphsignal.configure(api_key='16ddccf7551cb0e1b4a20435c495e999', deployment=self.graphsignal_name)

    def get_query_engine(
            self,
            temperature: float = None,
            model_name: str = None,
            roleMessages: Union[List[RoleMessage], None] = None,
    ):
        _llm = load_llm(temperature, model_name, self._openai_api_key, self._openai_proxy, self._openai_api_base,
                        self._max_tokens)
        _service_context = ServiceContext.from_defaults(
            llm=_llm,
            embed_model=self.embed_model,
            node_parser=self.node_parser,
            callback_manager=self.callback_manager,
            num_output=self._num_output,
            context_window=self._context_window
        )
        text_qa_template = parse_roleMessages_to_prompts(roleMessages)
        _index = VectorStoreIndex([])
        return _index.as_query_engine(
            service_context=_service_context,
            streaming=True,
            verbose=True,
            text_qa_template=text_qa_template,
        )

    def get_chat_engine(
            self,
            temperature: float = None,
            model_name: str = None,
            roleMessages: Union[List[RoleMessage], None] = None,
    ):
        _llm = load_llm(temperature, model_name, self._openai_api_key, self._openai_proxy, self._openai_api_base,
                        self._max_tokens)
        _service_context = ServiceContext.from_defaults(
            llm=_llm,
            embed_model=self.embed_model,
            node_parser=self.node_parser,
            callback_manager=self.callback_manager,
            num_output=self._num_output,
            context_window=self._context_window
        )
        text_qa_template = parse_roleMessages_to_prompts(roleMessages)
        _index = VectorStoreIndex.from_documents([])
        return _index.as_chat_engine(
            service_context=_service_context,
            streaming=True,
            verbose=True,
            text_qa_template=text_qa_template,
        )


llms = LLMs()
