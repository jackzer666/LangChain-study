from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from agent_rag_tool_project.utils.config_handler import rag_conf, agent_conf
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI

from env_utils import ZHIPUAI_API_KEY, MINIMAX_API_KEY, MINIMAX_BASE_URL


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatZhipuAI(
            model = agent_conf["chat_model_air"],
            api_key=ZHIPUAI_API_KEY
        )

class RewriteModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatZhipuAI(
            model=agent_conf["chat_model_rewrite"],
            api_key=ZHIPUAI_API_KEY,
        )
        # return ChatOpenAI(
        #     model=agent_conf["chat_model_rewrite"],
        #     api_key=(MINIMAX_API_KEY or "").strip(),
        #     base_url=(MINIMAX_BASE_URL or "https://api.minimax.io/v1").strip(),
        #     temperature=0,
        # )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ZhipuAIEmbeddings(
            model=agent_conf["embedding_model_name"],  # 指定使用的模型，例如 "embedding-3"
            # dimensions=1024     # 可选参数：指定输出向量的维度
            api_key=ZHIPUAI_API_KEY
        )



chat_model = ChatModelFactory().generator()
rewrite_model = RewriteModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
