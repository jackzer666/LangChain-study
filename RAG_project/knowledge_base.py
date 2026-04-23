"""
知识库
"""
import hashlib
import os
from datetime import datetime

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG_project.config_data import md5_path, collection_name, persist_directory, chunk_size, chunk_overlap, separators, \
    max_split_char_number
from langchain_community.embeddings import ZhipuAIEmbeddings
from env_utils import ZHIPUAI_API_KEY


def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理过了
        return False md5未处理过
        return True md5已经处理过了
    """
    if not os.path.exists(md5_path):
        # w模式下打开关闭就会默认创建文件
        open(md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip() # 处理处理串前后空格和回车 去掉
            if line == md5_str:
                return True
        return False

def save_md5(md5_str: str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + "\n")

def get_string_md5(input_str: str, encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""
    # 将字符串转换成bytes字节数组
    str_bytes = input_str.encode(encoding)

    # 创建md5对象
    md5_obj = hashlib.md5()         # 得到md5对象
    md5_obj.update(str_bytes)       # 更新内容 传入即将要转换的字节数组
    md5_hex = md5_obj.hexdigest()   # 得到md5的十六进制字符串

    return md5_hex

embedding_model = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key=ZHIPUAI_API_KEY
)

class KnowledgeBaseService(object):
    def __init__(self):
        # 如果文件夹不存在就创建，如果存在就跳过
        os.makedirs(persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=collection_name, # 数据库表名
            embedding_function=embedding_model,
            persist_directory=persist_directory, # 数据库本地存储的文件夹
        )

        # 文本分割对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,          # 分割后的文本段最大长度
            chunk_overlap=chunk_overlap,    # 连读文本段之间允许重复的长度
            separators=separators,          # 自然段落划分的符号
            length_function=len,            # 使用python自带的len函数做长度统计的依据
        )

    def upload_by_str(self, data: str, filename):
        """将传入的字符串进行向量化，存入向量数据库中"""
        # 先得到传入字符串的md5值
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) > max_split_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 元数据定义
        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "测试人员"
        }

        # 内容加载到向量数数据库中
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas = [metadata for _ in knowledge_chunks], # list[dict]类型才这么写的
        )

        save_md5(md5_hex)

        return '[成功]内容已经成功载入向量库'


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str('周杰伦', 'testfile')
    print(r)
