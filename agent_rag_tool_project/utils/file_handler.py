import hashlib
import os
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from agent_rag_tool_project.utils.logger_handler import logger

def get_file_md5_hex(filepath: str):
    """
    获取文件的md5的十六进制字符串
    :return:
    """
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()

    # 避免文件过大爆内存，需要分片，固定写法
    chunk_size = 4096 # 4kb
    try:
        # rb是二进制读取方式
        with open(filepath, "rb") as f:
            """
            py语法等同于以下，先赋值再作为while的判断条件
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_width_allowed_type(path: str, allowed_types: tuple[str]):
    """
    返回文件夹内的文件列表（允许的文件后缀）
    :return:
    """
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_width_allowed_type]{path}不是文件夹")
        return tuple(files)

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(allowed_types):
                files.append(os.path.join(dirpath, filename))

    return tuple(files)


def pdf_loader(filepath: str, passwd = None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()

def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()

def _process_frontmatter(filepath: str, content: str):
    """
    处理markdown中的frontmatter部分
    :return:
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            body = parts[2]

            metadata = {"source": filepath}
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

            # 如果内容没有h1，将title作为h1添加到内容开头
            title = metadata.get("title", "")
            if title and not body.strip().startswith("# "):
                body = f"# {title}\n\n{body}"

            return {"body": body, "metadata": metadata}

    return {"body": content, "metadata": {"source": filepath}}

# 匹配代码块
CODE_BLOCK_PATTEN = re.compile(
    r"```[\s\S]*?```",
    re.MULTILINE
)

def replace_code_blocks(text: str):
    """
    提取md中的代码块，存入metadata，暂时替换成placeholder占位符，避免被分割
    :param text:
    :return:
    """
    code_map = {}

    def replacer(match):
        code = match.group(0)
        placeholder = f"[CODE_BLOCK_{len(code_map)}]"
        code_map[placeholder] = code
        return placeholder

    replaced_text = CODE_BLOCK_PATTEN.sub(replacer, text)
    return replaced_text, code_map

def restore_code_blocks(text: str, code_map: dict):
    """
    恢复代码块部分
    :param text:
    :param code_map:
    :return:
    """
    for placeholder, code in code_map.items():
        text = text.replace(placeholder, code)
    return text


def md_loader(filepath: str) -> list[Document]:
    with open(filepath, "r", encoding="utf-8") as f:
        md_text = f.read()

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3")
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on = headers_to_split_on,
        return_each_line = False,
    )
    process_data = _process_frontmatter(filepath, md_text)
    docs = md_splitter.split_text(process_data.get("body"))
    final_docs = []

    for doc in docs:

        # 将metadata中的标题部分重新拼回page_content
        headers = []
        if "h1" in doc.metadata:
            headers.append(f"# {doc.metadata['h1']}")
        if "h2" in doc.metadata:
            headers.append(f"## {doc.metadata['h2']}")
        if "h3" in doc.metadata:
            headers.append(f"### {doc.metadata['h3']}")
        header_text = "\n".join(headers)
        doc.page_content = f"{header_text}\n\n{doc.page_content}"

        # 替换代码块部分
        replaced_text, code_map = replace_code_blocks(doc.page_content)
        doc.page_content = replaced_text
        doc.metadata["_code_map"] = code_map

        # 处理好的数据重新组装返回
        final_docs.append(doc)
    return final_docs