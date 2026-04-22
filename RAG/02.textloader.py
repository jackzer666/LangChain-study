from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(
    './test.txt',
    encoding='utf-8',
)

docs = loader.load()
print(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=5, # 分段的最大字符数
    chunk_overlap=0, # 分段之间允许的重叠字符数
    separators=["\n\n", "\n", "。", ".", "!", "?", " ", "？", "！", "，", ","], # 分隔符
    length_function=len,
)

split_docs = splitter.split_documents(docs)
print(split_docs)