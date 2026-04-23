md5_path = './md5.txt'

# Chroma
collection_name = "rag"
persist_directory = "./chroma_db"
similarity_threshold = 2 # 相似度检索返回匹配的文档数量

# spliter
chunk_size = 20
chunk_overlap = 5
separators = ['\n\n', '\n', '.', ',', '?', '!', '。', '，', '？', '！']
max_split_char_number = 1000