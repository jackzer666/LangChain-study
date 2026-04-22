from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="/Volumes/Project_Disk/document/前端/vue源码题1.pdf",
    mode="page", # 默认是page模式，默认一个页面是一个document;single模式是一个文件一个document
    # password="", # 如果文件加密了，需要配置密码
)

for doc in loader.lazy_load():
    print(doc)