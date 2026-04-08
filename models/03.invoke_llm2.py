"""
模型调用方式
1. invoke

2. Stream Invoke

3. Batch Invoke
"""
from langchain_core.messages import SystemMessage, HumanMessage
from openai import max_retries

from init_llm import deepseek_llm, zhipu_llm

# 1. 单条消息调用模型
# res = zhipu_llm.invoke('介绍你自己')
# print(type(res))
# print(res)

# 字典格式的消息列表
# res2 = zhipu_llm.invoke([
#     { "role": "user", "content": "你好" },
#     { "role": "assistant", "content": "你好，有什么可以帮助你？" },
#     { "role": "user", "content": "介绍你自己" },
# ])
# print(type(res2))
# print(res2)

# 3. 消息对象格式的消息列表
# res3 = zhipu_llm.invoke([
#     SystemMessage(content="你是一个翻译助手，将汉语翻译成英语"),
#     HumanMessage(content="翻译：我喜欢编程"),
# ])
# print(type(res3))
# print(res3)


# stream invoke
# res = zhipu_llm.stream('请介绍你自己')
# for chunk in res:
#     print(type(chunk))
#     print(chunk.content)
#

# batch invoke，有的大模型有短时间内调用大模型的次数限制，那么批量调用就能把多次变成一次
# 这样写需要等待所有问题都结束才返回
# res = zhipu_llm.batch([
#     "介绍你自己",
#     "什么是大模型"
# ])
#
# for msg in res:
#     print(msg.content)

# 谁先结束就先返回谁，返回结果以及对应哪个问题的结果。因此有可能回答与问题的顺序是错位的
res = zhipu_llm.batch_as_completed([
    "介绍你自己",
    "1+1等于几，直接告诉我答案",
],
config={
    "max_concurrency": 1 # 可以并发执行几个，一般要看大模型本身支持批量调用几个
})
for msg in res:
    print(msg)