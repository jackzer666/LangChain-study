# from my_llm import deepseek_llm, zhipu_llm, zhipu_llm2
#
# # print(deepseek_llm.invoke("请介绍一下你自己"))
# # print(zhipu_llm.invoke("请介绍一下你自己"))
#
# print(zhipu_llm2.invoke("请介绍一下你自己"))
from init_llm import zhipu_llm

print(zhipu_llm.invoke('请介绍一下你自己')
)
