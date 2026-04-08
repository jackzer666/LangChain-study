from typing import TypedDict, Annotated

from init_llm import zhipu_llm, deepseek_llm

class MyField(TypedDict):
    field: Annotated[str, "领域名称"]

class MyModel(TypedDict):
    name: Annotated[str, "大模型名称"]
    source: Annotated[str, "大模型的提供厂商"]
    goodAt: Annotated[list[MyField], '擅长领域']

model = deepseek_llm.with_structured_output(MyModel)
res = model.invoke('介绍你自己')
print(res)