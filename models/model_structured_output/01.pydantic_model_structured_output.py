from pydantic import BaseModel, Field

from init_llm import zhipu_llm, deepseek_llm

class MyModel(BaseModel):
    name: str = Field(description='模型名称')
    source: str = Field(description='提供厂商')
    goodAt: list[str] = Field(description='擅长领域')

model = deepseek_llm.with_structured_output(MyModel, include_raw=True) 
res = model.invoke('介绍你自己')
print(res)