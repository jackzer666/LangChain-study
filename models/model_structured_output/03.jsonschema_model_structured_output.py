from typing import TypedDict, Annotated


from init_llm import zhipu_llm, deepseek_llm

json_scheme = {
    # 必须要有title而且不能是中文，底层会转换成工具去调用
    "title": "getModelInfo",
    "description": "模型信息",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "大模型名称"},
        "source": {"type": "string", "description": "大模型的提供厂商"},
        "goodAt": {
            "type": "object",
            "description": "大模型擅长的领域",
            "properties": {
                "field": {"type": "string", "description": "领域名称"}
            }
        }
    }
}


model = deepseek_llm.with_structured_output(json_scheme)
res = model.invoke('介绍你自己')
print(res)

# 验证是否是json schema
# import jsonschema
# try:
#     error = jsonschema.validate(instance=res, schema=json_scheme)
# except jsonschema.exceptions.ValidationError:
#     print(error)