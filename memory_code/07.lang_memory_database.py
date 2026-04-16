from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.store.mysql import PyMySQLStore

from init_llm import deepseek_llm

DB_URI="xxx"

with (
    PyMySQLSaver.from_conn_string(DB_URI) as checkpointer,
    PyMySQLStore.from_conn_string(DB_URI) as store,
):
    checkpointer.setup()
    store.setup()

    # 长期记忆，和用户线程没关系，和会话也没关系
    store=InMemoryStore()

    store.put(
        # 命名空间，理解为层级目录
        ("users",),
        # 存的数据，理解为数据表
        key="user_123", # 键
        value={"name": "张三", "city": "北京"} # 值
    )

    store.put(
        ("users",),
        key="user_456",
        value={"name": "李四", "city": "杭州"}
    )

    @tool
    def get_user_info(runtime: ToolRuntime):
        """该工具返回用户信息，包括用户名称和城市信息"""

        store = runtime.store
        user_id = "user_123"
        user_info = store.get(("users",), user_id)

        if user_info:
            print(user_info)
            value = user_info.value
            return f"用户姓名：{value['name']},用户城市{value['city']}"

        return "用户信息不存在"



    agent = create_agent(
        model=deepseek_llm,
        tools=[get_user_info],
        system_prompt="你是一个用户信息查询助手，当用户需要获取用户信息时，调用get_user_info工具查询用户信息",
        checkpointer=InMemorySaver(),
        # 长期存储
        store=store
    )

    # 固定写法
    config1 = {"configurable": {"thread_id": "session01"}}
    config2 = {"configurable": {"thread_id": "session02"}}

    res1 = agent.invoke({"messages": [{"role": "user", "content": "从长期记忆里面获取用户信息"}]}, config=config1)
    print(res1["messages"][-1].content)

    print("*" * 50)
    res2 = agent.invoke({"messages": [{"role": "user", "content": "从长期记忆里面获取用户信息 "}]}, config=config2)
    print(res2["messages"][-1].content)
