from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from openapikey import load_openai_api_key




agent = create_agent(
    model=init_chat_model(
        model="openai:gpt-4", 
        temperature=0.1,
        max_tokens=1000,
        timeout=30,
        openai_api_key=load_openai_api_key(),
    )
    system_prompt=(
        "You are a helpful assistant. "
    ),
)


result = agent.invoke(
    {"input": "How many inhabitants does Paris have? "}
)

final_message = result["messages"][-1].content
print("\n=== Agent Response ===")
print(final_message)
