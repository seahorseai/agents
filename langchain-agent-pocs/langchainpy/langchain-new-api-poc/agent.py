from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from openapikey import load_openai_api_key

@tool
def multiplyTool(a: int, b: int) -> str:
    """Multiply two numbers."""
    return f"{a} × {b} = {a * b}"


agent = create_agent(
    model=init_chat_model(
        model="openai:gpt-4", 
        temperature=0.1,
        max_tokens=1000,
        timeout=30,
        openai_api_key=load_openai_api_key(),
    ),
    tools=[multiplyTool],
    system_prompt=(
        "You are a helpful assistant. "
        "You MUST use tools to answer. "
        "Do not answer directly."
    ),
)


result = agent.invoke(
    {"input": "Use multiplyTool with a=2 and b=2"}
)

final_message = result["messages"][-1].content
print("\n=== Agent Response ===")
print(final_message)
