# ✅ Required Imports
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama import ChatOllama  # Ollama model integration


# ✅ Define AI Model
MODEL_NAME = "gemma3:12b"
# you need a more powerful model  gemma3:12b   at least 
def langchain_agent(question: str):
    """Set up and run a LangChain agent with Wikipedia and Math tools."""
    
    # ✅ Initialize the Ollama model
    llm = ChatOllama(model=MODEL_NAME, temperature=0.5)

    # ✅ Load tools (Wikipedia first, then Math)
    tools = load_tools(['wikipedia']) + load_tools(['llm-math'], llm=llm)

    # ✅ Initialize the agent with tool usage
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Dynamic reasoning
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True
    )

    try:
        # ✅ Process the question through the agent
        result = agent.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Run the function if script is executed directly
if __name__ == "__main__":
    user_question = input("Enter your question: ")
    response = langchain_agent(user_question)
    print("\n🤖 AI Response:", response)
