import os
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from schemas import AgentResponse

# Load environment variables
load_dotenv(dotenv_path="langchain-course/.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")
if not TAVILY_API_KEY:
    raise ValueError("Missing TAVILY_API_KEY in .env")

# Initialize LLM and tools
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
tools = [TavilySearch(tavily_api_key=TAVILY_API_KEY)]

# Load the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Wrap it in an executor (handles calling the tools)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    result = executor.invoke({
        "input": (
            "Search for 3 job postings for an AI engineer using LangChain "
            "in the Bay Area on LinkedIn and list their details."
        )
    })
    print(result)

if __name__ == "__main__":
    main()