from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information ="""
    Elon Reeve Musk is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth to be US$500 billion
    """

    summary_template = """
    Given the following information about a person, extract a short summary:
    {information}
    1. Write a short summary of the person's life
    2. Give me two cool facts about the person
    """
    summary_promp_template = PromptTemplate(input_variables = "information", template = summary_template)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_promp_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
