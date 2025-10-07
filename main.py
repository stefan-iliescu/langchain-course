from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="/Users/tudoriliescu/Documents/langchain-course/.env")



def main():
    print("Hello from langchain-course!")
    print(os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
