import os
from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
# ReAcT : langchain 으로 에이전트룰 구현하는 가장 인기있는 방법
# 에이전트는 LLM으로 파이썬 함수를 이용해 상호작용 한다
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from pydantic import SecretStr

def lookup(name: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    secret_api_key = SecretStr(api_key)
    llm = GoogleGenerativeAI(
        temperature=0,
        model="gemini-2.0-flash",
        api_key=secret_api_key,
        timeout=10
    )
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                        Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["name_of_person"]
    )
