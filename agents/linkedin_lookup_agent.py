import os
from dotenv import load_dotenv

from tools.tool import get_profile_url_tavily

load_dotenv()
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from pydantic import SecretStr


def linkedin_lookup_agent(name: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    secret_api_key = SecretStr(api_key)
    llm = GoogleGenerativeAI(
        temperature=0, model="gemini-2.0-flash", api_key=secret_api_key, timeout=10
    )
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                        Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Linkedin page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url
