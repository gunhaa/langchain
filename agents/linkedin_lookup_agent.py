import os
from dotenv import load_dotenv

from tools.tool import get_profile_url_tavily

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
        # Tool 객체 초기화
        # 더 많은 옵션이 있지만 3가지만으로도 충분히 유용하다
        Tool(
            # 에이전트가 이 도구를 사용할때 쓰는 이름
            # log에 보이게 된다
            name="Crawl Google 4 linkedin profile page",
            # 이 도구가 실행하기를 원하는 python 함수
            func=get_profile_url_tavily,
            # LLM이 이 도구를 사용할지 말지를 결정하는 기준이다
            # 간결하면서도 최대한 많은 정보를 설명에 담아 LLM이 헷갈리지 않고 언제 어떤 도구를
            # 사용할지 알게 해야 한다
            description="useful for when you need get the Linkedin page URL",
        )
    ]

    # ReAct 프롬프트 가져오기
    # LLM에 보내지는 프롬프트
    react_prompt = hub.pull("hwchase17/react")
    # 레시피
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    # 실제로 agent를 동작시킨다
    # verbose 설정을 통해 로깅과 에이전트 작동을 확인할 수 있다
    # 모든 것을 조율하고 실제로 Python함수를 호출한다
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url
