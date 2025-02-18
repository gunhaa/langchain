from typing import Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# 운영 체제와 상호작용을 하기 위해 파이썬에서 제공하는 표준 라이브러리
from langchain_core.prompts import PromptTemplate
import os

# Suppress logging warnings
# os.environ["GRPC_VERBOSITY"] = "ERROR"
# os.environ["GLOG_minloglevel"] = "2"
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from agents.linkedin_lookup_agent import linkedin_lookup_agent
from output_parsers import summary_parser, Summary
from third_parties.linkedin import scrape_linkedin_profile


def ice_breaker_with(name: str) -> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)
    summary_template = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
        
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    secret_api_key = SecretStr(api_key)
    # Gemini 사용
    llm = GoogleGenerativeAI(
        temperature=0, model="gemini-2.0-flash", api_key=secret_api_key, timeout=10
    )
    chain = summary_prompt_template | llm | summary_parser

    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/eden-marco/"
    )
    res: Summary = chain.invoke(input={"information": linkedin_data})

    print(res)

    return res, linkedin_data.get("profile_pic_url")
