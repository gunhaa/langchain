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

from third_parties.linkedin import scrape_linkedin_profile

# __name__은 파이썬에서 제공하는 특수 변수로, 현재 실행 중인 파일의 이름을 담고있다
# 파이썬 스크립트가 직접 실행될 때만 True가 된다
if __name__ == "__main__":

    load_dotenv()
    print("hello langchain")
    # os.environ은 현재 운영 체제의 환경 변수를 담고 있는 딕셔너리
    # print(os.environ)
    # print(os.environ['GEMINI_API_KEY'])
    api_key = os.environ.get("GEMINI_API_KEY")
    secret_api_key = SecretStr(api_key)

    summary_template = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # 단순히 언어모델을 감싸는 wrapper class
    # temperature는 모델의 창의성 결정 : 0이면 창의적이지 않음
    # timeout= 요청하고 대기까지의 시간

    # Gemini 사용
    llm = GoogleGenerativeAI(
        temperature=0, model="gemini-2.0-flash", api_key=secret_api_key, timeout=10
    )

    chain = summary_prompt_template | llm


    # llama3 사용
    # llm = ChatOllama(
    #     model="llama3"
    # )

    # output parse 사용
    # chain = summary_prompt_template | llm | StrOutputParser()


    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco/")
    res = chain.invoke(input={"information": linkedin_data})

    print(res)
