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

information = """
Elon Reeve Musk (/ˈiːlɒn mʌsk/; born June 28, 1971) is a businessman and U.S. special government employee, best known for his key roles in Tesla, Inc., SpaceX, and the Department of Government Efficiency (DOGE), and his ownership of Twitter. Musk is the wealthiest individual in the world; as of February 2025, Forbes estimates his net worth to be US$397 billion.

Musk was born to an affluent South African family in Pretoria before immigrating to Canada, acquiring its citizenship from his mother. He moved to California in 1995 to attend Stanford University, and with his brother Kimbal co-founded the software company Zip2, that was later acquired by Compaq in 1999. That same year, Musk co-founded X.com, a direct bank, that later formed PayPal. In 2002, Musk acquired U.S. citizenship, and eBay acquired PayPal. Using the money he made from the sale, Musk founded SpaceX, a spaceflight services company, in 2002. In 2004, Musk was an early investor in electric vehicle manufacturer Tesla and became its chairman and later CEO. In 2018, the U.S. Securities and Exchange Commission (SEC) sued Musk for fraud, alleging he falsely announced that he had secured funding for a private takeover of Tesla; he stepped down as chairman and paid a fine. Musk was named Time magazine's Person of the Year in 2021. In 2022, he acquired Twitter, and rebranded the service as X the following year. In January 2025, he was appointed head of Trump's newly created DOGE.

His political activities and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, affirming antisemitic and transphobic comments, and promoting conspiracy theories. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. Musk has engaged in political activities in several countries, including as a vocal and financial supporter of U.S. president Donald Trump. He was the largest donor in the 2024 United States presidential election, and is a supporter of far-right activists, causes, and political parties.
"""


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
        given the information {information} about a person from I want you to create:
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
    # llm = GoogleGenerativeAI(
    #     temperature=0, model="gemini-2.0-flash", api_key=secret_api_key, timeout=30
    # )

    # llama3 사용
    llm = ChatOllama(
        model="llama3"
    )

    # Gemini
    # chain = summary_prompt_template | llm

    # llama3
    # output parse 사용
    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={"information": information})

    print(res)
