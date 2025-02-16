# pipenv
```shell
# pip 최신 버전 업데이트
python -m pip install --upgrade pip

pip install pipenv
pipenv shell
pipenv install langchain
pipenv install langchain-openai
pipenv install langchain-community
pipenv install langchainhub
# 포매터 설치 
# black . 으로 실행 가능
# 파이썬 스타일에 맞게 코드 자동 수정
pipenv install black
# 환경 변수 관리
pipenv install python-dotenv
pipenv install langchain-google-genai
pipenv install pydantic

# 라마 설치 | 실행
ollama run llama3
pipenv install langchain-ollama

# pipenv의 모든 의존성 제거
# .venv를 수동삭제해야 할 수도 있음
pipenv uninstall --all
```
