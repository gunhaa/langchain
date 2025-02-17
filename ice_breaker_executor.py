from dotenv import load_dotenv

from ice_breaker import ice_breaker_with

if __name__ == "__main__":
    load_dotenv()
    print("ice breaker start")
    ice_breaker_with(name = "Eden Marco Udemy")