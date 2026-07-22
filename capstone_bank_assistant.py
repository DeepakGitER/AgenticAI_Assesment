
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

MODEL = os.environ.get(
    "MODEL",
    "openai/gpt-4o-mini"
)

model = init_chat_model(
    MODEL,
    model_provider="openrouter",
    temperature=0
)

# -----------------------
# PART A
# -----------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Meridian Bank's assistant. Answer in one or two short sentences."
        ),
        (
            "human",
            "{question}"
        )
    ]
)

chain = prompt | model | StrOutputParser()

print("=== Meridian Bank Assistant ===\n")

q1 = "What documents do I need for a home loan?"

a1 = chain.invoke(
    {
        "question": q1
    }
)

print("Q:", q1)
print("A:", a1)
print()

q2 = "What is a fixed deposit?"

a2 = chain.invoke(
    {
        "question": q2
    }
)

print("Q:", q2)
print("A:", a2)
print()

# -----------------------
# PART B
# -----------------------

@tool
def calculate_emi(
    principal: float,
    annual_rate: float,
    years: int
) -> dict:
    """Calculate monthly EMI."""

    r = annual_rate / 100 / 12
    n = years * 12

    emi = (
        principal / n
        if r == 0
        else principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    )

    return {
        "emi": round(
            emi,
            2
        )
    }

agent = create_agent(
    model,
    tools=[calculate_emi],
    system_prompt=
    "You are Meridian Bank's assistant. Use tools for EMI maths."
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content":
                "What is the EMI on a 50 lakh loan at 8.4% for 25 years?"
            }
        ]
    }
)

print("EMI Answer:")
print(
    result["messages"][-1].content
)

# -----------------------
# PART C
# -----------------------

class LoanRequest(BaseModel):
    intent: str
    product: str
    amount: str

structured_model = (
    model.with_structured_output(
        LoanRequest
    )
)

loan_info = structured_model.invoke(
    "I'd like a home loan of about 30 lakh."
)

print("\nStructured Output:")
print(loan_info)