import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv




load_dotenv()


model= ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

while True:

    query = input("ASk Away: ").lower()
    if  query=="n":
        print("See Ya Later")
        break

    messages = [
        SystemMessage(content="You are a helpful Trade ChatBot.Provide very short and direct answers"),
        HumanMessage(content = query)
    ]

    result = model.invoke(messages)

    print(query)
    print(f"\nGenerated Response: {result.content}")

    

  