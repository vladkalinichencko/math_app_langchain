import asyncio
from tabnanny import verbose
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
import os

load_dotenv()

def math_chatbot():
    llm = AzureChatOpenAI(
        openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
        azure_deployment=os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
        # temperature=0,
        # max_tokens=None,
        # timeout=None,
        # max_retries=3,
        # model_name='gpt-3.5-turbo',
        # openai_api_key='AZURE_OPENAI_API_KEY',
    )

    word_problem_template = """You are a reasoning agent tasked with solving the user's logic-based questions.
    Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved and give
    the final answer. Provide the response in bullet points. Question  {question} Answer"""

    math_assistant_prompt = PromptTemplate(input_variables=["question"], template=word_problem_template)

    word_problem_chain = LLMChain(llm=llm, prompt=math_assistant_prompt)
    word_problem_tool = Tool.from_function(name="Reasoning Tool", func=word_problem_chain.run, description="Useful for when you need to answer logic-based/reasoning questions.")

    problem_chain = LLMMathChain.from_llm(llm=llm)
    math_tool = Tool.from_function(name="Calculator", func=problem_chain.run, description="Useful for when you need to answer numeric questions. This tool is only for math questions and nothing else. Only input math expressions, without text")


    agent = initialize_agent(tools=[math_tool, word_problem_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False, handle_parsing_errors=True)

    return agent

async def process_user_query(agent, message: str):
    response = await agent.ainvoke(message, verbose=True)
    print(response["output"])

agent = math_chatbot()

query = input("Enter your query: ")

loop = asyncio.get_event_loop()
if loop.is_running():
    loop.create_task(process_user_query(agent, query))
else:
    loop.run_until_complete(process_user_query(agent, query))
