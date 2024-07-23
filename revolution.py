import asyncio
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents import tool
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os
import pprint
from langchain import hub

from tabnanny import verbose
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

MODEL = 'gpt-3.5-turbo'
TEMPERATURE = 1.0

# def math_chatbot():
    
#     word_problem_template = """You are a reasoning agent tasked with solving the user's logic-based questions.
#     Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved and give
#     the final answer. Provide the response in bullet points. Question  {question} Answer"""

#     math_assistant_prompt = PromptTemplate(input_variables=["question"], template=word_problem_template)

#     word_problem_chain = LLMChain(llm=llm, prompt=math_assistant_prompt)
#     word_problem_tool = Tool.from_function(name="Reasoning Tool", func=word_problem_chain.run, description="Useful for when you need to answer logic-based/reasoning questions.")

#     problem_chain = LLMMathChain.from_llm(llm=llm)
#     math_tool = Tool.from_function(name="Calculator", func=problem_chain.run, description="Useful for when you need to answer numeric questions. This tool is only for math questions and nothing else. Only input math expressions, without text")

#     wikipedia = WikipediaAPIWrapper()
#     wikipedia_tool = Tool(name="Wikipedia", func=wikipedia.run, description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")

#     agent = initialize_agent(tools=[wikipedia_tool, math_tool, word_problem_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False, handle_parsing_errors=True)

#     return agent

# async def process_user_query(agent, message: str):
#     response = await agent.ainvoke(message, verbose=True)
#     print(response["output"])

# agent = math_chatbot()

#############################

@tool
async def sum_up(numbers: str) -> int:
    '''Tool to sum up two integer numbers'''
    
    a, b = map(int, numbers.split(','))
    
    return a + b

model = AzureChatOpenAI(
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    azure_deployment=os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME']
)

tools = [sum_up]

message = HumanMessage(
    content='''
    а) Решите уравнение $\sin 2 x+\sqrt{3}(\cos x-\sin x)=1,5$.
    б) Укажите корни этого уравнения, принадлежащие отрезку $\left[-\frac{7 \pi}{2} ;-2 \pi\right]$.
    '''
)

prompt = hub.pull("hwchase17/react-chat")

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

input_message = 'what is 57 plus 82? Only use a tool if needed, otherwise respond with Final Answer'

for step in agent_executor.iter({'input': input_message}):
    if output := step.get("intermediate_step"):
        action, value = output[0]
        
        if action.tool == 'sum_up':
            print('Using sum tool, ', value)
        
        _continue = input("Should the agent continue (Y/n)?:\n") or "Y"
        if _continue.lower() != "y":
            break
        


# chunks = []
# 
# async def process_chunks():
#     async for chunk in agent_executor.astream(
#         {
#             "chat_history": [
#                 {"role": "system", "content": "You are an adding agent."}
#             ],
#             "input": input_message
#         }
#     ):
#         chunks.append(chunk)
#         print("------")
#         pprint.pprint(chunk, depth=1)
# 
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)
# 
# task = loop.create_task(process_chunks())
# if not loop.is_running():
#     loop.run_until_complete(task)



# agent_executor.invoke({
#         "input": input_message,
#         "chat_history": "",
#     }
# )
