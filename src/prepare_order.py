import os
import pathlib
import sys
from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import ChatPromptTemplate

from src.Assistant import Assistant, State
from src.vector_search import order_item

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from vector_search import get_category_items,get_categories
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode

'''
%pip install azure-cosmos
%pip install langchain-openai~=0.3.2
%pip install azure-core~=1.32.0
%pip install tenacity~=9.0.0
%pip install langgraph~=0.2.67
%pip install langchain-core~=0.3.31
'''

load_dotenv()

AZURE_OPEN_AI_KEY = os.getenv('AZURE_OPEN_AI_KEY')
AZURE_OPEN_AI_ENDPOINT = os.getenv('AZURE_OPEN_AI_ENDPOINT')

questions = ["what are the products available?","what are the products in Dals?","order Toor Dal"]

prompt = '''
if it is a generic question list the categories from the database and show to the customer asking for a category.Get the items of the selected category from the database.
if the message contains a category get the items from the category.Retrieve the category name from the text and pass it to the agents.
For the items selected place the order.
Inform to customer and sales agent regarding the order status.

'''

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",prompt),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

chat_model = AzureChatOpenAI(
    azure_deployment="gpt-4",
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=AZURE_OPEN_AI_KEY,
    azure_endpoint=AZURE_OPEN_AI_ENDPOINT,
    streaming=False,
)

tools = [get_categories, get_category_items,order_item]
assistant_runnable = primary_assistant_prompt | chat_model.bind_tools(tools)

builder = StateGraph(State)

tool_node = ToolNode(tools)
# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools",tool_node)
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)

for q in questions:
    # Use the Runnable
    final_state = part_1_graph.invoke(
        {"messages": ("user",q)},
        config={"configurable": {"thread_id": 42}}
    )
    print(final_state["messages"][-1].content)
