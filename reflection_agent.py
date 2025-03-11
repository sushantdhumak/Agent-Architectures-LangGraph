# ===============================================
# Reflection Agent using Chains
# ===============================================

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Tweet Generation Prompt

tweet_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Twitter tech influencer assistant, specialized in crafting engaging and impactful Twitter posts."
            "Your task is to generate the best possible tweet based on the user's request, using creativity, clarity, and relevance."
            "If the user provides feedback, revise your previous tweet to better align with their critique and enhance the content.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Reflection Prompt

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral Twitter influencer tasked with evaluating tweets. Provide detailed feedback and recommendations for improving the user's tweet."
            "Your critique should include insights on length, style, virality potential, tone, and any other factors that can enhance the tweet's impact.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
    
)

# LLM

llm = ChatOpenAI(model="gpt-4o-mini")

# Chains

tweet_generation_chain = tweet_generation_prompt | llm
reflection_chain = reflection_prompt | llm


# -----------------------------------------------
# Define a Graph
# -----------------------------------------------

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from IPython.display import display, Image

# Load environment variables

from dotenv import load_dotenv
load_dotenv()

# Graph
 
builder = StateGraph(MessagesState)

# Node functions

def generate_tweet_node(state: MessagesState):
    
    print("Generating tweet...")

    messages = state["messages"]
    response = tweet_generation_chain.invoke(messages)

    return {"messages": [response]}

def reflection_node(state: MessagesState):
    
    print("Reflecting on tweet...")

    cls_map = {"ai": HumanMessage, "human": AIMessage}

    # First message is the original user request so we will keep it same for all nodes

    messages = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]

    # messages = state["messages"]
    response = reflection_chain.invoke(messages)

    return {"messages": [HumanMessage(content=response.content)]}

def should_continue_node(state: MessagesState):

    print("Should continue?")

    messages = state["messages"]
    
    if len(messages) > 6:
        return END
    
    return "reflection"

# Nodes

builder.add_node("generate_tweet", generate_tweet_node)
builder.add_node("reflection", reflection_node)

# Edges

builder.add_edge(START, "generate_tweet")
builder.add_conditional_edges("generate_tweet", should_continue_node)
builder.add_edge("reflection", "generate_tweet")
# builder.add_edge("generate_tweet", END)

# Compile the graph

graph = builder.compile()

# Visualize the graph

display(Image(graph.get_graph().draw_mermaid_png()))

# Invoke the graph

initial_tweet = HumanMessage(content="Write a tweet about AI Agents taking over the software industry.")
response = graph.invoke({"messages": [initial_tweet]})

for msg in response["messages"]:
    msg.pretty_print()


# -----------------------------------------------