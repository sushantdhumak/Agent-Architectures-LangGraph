# ===============================================
# Reflexion Agent using Chains
# ===============================================


# -----------------------------------------------
# LLM Model
# -----------------------------------------------

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


# -----------------------------------------------
# Actor (with reflection)
# -----------------------------------------------

# Construct tools

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)


# -----------------------------------------------
# Responder Answer Schema

from pydantic import BaseModel, Field
from typing import List

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous") 


class AnswerQuestion(BaseModel):
    """
    Answer the question. Provide an answer, reflection, 
    and then follow up with search queries to improve the answer.
    """
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")
    

# -----------------------------------------------
# Response with retries

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

class ResponseWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: list):
        response = []

        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": state["messages"]}, {"tags": [f"attempt:{attempt}"]}
            )
            
            try:
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                state["messages"] = state["messages"] + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]

        return {"messages": response}


# -----------------------------------------------
# Responder Agent Prompt

import datetime
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are expert AI researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat(),)

responder_prompt_template = actor_prompt_template.partial(first_instruction="Provide a detailed ~250 word answer", 
                                                          function_name=AnswerQuestion.__name__,)

responder_chain = responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice='AnswerQuestion')

responder_validator = PydanticToolsParser(tools=[AnswerQuestion])

responder = ResponseWithRetries(runnable=responder_chain, validator=responder_validator)

# # Invoke the chain

# response = responder_chain.invoke({
#     "messages": [HumanMessage(content="Write me a blog on startup ideas using AI agents.")]
#     })

# print(response)


# -----------------------------------------------
# Revisor Answer Schema

class ReviseAnswer(AnswerQuestion):
    """
    Revise your original answer to your question. 
    Provide an answer, reflection, cite your reflection with references,
    and finally add search queries to improve the answer.
    """
    references: list[str] = Field(
        description="Citations motivating your updated answer."
    )


# -----------------------------------------------
# Revisor Agent Prompt

revise_instructions = """
    Revise your previous answer using the new information.

    - You should use the previous critique to add important information to your answer.    
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com

    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
    """

revisor_chain = actor_prompt_template.partial(first_instruction=revise_instructions, function_name=ReviseAnswer.__name__,
                                              ) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

revision_validator  = PydanticToolsParser(tools=[ReviseAnswer])

revisor = ResponseWithRetries(runnable=revisor_chain, validator=revision_validator)


# -----------------------------------------------
# Create a Tool Node
# -----------------------------------------------

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode


def run_queries(search_queries: list[str], **kwargs):
    """
    Run the generated queries.
    """
    return tavily_tool.batch([{"query": query} for query in search_queries])


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)


# -----------------------------------------------
# Define the Graph
# -----------------------------------------------

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


class State(TypedDict):
    messages: Annotated[list, add_messages]

MAX_ITERATIONS = 2

builder = StateGraph(State)

# Add nodes

builder.add_node("draft", responder.respond)
builder.add_node("execute_tools", tool_node)
builder.add_node("revisor", revisor.respond)

# Add edges

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revisor")

# Define looping logic

def _get_num_iterations(state: list):
    i = 0

    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1

    return i

# Add conditional edges

def event_loop(state: list):
    num_iterations = _get_num_iterations(state["messages"])

    if num_iterations > MAX_ITERATIONS:
        return END
    
    return "execute_tools"

builder.add_conditional_edges("revisor", event_loop, ["execute_tools", END])

# Compile graph

graph = builder.compile()

# Visualize graph

display(Image(graph.get_graph().draw_mermaid_png()))

# Invoke graph

events = graph.stream(
    {"messages": [("user", "How should we handle the climate crisis?")]},
    stream_mode="values",
)

for i, step in enumerate(events):
    print(f"Step {i}")
    step["messages"][-1].pretty_print()


# -----------------------------------------------
