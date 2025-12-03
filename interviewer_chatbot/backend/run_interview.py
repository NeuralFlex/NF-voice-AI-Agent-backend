from langgraph.graph import StateGraph, END
from typing import TypedDict


class State(TypedDict):
    user_input: str
    response: str


def echo_node(state: State):
    return {"response": state["user_input"]}


def create_workflow():
    graph = StateGraph(State)

    graph.add_node("echo", echo_node)
    graph.set_entry_point("echo")
    graph.add_edge("echo", END)

    return graph.compile()
