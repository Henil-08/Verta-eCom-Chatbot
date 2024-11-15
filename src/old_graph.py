import os
from src.components.state import MultiAgentState
import src.components.nodes as node
import src.components.agents as agent
from dotenv import load_dotenv
from src.constants import MEMBERS, CONDITIONAL_MAP
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver


def create_graph(isMemory=True):
    memory = MemorySaver()
    builder = StateGraph(MultiAgentState)

    builder.add_node("Metadata", agent.metadata_node)
    builder.add_node("Review-Vectorstore", agent.retrieve)
    builder.add_node("supervisor", agent.supervisor_agent)
    builder.add_node("generate", node.final_llm_node)
    builder.add_node("final", node.followup_node)

    for member in MEMBERS:
        builder.add_edge(member, "supervisor")

    builder.add_conditional_edges("supervisor", node.route_question, CONDITIONAL_MAP)

    builder.add_edge(START, "supervisor")
    builder.add_edge("generate", "final")
    builder.add_edge("final", END)

    graph = builder.compile(checkpointer=memory) if isMemory else builder.compile()

    return graph


if __name__ == '__main__':
    _ = load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    app = create_graph()