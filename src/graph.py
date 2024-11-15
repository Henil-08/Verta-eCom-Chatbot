from functools import partial
import src.components.nodes as node
import src.components.agents as agent
from src.components.state import MultiAgentState
from src.constants import MEMBERS, CONDITIONAL_MAP
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from src.entity.config_entity import (PrepareBaseModelConfig)


class Graph:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def create_graph(self, isMemory=True) -> CompiledStateGraph:
        memory = MemorySaver()
        builder = StateGraph(MultiAgentState)

        builder.add_node("Metadata", partial(agent.metadata_node, 
                                             prompt=self.config.prompt_metadata, 
                                             model=self.config.metadata_model))
        builder.add_node("Review-Vectorstore", agent.retrieve)
        builder.add_node("supervisor", partial(agent.supervisor_agent, 
                                               prompt=self.config.prompt_supervisor, 
                                               model=self.config.supervisor_model))
        builder.add_node("generate", partial(node.final_llm_node,
                                             prompt=self.config.prompt_base_model, 
                                             model=self.config.base_model))
        builder.add_node("final", partial(node.followup_node,
                                         prompt=self.config.prompt_followup, 
                                         model=self.config.followup_model))

        for member in MEMBERS:
            builder.add_edge(member, "supervisor")

        builder.add_conditional_edges("supervisor", node.route_question, CONDITIONAL_MAP)

        builder.add_edge(START, "supervisor")
        builder.add_edge("generate", "final")
        builder.add_edge("final", END)

        graph = builder.compile(checkpointer=memory) if isMemory else builder.compile()
        
        return graph
