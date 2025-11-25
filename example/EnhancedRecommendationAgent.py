"""
Enhanced Recommendation Agent composed from smaller, easier-to-maintain pieces.
"""

import json
import logging

from websocietysimulator import Simulator
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
from websocietysimulator.agent.modules.planning_modules import PlanningIO
from websocietysimulator.agent.modules.reasoning_modules import ReasoningStepBack
from websocietysimulator.llm import InfinigenceLLM

from enhanced_agent.base_agent import EnhancedRecommendationAgentBase
from enhanced_agent.workflow_mixins import EnhancedWorkflowMixin

logging.basicConfig(level=logging.INFO)


class EnhancedRecommendationAgent(EnhancedWorkflowMixin, EnhancedRecommendationAgentBase):
    """
    Final concrete agent that bundles the base functionality with the library of
    workflow combinations.
    """

    def __init__(self, llm):
        """
        Initialize the enhanced recommendation agent.
        
        Args:
            llm: LLM instance
        """
        from websocietysimulator.agent.modules.info_orchestrator_module import InfoOrchestrator
        
        planning = PlanningIO(llm)
        memory = MemoryGenerative(llm)
        reasoning = ReasoningStepBack(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=llm,
        )
        
        # Initialize InfoOrchestrator with optimization settings
        # SchemaFitterIO will be initialized with interaction_tool later
        info_orchestrator = InfoOrchestrator(
            memory=memory,
            llm=llm,
            schema_fitter=None,  # Will be set when interaction_tool is available
            interaction_tool=None,  # Will be set when interaction_tool is available
            use_fixed_item_params=True,  # Use user-aligned params for all items (faster)
            max_candidates_to_profile=None  # Profile all candidates (set to 10 for top-10 only)
        )
        
        super().__init__(
            llm=llm,
            planning_module=planning,
            memory_module=memory,
            reasoning_module=reasoning,
            info_orchestrator=info_orchestrator,
        )
        
        # Store for later initialization
        self._schema_fitter_llm = llm
    
    def insert_task(self, task):
        """
        Override insert_task to initialize InfoOrchestrator with interaction_tool.
        """
        super().insert_task(task)
        
        # Initialize InfoOrchestrator's schema_fitter and interaction_tool
        if self.info_orchestrator and self.interaction_tool:
            from websocietysimulator.agent.modules.schemafitter_module import SchemaFitterIO
            
            if self.info_orchestrator.schema_fitter is None:
                schema_fitter = SchemaFitterIO(self._schema_fitter_llm, self.interaction_tool)
                self.info_orchestrator.schema_fitter = schema_fitter
            
            # Always update interaction_tool (it may be None initially)
            self.info_orchestrator.interaction_tool = self.interaction_tool
            # Update retrievers' interaction_tool
            if self.info_orchestrator.user_retriever:
                self.info_orchestrator.user_retriever.interaction_tool = self.interaction_tool
            if self.info_orchestrator.item_retriever:
                self.info_orchestrator.item_retriever.interaction_tool = self.interaction_tool


if __name__ == "__main__":
    logging.info("Initializing simulator...")
    task_set = "yelp"
    data_dir = "./data_processed"

    simulator = Simulator(data_dir=data_dir, device="auto", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"./track2/{task_set}/tasks",
        groundtruth_dir=f"./track2/{task_set}/groundtruth",
    )
    simulator.set_agent(EnhancedRecommendationAgent)
    simulator.set_llm(InfinigenceLLM(api_key="your_api_key_here"))

    logging.info("Starting recommendation simulation...")
    simulator.run_simulation(number_of_tasks=10, enable_threading=True, max_workers=5)
    logging.info("Evaluating results...")
    evaluation_results = simulator.evaluate()

    output_file = f"./evaluation_results_enhanced_track2_{task_set}.json"
    with open(output_file, "w", encoding="utf-8") as file_handle:
        json.dump(evaluation_results, file_handle, indent=4)

    logging.info("Evaluation complete! Results saved to %s", output_file)
    logging.info("Results: %s", evaluation_results)

