"""Baseline recommendation agent test script."""

import json
import logging
import os
import random
import re
import sys
import time

from dotenv import load_dotenv
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningIO
from websocietysimulator.agent.modules.info_orchestrator_module import InfoOrchestrator
from websocietysimulator.agent.modules.schemafitter_module import SchemaFitterIO
from websocietysimulator.agent.modules.pairwise_modules import PairwiseRanker
from google_gemini_llm import GoogleGeminiLLM
from websocietysimulator.agent.modules.planning_modules_custom import PlanningVoyagerCustom
from websocietysimulator.agent.modules.memory_modules_custom import MemoryDILU

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, CURRENT_DIR)

logs_dir = os.path.join(ROOT, "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"sim_run_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

USE_PAIRWISE = True
PAIRWISE_TOP_K = 5


def parse_ranked_list(reasoning_output, candidate_list):
    """Parse ranked list from reasoning output."""
    ranked_list = []
    try:
        match = re.search(r"\[.*?\]", reasoning_output, re.DOTALL)
        if match:
            list_str = match.group(0)
            parsed = eval(list_str)
            if isinstance(parsed, list):
                ranked_list = parsed
    except Exception:
        ranked_list = []

    seen = set()
    ranked_filtered = []
    for cid in ranked_list:
        if isinstance(cid, str) and cid in candidate_list and cid not in seen:
            seen.add(cid)
            ranked_filtered.append(cid)
    for cid in candidate_list:
        if cid not in seen:
            ranked_filtered.append(cid)
    return ranked_filtered


def build_pairwise_context(interaction_tool, user_id, top_ids):
    """Build context for pairwise ranking."""
    raw_user_info = interaction_tool.get_user(user_id=user_id)
    user_reviews = interaction_tool.get_reviews(user_id=user_id)

    clean_history = []
    if isinstance(user_reviews, list):
        for r in user_reviews[:10]:
            clean_history.append({
                "item_id": r.get("item_id"),
                "rating": r.get("stars", r.get("rating", "N/A")),
                "text": r.get("text", "")[:200],
            })

    rich_user_profile = {
        "basic_info": raw_user_info,
        "history_reviews": clean_history,
        "instruction": "Please infer user taste from history_reviews.",
    }

    item_profiles_for_pairwise = []
    for item_id in top_ids:
        info = interaction_tool.get_item(item_id=item_id)
        if info:
            info["item_id"] = item_id
            item_profiles_for_pairwise.append(info)

    return {
        "user_profile": rich_user_profile,
        "item_profiles": item_profiles_for_pairwise,
    }


class MyRecommendationAgent(RecommendationAgent):
    """
    Recommendation agent for track2 using:
    - PlanningVoyagerCustom as planner
    - InfoOrchestrator + SchemaFitterIO for user/item profiles
    - MemoryDILU as long-term memory (trajectory storage)
    - ReasoningIO to rank candidate_list
    - Optional PairwiseRanker for reranking top-K items
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = PlanningVoyagerCustom(llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        self.reasoning = ReasoningIO(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=self.llm,
        )
        self.info_orchestrator = InfoOrchestrator(
            memory=None,
            llm=self.llm,
            schema_fitter=None,
            interaction_tool=None,
            use_fixed_item_params=True,
            max_candidates_to_profile=10,
        )
        self._schema_fitter_llm = self.llm
        self.pairwise_ranker = PairwiseRanker(self.llm) if USE_PAIRWISE else None

    def set_interaction_tool(self, interaction_tool):
        """Wire interaction_tool into InfoOrchestrator."""
        super().set_interaction_tool(interaction_tool)
        if self.info_orchestrator and interaction_tool:
            if self.info_orchestrator.schema_fitter is None:
                schema_fitter = SchemaFitterIO(
                    self._schema_fitter_llm, interaction_tool
                )
                self.info_orchestrator.schema_fitter = schema_fitter
            self.info_orchestrator.interaction_tool = interaction_tool

    def workflow(self):
        """Main workflow for a single recommendation task."""
        task = self.task
        task_description = json.dumps(task, indent=2)
        task_query = json.dumps(task, ensure_ascii=False)
        few_shot = self.memory(task_query) or ""

        plan = self.planning(
            task_type="Recommendation Task",
            task_description=task_description,
            feedback="",
            few_shot=few_shot,
        )

        profiles = self.info_orchestrator(
            planner_steps=plan,
            user_id=task.get("user_id"),
            candidate_list=task.get("candidate_list"),
        )
        user_profile = profiles.get("user_profile")
        item_profiles = profiles.get("item_profiles", [])

        reasoning_context = {
            "task": task,
            "plan": plan,
            "user_profile": user_profile,
            "item_profiles": item_profiles,
            "candidate_list": task["candidate_list"],
        }
        reasoning_prompt = (
            "You are a recommendation system. "
            "Given the JSON context below, rank all items in candidate_list from most to least preferred "
            "for the user. Return ONLY a Python list of item_id strings, in descending preference order, "
            "and each id MUST come from candidate_list. Do not include any explanations. "
            "Your entire output must be very short (no more than 1 line, strictly no analysis).\n\n"
            f"CONTEXT:\n{json.dumps(reasoning_context, ensure_ascii=False)}"
        )

        reasoning_output = self.reasoning(reasoning_prompt)
        ranked_filtered = parse_ranked_list(reasoning_output, task["candidate_list"])

        if self.pairwise_ranker and self.interaction_tool:
            user_id = task.get("user_id")
            top_ids = ranked_filtered[:PAIRWISE_TOP_K]
            pairwise_context = build_pairwise_context(
                self.interaction_tool, user_id, top_ids
            )
            ranked_filtered = self.pairwise_ranker.rerank(
                ranked_filtered, pairwise_context
            )

        return ranked_filtered


def setup_simulator(task_set, data_dir):
    """Setup and configure simulator."""
    simulator = Simulator(data_dir=data_dir, device="auto", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=os.path.join(ROOT, "example", "track2", task_set, "tasks"),
        groundtruth_dir=os.path.join(ROOT, "example", "track2", task_set, "groundtruth"),
    )
    return simulator


def select_random_tasks(simulator, k, seed):
    """Select k random tasks from simulator."""
    random.seed(seed)
    num_tasks_available = len(simulator.tasks)
    if num_tasks_available == 0:
        raise ValueError("No tasks loaded from the specified directory")
    
    k = min(k, num_tasks_available)
    all_indices = list(range(num_tasks_available))
    selected = random.sample(all_indices, k=k)
    simulator.tasks = [simulator.tasks[i] for i in selected]
    simulator.groundtruth_data = [simulator.groundtruth_data[i] for i in selected]
    return k


def save_evaluation_results(evaluation_results, seed, planning_name, reasoning_name, k):
    """Save evaluation results to file."""
    eval_dir = os.path.join(ROOT, "evaluation_result")
    os.makedirs(eval_dir, exist_ok=True)
    
    filename = f"rs{seed}_{planning_name}_{reasoning_name}_{k}.json"
    out_path = os.path.join(eval_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    return out_path


if __name__ == "__main__":
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY in environment variables")

    task_set = "goodreads"
    data_dir = os.path.join(ROOT, "data_processed")
    simulator = setup_simulator(task_set, data_dir)

    k = 50
    seed = 82
    k = select_random_tasks(simulator, k, seed)

    simulator.set_agent(MyRecommendationAgent)
    llm_google = GoogleGeminiLLM(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    simulator.set_llm(llm_google)

    start = time.time()
    simulator.run_simulation(
        number_of_tasks=None,
        enable_threading=False,
        max_workers=1,
    )
    evaluation_results = simulator.evaluate()
    end = time.time()

    total_time = end - start
    num_tasks_run = len(simulator.tasks)

    planning_name = type(MyRecommendationAgent(llm_google).planning).__name__.lower()
    reasoning_name = type(MyRecommendationAgent(llm_google).reasoning).__name__.lower()

    evaluation_results["run_info"] = {
        "planning_module": planning_name,
        "reasoning_module": reasoning_name,
        "random_seed": seed,
        "num_tasks": num_tasks_run,
        "total_time_seconds": total_time,
        "avg_time_per_task_seconds": (
            total_time / num_tasks_run if num_tasks_run > 0 else None
        ),
        "max_workers": 10,
        "max_reviews_per_profile": 70,
        "limit_on_schema_fields(prompt_instruction)": "no more than 50 English words",
    }

    out_path = save_evaluation_results(
        evaluation_results, seed, planning_name, reasoning_name, k
    )
