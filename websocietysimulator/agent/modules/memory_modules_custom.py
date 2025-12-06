import os
import re
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import shutil
import uuid


class MemoryBase:
    def __init__(self, memory_type: str, llm, reset: bool = False) -> None:
        """
        Initialize the memory base class

        Args:
            memory_type: Type of memory
            llm: LLM instance used to generate memory-related text
        """
        self.llm = llm
        self.embedding = self.llm.get_embedding_model()
        db_path = os.path.join("./db", memory_type)

        if reset and os.path.exists(db_path):
            shutil.rmtree(db_path)

        try:
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path,
            )
        except Exception as e:
            import logging
            logger = logging.getLogger("websocietysimulator")
            logger.warning(f"ChromaDB initialization failed for {db_path}, resetting: {e}")
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path,
            )

    def __call__(self, current_situation: str = ""):
        """Route calls to addMemory or retriveMemory based on input format."""
        if "review:" in current_situation:
            self.addMemory(current_situation.replace("review:", ""))
        else:
            return self.retriveMemory(current_situation)

    def retriveMemory(self, query_scenario: str):
        """Retrieve memory based on query scenario."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def addMemory(self, current_situation: str):
        """Add memory entry."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class MemoryDILU(MemoryBase):
    def __init__(self, llm, reset: bool = False):
        """Initialize DILU memory with dedicated long-term store."""
        super().__init__(memory_type="dilu_longterm", llm=llm, reset=reset)

    def retriveMemory(self, query_scenario: str):
        """Retrieve most similar memory trajectory."""
        task_name = query_scenario

        if self.scenario_memory._collection.count() == 0:
            return ""

        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1
        )

        task_trajectories = [
            result[0].metadata["task_trajectory"] for result in similarity_results
        ]

        return "\n".join(task_trajectories)

    def addMemory(self, current_situation: str):
        """Add trajectory to memory store."""
        task_name = current_situation

        memory_doc = Document(
            page_content=task_name,
            metadata={"task_name": task_name, "task_trajectory": current_situation},
        )

        self.scenario_memory.add_documents([memory_doc])


class MemoryGenerative(MemoryBase):
    def __init__(self, llm, reset: bool = False):
        """Initialize generative memory with long-term store."""
        super().__init__(memory_type="generative_longterm", llm=llm, reset=reset)

    def retriveMemory(self, query_scenario: str):
        """Retrieve memory with importance scoring."""
        task_name = query_scenario

        if self.scenario_memory._collection.count() == 0:
            return ""

        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=3
        )

        fewshot_results = []
        importance_scores = []

        for result in similarity_results:
            trajectory = result[0].metadata["task_trajectory"]
            fewshot_results.append(trajectory)

            prompt = f"""You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Your output format should be:
Score: """

            response = self.llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                stop_strs=["\n"],
            )
            score = (
                int(re.search(r"\d+", response).group())
                if re.search(r"\d+", response)
                else 0
            )
            importance_scores.append(score)

        max_score_idx = importance_scores.index(max(importance_scores))
        return similarity_results[max_score_idx][0].metadata["task_trajectory"]

    def addMemory(self, current_situation: str):
        """Add trajectory to memory store."""
        task_name = current_situation

        memory_doc = Document(
            page_content=task_name,
            metadata={"task_name": task_name, "task_trajectory": current_situation},
        )

        self.scenario_memory.add_documents([memory_doc])


class MemoryTP(MemoryBase):
    def __init__(self, llm, reset: bool = False):
        """Initialize TP memory with long-term store."""
        super().__init__(memory_type="tp_longterm", llm=llm, reset=reset)

    def retriveMemory(self, query_scenario: str):
        """Retrieve memory and generate plan based on similar experiences."""
        task_name = query_scenario

        if self.scenario_memory._collection.count() == 0:
            return ""

        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1
        )

        experience_plans = []
        task_description = query_scenario

        for result in similarity_results:
            prompt = f"""You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather use the successful case to think about the strategy and path you took to attempt to complete the task in the ongoing task. Devise a concise, new plan of action that accounts for your task with reference to specific actions that you should have taken. You will need this later to solve the task. Give your plan after "Plan".
Success Case:
{result[0].metadata['task_trajectory']}
Ongoing task:
{task_description}
Plan:
"""
            experience_plans.append(self.llm(messaage=prompt, temperature=0.1))

        return "Plan from successful attempt in similar task:\n" + "\n".join(
            experience_plans
        )

    def addMemory(self, current_situation: str):
        """Add trajectory to memory store."""
        task_name = current_situation

        memory_doc = Document(
            page_content=task_name,
            metadata={"task_name": task_name, "task_trajectory": current_situation},
        )

        self.scenario_memory.add_documents([memory_doc])


class MemoryVoyager(
    MemoryBase,
):
    def __init__(self, llm, reset: bool = False):
        """Initialize Voyager memory with long-term store."""
        super().__init__(memory_type="voyager_longterm", llm=llm, reset=reset)

    def retriveMemory(self, query_scenario: str):
        """Retrieve summarized memory trajectories."""
        task_name = query_scenario

        if self.scenario_memory._collection.count() == 0:
            return ""

        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1
        )

        memory_trajectories = [
            result[0].metadata["task_trajectory"] for result in similarity_results
        ]

        return "\n".join(memory_trajectories)

    def addMemory(self, current_situation: str):
        """Add summarized trajectory to memory store."""
        voyager_prompt = """You are a helpful assistant that writes a description of the task resolution trajectory.

        1) Try to summarize the trajectory in no more than 6 sentences.
        2) Your response should be a single line of text.

        For example:

Please fill in this part yourself

        Trajectory:
        """

        prompt = voyager_prompt + current_situation
        trajectory_summary = self.llm(
            messages=[{"role": "user", "content": prompt}], temperature=0.1
        )

        doc = Document(
            page_content=trajectory_summary,
            metadata={
                "task_description": trajectory_summary,
                "task_trajectory": current_situation,
            },
        )

        self.scenario_memory.add_documents([doc])