import re
import ast
import logging

SCHEMA_HINT = """Return ONLY valid JSON, no prose.
Example format:
[
  {
    "description": "concise subtask description",
    "reasoning instruction": "how to think about it",
    "tool instruction": "tool usage guidance"
  }
]"""

STRICT_SCHEMA_INSTRUCTIONS = f"""STRICT FORMAT REQUIREMENTS:
1. Output MUST be valid JSON.
2. Output MUST be a list of objects following this schema:
{SCHEMA_HINT}
3. Do NOT include extra commentary or markdown."""

# Concise field guide for recommendation tasks
FIELD_GUIDE = """
Datasets: user (user_id, review_count, average_stars), item (item_id, title, description, average_rating, authors), review (user_id, item_id, stars, text).
Goal: Generate subgoals to:
1. Analyze user profile, preferences, and review history
2. Examine candidate item metadata, book characteristics, and categories
3. Retrieve and compare information to rank candidates by relevance
"""

def build_few_shot_block(few_shot):
    """Build a formatted few-shot examples block if provided."""
    if few_shot in [None, "", []]:
        return ""
    return f"Examples:\n{few_shot}\n\n"

class PlanningBase():
    def __init__(self, llm):
        """
        Initialize the planning base class
        
        Args:
            llm: LLM instance used to generate planning
        """
        self.plan = []
        self.llm = llm
        self.last_raw_output = ""
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        raise NotImplementedError("Subclasses should implement this method")
    
    def __call__(self, task_type, task_description, feedback, few_shot='few_shot'):
        """Generate plan from task description."""
        prompt = self.create_prompt(task_type, task_description, feedback, few_shot)
        
        messages = [{"role": "user", "content": prompt}]
        string = self.llm(
            messages=messages,
            temperature=0.1
        )
        self.last_raw_output = string
        
        try:
            dict_strings = re.findall(r"\{[^{}]*\}", string)
            dicts = [ast.literal_eval(ds) for ds in dict_strings]
        except (ValueError, SyntaxError) as exc:
            raise
        self.plan = dicts
        return self.plan
    
class PlanningIO(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Create prompt for IO planning."""
        few_shot_block = build_few_shot_block(few_shot)
        field_guide_block = FIELD_GUIDE if task_type == "recommendation" else ""
        
        if feedback == '':
            prompt = f'''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}Task: {task_description}
'''
        else:
            prompt = f'''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt

class PlanningDEPS(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Create prompt for DEPS multi-hop planning."""
        few_shot_block = build_few_shot_block(few_shot)
        field_guide_block = FIELD_GUIDE if task_type == "recommendation" else ""
        
        if feedback == '':
            prompt = f'''You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}Task: {task_description}
'''
        else:
            prompt = f'''You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt

class PlanningTD(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Create prompt for temporal dependency planning."""
        few_shot_block = build_few_shot_block(few_shot)
        field_guide_block = FIELD_GUIDE if task_type == "recommendation" else ""
        
        if feedback == '':
            prompt = f'''You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}Task: {task_description}
'''
        else:
            prompt = f'''You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt

class PlanningVoyager(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Create prompt for Voyager subgoal planning."""
        few_shot_block = build_few_shot_block(few_shot)
        field_guide_block = FIELD_GUIDE if task_type == "recommendation" else ""
        
        if feedback == '':
            prompt = f'''You are a helpful assistant that generates subgoals to complete any {task_type} task specified by me.
I'll give you a final task, you need to decompose the task into a list of subgoals.
You must follow the following criteria:
1) Return a list of subgoals that can be completed in order to complete the specified task.
2) Give the reasoning instructions for each subgoal and the instructions for calling the tool. 
You also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}Task: {task_description}
'''
        else:
            prompt = f'''You are a helpful assistant that generates subgoals to complete any {task_type} task specified by me.
I'll give you a final task, you need to decompose the task into a list of subgoals.
You must follow the following criteria:
1) Return a list of subgoals that can be completed in order to complete the specified task.
2) Give the reasoning instructions for each subgoal and the instructions for calling the tool. 
You also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}end
--------------------
reflexion:{feedback}
task:{task_description}
'''
        return prompt

class PlanningOPENAGI(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Create prompt for OpenAGI todo-list planning."""
        few_shot_block = build_few_shot_block(few_shot)
        field_guide_block = FIELD_GUIDE if task_type == "recommendation" else ""
        
        if feedback == '':
            prompt = f'''You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}Task: {task_description}
'''
        else:
            prompt = f'''You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt

class PlanningHUGGINGGPT(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Create prompt for HuggingGPT dependency-aware planning."""
        few_shot_block = build_few_shot_block(few_shot)
        field_guide_block = FIELD_GUIDE if task_type == "recommendation" else ""
        
        if feedback == '':
            prompt = f'''You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}Task: {task_description}
'''
        else:
            prompt = f'''You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.
{field_guide_block}{STRICT_SCHEMA_INSTRUCTIONS}
{few_shot_block}end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
        return prompt