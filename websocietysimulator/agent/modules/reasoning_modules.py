from collections import Counter
import json
import re

class ReasoningBase:
    def __init__(self, profile_type_prompt, memory, llm, profile_retriever=None):
        """
        Initialize the reasoning base class
        
        Args:
            profile_type_prompt: Profile type prompt
            memory: Memory module
            llm: LLM instance used to generate reasoning
            profile_retriever: Optional callable returning a user profile dict given user_id
        """
        self.profile_type_prompt = profile_type_prompt
        self.memory = memory
        self.llm = llm
        self.profile_retriever = profile_retriever
    
    def build_prompt_context(self, task_description, user_id: str = None):
        """Build prompt context with user profile and memory examples."""
        user_profile = self.get_user_profile(user_id)
        profile_block = f"\nUser Profile:\n{user_profile}\n" if user_profile else ''
        examples = ''
        if self.memory:
            retrieved_examples = self.memory(task_description)
            if retrieved_examples:
                examples = retrieved_examples
        return examples, profile_block

    def get_user_profile(self, user_id):
        """Retrieve user profile if profile_retriever is available."""
        if not user_id or not self.profile_retriever:
            return ''
        profile = self.profile_retriever(user_id)
        if not profile:
            return ''
        if isinstance(profile, dict):
            return json.dumps(profile, ensure_ascii=False)
        return str(profile)

class ReasoningIO(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= '', user_id: str = None):
        """Execute IO reasoning with examples and user profile."""
    def __call__(self, task_description: str, feedback :str= '', user_id: str = None):
        examples, profile_block = self.build_prompt_context(task_description, user_id=user_id)
        prompt = '''Your instructions must follow the examples.
Here are some examples.
{examples}
{profile_block}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples, profile_block=profile_block)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        
        return reasoning_result
    
class ReasoningCOT(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= '', user_id: str = None):
        """Execute chain-of-thought reasoning."""
        examples, profile_block = self.build_prompt_context(task_description, user_id=user_id)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
{profile_block}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples, profile_block=profile_block)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result

class ReasoningCOTSC(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= '', user_id: str = None):
        """Execute chain-of-thought with self-consistency (multiple samples)."""
        examples, profile_block = self.build_prompt_context(task_description, user_id=user_id)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
{profile_block}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples, profile_block=profile_block)
        messages = [{"role": "user", "content": prompt}]
        reasoning_results = self.llm(
            messages=messages,
            temperature=0.1,
            n=5
        )
        if not isinstance(reasoning_results, list):
            reasoning_results = [str(reasoning_results)]
        reasoning_results = [str(r) if not isinstance(r, str) else r for r in reasoning_results]
        
        string_counts = Counter(reasoning_results)
        reasoning_result = string_counts.most_common(1)[0][0]
        return reasoning_result
    
class ReasoningTOT(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= '', user_id: str = None):
        """Execute tree-of-thoughts reasoning with voting."""
        examples, profile_block = self.build_prompt_context(task_description, user_id=user_id)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
{profile_block}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples, profile_block=profile_block)
        messages = [{"role": "user", "content": prompt}]
        reasoning_results = self.llm(
            messages=messages,
            temperature=0.1,
            n=3
        )
        if not isinstance(reasoning_results, list):
            reasoning_results = [str(reasoning_results)]
        reasoning_results = [str(r) if not isinstance(r, str) else r for r in reasoning_results]
        
        reasoning_result = self.get_votes(task_description, reasoning_results, examples)
        return reasoning_result
    
    def get_votes(self, task_description, reasoning_results, examples):
        """Vote on best reasoning result from multiple candidates."""
        if reasoning_results and isinstance(reasoning_results[0], str) and 'think' in reasoning_results[0].lower():
            return reasoning_results[0]
        prompt = '''Given the reasoning process for two completed tasks and one ongoing task, and several answers for the next step, decide which answer best follows the reasoning process for example command format. Output "The best answer is {{s}}", where s is the integer id chosen.
Here are some examples.
{examples}
Here is the task:
{task_description}

'''     
        prompt = prompt.format(task_description=task_description, examples=examples)
        for i, y in enumerate(reasoning_results, 1):
            prompt += f'Answer {i}:\n{y}\n'
        messages = [{"role": "user", "content": prompt}]
        vote_outputs = self.llm(
            messages=messages,
            temperature=0.7,
            n=5
        )
        vote_results = [0] * len(reasoning_results)
        for vote_output in vote_outputs:
            pattern = r".*best answer is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(len(reasoning_results)):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        ids = list(range(len(reasoning_results)))
        select_id = sorted(ids, key=lambda x: vote_results[x], reverse=True)[0]
        return reasoning_results[select_id]

class ReasoningDILU(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= '', user_id: str = None):
        """Execute DILU reasoning with system prompt."""
        examples, profile_block = self.build_prompt_context(task_description, user_id=user_id)
        messages = [
            {
                "role": "system",
                "content": '''You are ChatGPT, a large language model trained by OpenAI. Now you act as a real human user on Yelp. You will be given a detailed description of the scenario of current frame along with your history of previous decisions. 
'''
            },
            {
                "role": "user",
                "content": f'''Above messages are some examples of how you make a step successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a step for the current scenario. Your instructions must follow the examples.
Here are two examples.
{examples}
{profile_block}
Here is the task:
{task_description}'''
            }
        ]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result

class ReasoningSelfRefine(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= '', user_id: str = None):
        """Execute reasoning with self-refinement step."""
        examples, profile_block = self.build_prompt_context(task_description, user_id=user_id)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
{profile_block}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples, profile_block=profile_block)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        reasoning_result = self.refine(reasoning_result)
        return reasoning_result
    
    def refine(self, reasoning_result):
        """Refine reasoning result through reflection."""
        prompt = f'''Reflect on the reasoning process and identify any potential errors or areas for improvement. Provide a revised version of the reasoning if necessary.
Here is the original reasoning:
{reasoning_result}
'''     
        messages = [{"role": "user", "content": prompt}]
        feedback_result = self.llm(
            messages=messages,
            temperature=0.0
        )
        return feedback_result
        
class ReasoningStepBack(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= '', user_id: str = None):
        """Execute step-back reasoning by extracting principles first."""
        examples, profile_block = self.build_prompt_context(task_description, user_id=user_id)
        self.principle = self.stepback(task_description)
            
        prompt = f'''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}{profile_block}{self.principle}
Here is the task:
{task_description}'''
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result
    
    def stepback(self, task_description):
        """Extract general principles from task description."""
        stepback_prompt = f'''What common sense, instruction structure is involved in solving this task?
{task_description}'''
        messages = [{"role": "user", "content": stepback_prompt}]
        principle = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return principle
    

