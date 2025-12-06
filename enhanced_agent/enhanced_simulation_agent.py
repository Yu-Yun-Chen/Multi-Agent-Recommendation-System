"""
Enhanced Simulation Agent using Planning, Reasoning, and Memory Modules

This agent demonstrates proper usage of the modular architecture available in the
websocietysimulator framework for Track 1 (User Behavior Simulation).
"""

import json
import logging
from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import (
    PlanningIO, PlanningVoyager, PlanningOPENAGI, PlanningHUGGINGGPT, PlanningTD
)
from websocietysimulator.agent.modules.reasoning_modules import (
    ReasoningCOT, ReasoningSelfRefine, ReasoningDILU, ReasoningStepBack, ReasoningCOTSC
)
from websocietysimulator.agent.modules.memory_modules import (
    MemoryDILU, MemoryGenerative, MemoryVoyager, MemoryTP
)

logging.basicConfig(level=logging.INFO)


class EnhancedPlanning(PlanningIO):
    """
    Custom planning module that creates a structured plan for user behavior simulation.
    Inherits from PlanningIO for basic planning capabilities.
    """
    
    def __init__(self, llm):
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """
        Create a planning prompt specifically for user behavior simulation.
        
        The plan will include steps to:
        1. Gather user profile information
        2. Gather item/business information
        3. Retrieve relevant reviews from other users
        4. Retrieve user's own review history
        5. Synthesize all information to generate review
        """
        if feedback == '':
            prompt = '''You are a planning agent for a user behavior simulation task. 
Your goal is to create a systematic plan to gather all necessary information before simulating a user's review.

Here are example plans:
sub-task 1: {"description": "Retrieve the target user's profile and historical behavior", "reasoning instruction": "Understand the user's preferences, writing style, and rating patterns"}
sub-task 2: {"description": "Retrieve the target item/business details", "reasoning instruction": "Understand what the business offers and its characteristics"}
sub-task 3: {"description": "Retrieve reviews from other users about this item", "reasoning instruction": "Understand what others think and identify key aspects to comment on"}
sub-task 4: {"description": "Retrieve the user's own review history", "reasoning instruction": "Identify the user's personal writing style and rating patterns"}
sub-task 5: {"description": "Synthesize all information to generate an authentic review", "reasoning instruction": "Generate a review that matches the user's style while being relevant to the business"}

Now create a plan for this task:
Task Type: {task_type}
Task Description: {task_description}

Output your plan as a series of sub-tasks in the same format as the example above.
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = f'''You are a planning agent for a user behavior simulation task.
Based on the following feedback, adjust your planning strategy:

Feedback: {feedback}

Task Type: {task_type}
Task Description: {task_description}

Create an improved plan that addresses the feedback.
'''
        return prompt


class EnhancedReasoning(ReasoningCOT):
    """
    Custom reasoning module that uses Chain-of-Thought reasoning
    to generate realistic user reviews.
    """
    
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
    
    def __call__(self, task_description: str, feedback: str = ''):
        """
        Generate reasoning with detailed step-by-step analysis.
        """
        prompt = '''You are simulating a real user on a review platform. 
Think step-by-step to generate an authentic review.

Step 1: Analyze the user's profile and historical patterns
Step 2: Consider the business characteristics
Step 3: Reflect on what other reviewers have said
Step 4: Match your response to the user's writing style
Step 5: Generate appropriate rating and review text

Task Description:
{task_description}

Let's think through this step by step, then provide your final output in this format:
stars: [rating as float: 1.0, 2.0, 3.0, 4.0, or 5.0]
review: [your review text, 2-4 sentences]
'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.7,  # Slightly higher temperature for more natural reviews
            max_tokens=1500
        )
        
        return reasoning_result


class EnhancedSimulationAgent(SimulationAgent):
    """
    Enhanced Simulation Agent that demonstrates proper usage of
    Planning, Reasoning, and Memory modules.
    
    This agent:
    1. Uses a planning module to create a structured information gathering plan
    2. Uses a reasoning module with chain-of-thought for review generation
    3. Uses memory to store and retrieve relevant reviews for better context
    """
    
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        
        self.planning = EnhancedPlanning(llm=self.llm)
        
        self.reasoning = EnhancedReasoning(
            profile_type_prompt='You are a realistic user on a review platform.',
            llm=self.llm
        )
        
        self.memory = MemoryDILU(llm=self.llm)
    
    def workflow(self):
        """
        Main workflow for simulating user behavior.
        
        Returns:
            dict: Contains 'stars' (float) and 'review' (str)
        """
        try:
            plan = [
                {
                    'description': 'Retrieve the target user profile and behavior',
                    'reasoning instruction': 'Understand user preferences and patterns'
                },
                {
                    'description': 'Retrieve the target business information',
                    'reasoning instruction': 'Understand business characteristics'
                },
                {
                    'description': 'Retrieve relevant reviews from other users',
                    'reasoning instruction': 'Learn what others think about this business'
                },
                {
                    'description': 'Retrieve user own review history',
                    'reasoning instruction': 'Match the user writing style'
                }
            ]
            
            user_info = None
            business_info = None
            other_reviews = []
            user_reviews = []
            
            for sub_task in plan:
                description = sub_task['description'].lower()
                
                if 'user profile' in description or 'user behavior' in description:
                    user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
                
                elif 'business' in description or 'item' in description:
                    business_info = self.interaction_tool.get_item(item_id=self.task['item_id'])
                
                elif 'other users' in description or 'relevant reviews' in description:
                    reviews = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
                    if reviews:
                        for review in reviews[:20]:
                            if 'text' in review and review['text']:
                                self.memory(f"review: {review['text']}")
                                other_reviews.append(review)
                
                elif 'user.*review history' in description or 'own review' in description:
                    user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            relevant_context = ""
            if user_reviews and len(user_reviews) > 0:
                sample_review = user_reviews[0].get('text', '') if user_reviews[0].get('text') else ''
                if sample_review:
                    relevant_context = self.memory(sample_review[:200])
            
            user_summary = self._summarize_user(user_info, user_reviews)
            
            business_summary = self._summarize_business(business_info)
            
            other_reviews_summary = self._summarize_reviews(other_reviews, limit=5)
            
            user_style_summary = self._summarize_user_style(user_reviews, limit=3)
            
            task_description = f"""
You are simulating user behavior for a review platform.

USER PROFILE:
{user_summary}

BUSINESS/ITEM DETAILS:
{business_summary}

WHAT OTHERS ARE SAYING (sample reviews):
{other_reviews_summary}

USER'S WRITING STYLE (from history):
{user_style_summary}

RELEVANT CONTEXT FROM MEMORY:
{relevant_context if relevant_context else "No similar context found"}

YOUR TASK:
Based on all the information above, generate an authentic review that this user would write for this business.
Consider:
1. The user's rating patterns and preferences
2. The business's characteristics and what it offers
3. What other users have mentioned (positive/negative aspects)
4. The user's personal writing style and tone
5. Be authentic - users give 5 stars when delighted, 1-2 stars when disappointed

Output format:
stars: [1.0, 2.0, 3.0, 4.0, or 5.0]
review: [2-4 sentences matching the user's style]
"""
            
            result = self.reasoning(task_description)
            stars, review_text = self._parse_result(result)
            
            if len(review_text) > 512:
                review_text = review_text[:512]
            
            return {
                "stars": stars,
                "review": review_text
            }
            
        except Exception as e:
            logging.error(f"Error in workflow: {e}", exc_info=True)
            return {
                "stars": 3.0,
                "review": "Good experience overall."
            }
    
    def _summarize_user(self, user_info, user_reviews):
        """Summarize user information for the prompt."""
        if not user_info:
            return "User information not available."
        
        summary = f"User ID: {user_info.get('user_id', 'unknown')}\n"
        
        if 'name' in user_info:
            summary += f"Name: {user_info['name']}\n"
        
        if 'average_stars' in user_info or 'stars' in user_info:
            avg_stars = user_info.get('average_stars') or user_info.get('stars', 'N/A')
            summary += f"Average Rating Given: {avg_stars}\n"
        
        if 'review_count' in user_info:
            summary += f"Total Reviews Written: {user_info['review_count']}\n"
        
        if user_reviews:
            summary += f"Recent Review Count: {len(user_reviews)}"
        
        return summary
    
    def _summarize_business(self, business_info):
        """Summarize business/item information for the prompt."""
        if not business_info:
            return "Business information not available."
        
        summary = ""
        
        if 'name' in business_info:
            summary += f"Name: {business_info['name']}\n"
        elif 'title' in business_info:
            summary += f"Title: {business_info['title']}\n"
        
        if 'stars' in business_info:
            summary += f"Average Rating: {business_info['stars']}\n"
        elif 'average_rating' in business_info:
            summary += f"Average Rating: {business_info['average_rating']}\n"
        
        if 'review_count' in business_info or 'rating_number' in business_info:
            count = business_info.get('review_count') or business_info.get('rating_number', 'N/A')
            summary += f"Number of Reviews: {count}\n"
        
        if 'categories' in business_info:
            summary += f"Categories: {business_info['categories']}\n"
        
        if 'attributes' in business_info and business_info['attributes']:
            summary += f"Attributes: {str(business_info['attributes'])[:200]}\n"
        
        if 'description' in business_info and business_info['description']:
            desc = business_info['description']
            summary += f"Description: {desc[:300]}...\n"
        
        return summary
    
    def _summarize_reviews(self, reviews, limit=5):
        """Get a sample of other users' reviews."""
        if not reviews:
            return "No reviews from other users available."
        
        summary = ""
        for i, review in enumerate(reviews[:limit]):
            if 'text' in review and review['text']:
                stars = review.get('stars', 'N/A')
                text = review['text'][:150]  # Limit length
                summary += f"{i+1}. ({stars} stars) {text}...\n"
        
        return summary
    
    def _summarize_user_style(self, user_reviews, limit=3):
        """Extract user's writing style from their review history."""
        if not user_reviews:
            return "No previous reviews from this user."
        
        summary = "User's previous reviews:\n"
        for i, review in enumerate(user_reviews[:limit]):
            if 'text' in review and review['text']:
                stars = review.get('stars', 'N/A')
                text = review['text'][:200]
                summary += f"{i+1}. ({stars} stars) {text}...\n"
        
        return summary
    
    def _parse_result(self, result):
        """Parse the LLM result to extract stars and review text."""
        try:
            lines = result.split('\n')
            
            stars = 3.0
            review_text = "Good experience."
            
            for line in lines:
                if 'stars:' in line.lower():
                    parts = line.split(':')
                    if len(parts) > 1:
                        star_str = parts[1].strip()
                        star_str = ''.join(c for c in star_str if c.isdigit() or c == '.')
                        if star_str:
                            stars = float(star_str)
                            stars = max(1.0, min(5.0, stars))
                            stars = round(stars * 2) / 2
                            if stars not in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                                stars = round(stars)
                
                elif 'review:' in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        review_text = parts[1].strip()
            
            if review_text == "Good experience.":
                for line in lines:
                    if len(line) > 20 and ':' not in line[:20]:
                        review_text = line.strip()
                        break
            
            return stars, review_text
            
        except Exception as e:
            logging.error(f"Error parsing result: {e}")
            return 3.0, "Good experience overall."
    
    def workflow_with_dilu_reasoning(self):
        """
        Alternative workflow using ReasoningDILU module.
        
        DILU uses system prompts and focuses on acting as a realistic user.
        Good for capturing authentic user behavior patterns.
        
        Returns:
            dict: Contains 'stars' (float) and 'review' (str)
        """
        try:
            dilu_reasoning = ReasoningDILU(
                profile_type_prompt='You are a realistic user on a review platform.',
                memory=self.memory,
                llm=self.llm
            )
            
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            business_info = self.interaction_tool.get_item(item_id=self.task['item_id'])
            
            other_reviews = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in other_reviews[:20]:
                if 'text' in review and review['text']:
                    self.memory(f"review: {review['text']}")
            
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            relevant_context = ""
            if user_reviews and len(user_reviews) > 0:
                sample_review = user_reviews[0].get('text', '')
                if sample_review:
                    relevant_context = self.memory(sample_review[:200])
            
            user_summary = self._summarize_user(user_info, user_reviews)
            business_summary = self._summarize_business(business_info)
            other_reviews_summary = self._summarize_reviews(other_reviews, limit=5)
            user_style_summary = self._summarize_user_style(user_reviews, limit=3)
            
            task_description = f"""
You are simulating a real user writing a review.

USER PROFILE:
{user_summary}

BUSINESS/ITEM:
{business_summary}

WHAT OTHERS SAY:
{other_reviews_summary}

YOUR PAST REVIEWS:
{user_style_summary}

RELEVANT CONTEXT:
{relevant_context if relevant_context else "No similar context found"}

Generate an authentic review with:
stars: [1.0, 2.0, 3.0, 4.0, or 5.0]
review: [2-4 sentences matching the user's style]
"""
            
            result = dilu_reasoning(task_description)
            stars, review_text = self._parse_result(result)
            
            if len(review_text) > 512:
                review_text = review_text[:512]
            
            return {"stars": stars, "review": review_text}
            
        except Exception as e:
            logging.error(f"Error in DILU workflow: {e}", exc_info=True)
            return {"stars": 3.0, "review": "Good experience overall."}
    
    def workflow_with_self_refine(self):
        """
        Alternative workflow using ReasoningSelfRefine module.
        
        Self-refine generates a review, then reflects and improves it.
        Good for creating higher quality, more polished reviews.
        
        Returns:
            dict: Contains 'stars' (float) and 'review' (str)
        """
        try:
            self_refine_reasoning = ReasoningSelfRefine(
                profile_type_prompt='You are a realistic user on a review platform.',
                memory=self.memory,
                llm=self.llm
            )
            
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            business_info = self.interaction_tool.get_item(item_id=self.task['item_id'])
            
            other_reviews = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            if other_reviews:
                for review in other_reviews[:20]:
                    if 'text' in review and review['text']:
                        self.memory(f"review: {review['text']}")
            
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            # Create task description
            user_summary = self._summarize_user(user_info, user_reviews)
            business_summary = self._summarize_business(business_info)
            other_reviews_summary = self._summarize_reviews(other_reviews, limit=5)
            user_style_summary = self._summarize_user_style(user_reviews, limit=3)
            
            task_description = f"""
You are simulating user behavior for a review platform.

USER PROFILE:
{user_summary}

BUSINESS/ITEM:
{business_summary}

WHAT OTHERS SAY:
{other_reviews_summary}

USER'S PAST STYLE:
{user_style_summary}

Generate a review that matches the user's style:
stars: [1.0, 2.0, 3.0, 4.0, or 5.0]
review: [2-4 sentences]
"""
            
            result = self_refine_reasoning(task_description)
            stars, review_text = self._parse_result(result)
            
            if len(review_text) > 512:
                review_text = review_text[:512]
            
            return {"stars": stars, "review": review_text}
            
        except Exception as e:
            logging.error(f"Error in Self-Refine workflow: {e}", exc_info=True)
            return {"stars": 3.0, "review": "Good experience overall."}
    
    def workflow_with_stepback_reasoning(self):
        """
        Alternative workflow using ReasoningStepBack module.
        
        StepBack first understands general principles about the user,
        then applies them to generate the review.
        Good for principled, consistent review generation.
        
        Returns:
            dict: Contains 'stars' (float) and 'review' (str)
        """
        try:
            stepback_reasoning = ReasoningStepBack(
                profile_type_prompt='You are a realistic user on a review platform.',
                memory=self.memory,
                llm=self.llm
            )
            
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            business_info = self.interaction_tool.get_item(item_id=self.task['item_id'])
            
            other_reviews = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            if other_reviews:
                for review in other_reviews[:20]:
                    if 'text' in review and review['text']:
                        self.memory(f"review: {review['text']}")
            
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            # Create task description
            user_summary = self._summarize_user(user_info, user_reviews)
            business_summary = self._summarize_business(business_info)
            other_reviews_summary = self._summarize_reviews(other_reviews, limit=5)
            user_style_summary = self._summarize_user_style(user_reviews, limit=3)
            
            task_description = f"""
USER PROFILE:
{user_summary}

BUSINESS/ITEM:
{business_summary}

WHAT OTHERS SAY:
{other_reviews_summary}

USER'S WRITING STYLE:
{user_style_summary}

Task: Generate a review matching this user's style and preferences.

Output format:
stars: [1.0, 2.0, 3.0, 4.0, or 5.0]
review: [2-4 sentences]
"""
            
            result = stepback_reasoning(task_description)
            stars, review_text = self._parse_result(result)
            
            if len(review_text) > 512:
                review_text = review_text[:512]
            
            return {"stars": stars, "review": review_text}
            
        except Exception as e:
            logging.error(f"Error in StepBack workflow: {e}", exc_info=True)
            return {"stars": 3.0, "review": "Good experience overall."}
    
    def workflow_with_voyager_memory(self):
        """
        Alternative workflow using MemoryVoyager module.
        
        MemoryVoyager summarizes trajectories before storing, providing
        concise memory retrieval with LLM-generated summaries.
        Good for learning from condensed past patterns.
        
        Returns:
            dict: Contains 'stars' (float) and 'review' (str)
        """
        try:
            voyager_memory = MemoryVoyager(llm=self.llm)
            
            other_reviews = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            if other_reviews:
                for review in other_reviews[:15]:
                    if 'text' in review and review['text']:
                        voyager_memory(f"review: {review['text']}")
            
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            relevant_context = ""
            if user_reviews and len(user_reviews) > 0:
                sample_review = user_reviews[0].get('text', '')
                if sample_review:
                    relevant_context = voyager_memory(sample_review[:200])
            
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            business_info = self.interaction_tool.get_item(item_id=self.task['item_id'])
            
            user_summary = self._summarize_user(user_info, user_reviews)
            business_summary = self._summarize_business(business_info)
            other_reviews_summary = self._summarize_reviews(other_reviews, limit=5)
            user_style_summary = self._summarize_user_style(user_reviews, limit=3)
            
            task_description = f"""
USER PROFILE:
{user_summary}

BUSINESS/ITEM:
{business_summary}

WHAT OTHERS SAY:
{other_reviews_summary}

USER'S WRITING STYLE:
{user_style_summary}

VOYAGER MEMORY INSIGHTS (summarized):
{relevant_context if relevant_context else "No similar context found"}

Generate an authentic review:
stars: [1.0, 2.0, 3.0, 4.0, or 5.0]
review: [2-4 sentences matching the user's style]
"""
            
            result = self.reasoning(task_description)
            stars, review_text = self._parse_result(result)
            
            if len(review_text) > 512:
                review_text = review_text[:512]
            
            return {"stars": stars, "review": review_text}
            
        except Exception as e:
            logging.error(f"Error in Voyager Memory workflow: {e}", exc_info=True)
            return {"stars": 3.0, "review": "Good experience overall."}
    
    def workflow_with_openagi_planning(self):
        """
        Alternative workflow using PlanningOPENAGI module.
        
        OpenAGI creates minimal, efficient todo lists for the task.
        Good for streamlined, efficient review generation.
        
        Returns:
            dict: Contains 'stars' (float) and 'review' (str)
        """
        try:
            openagi_planning = PlanningOPENAGI(llm=self.llm)
            
            plan_task = f"Simulate a review from user {self.task['user_id']} for item {self.task['item_id']}"
            plan = openagi_planning(
                task_type='Review Simulation',
                task_description=plan_task,
                feedback='',
                few_shot='sub-task 1: {"description": "Get user profile", "reasoning instruction": "Understand preferences"}'
            )
            
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            business_info = self.interaction_tool.get_item(item_id=self.task['item_id'])
            
            other_reviews = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            if other_reviews:
                for review in other_reviews[:15]:
                    if 'text' in review and review['text']:
                        self.memory(f"review: {review['text']}")
            
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            user_summary = self._summarize_user(user_info, user_reviews)
            business_summary = self._summarize_business(business_info)
            other_reviews_summary = self._summarize_reviews(other_reviews, limit=5)
            user_style_summary = self._summarize_user_style(user_reviews, limit=3)
            
            task_description = f"""
USER: {user_summary}
BUSINESS: {business_summary}
OTHERS SAY: {other_reviews_summary}
USER STYLE: {user_style_summary}

Generate review:
stars: [1.0-5.0]
review: [2-4 sentences]
"""
            
            result = self.reasoning(task_description)
            stars, review_text = self._parse_result(result)
            
            if len(review_text) > 512:
                review_text = review_text[:512]
            
            return {"stars": stars, "review": review_text}
            
        except Exception as e:
            logging.error(f"Error in OpenAGI workflow: {e}", exc_info=True)
            return {"stars": 3.0, "review": "Good experience overall."}
    
    def workflow_hybrid_advanced(self):
        """
        Advanced hybrid workflow combining multiple sophisticated modules:
        - PlanningHUGGINGGPT for dependency-aware planning
        - MemoryGenerative for importance-scored memory retrieval
        - ReasoningCOTSC for self-consistent review generation
        
        This is the most sophisticated approach, maximizing review quality.
        Note: More expensive due to multiple LLM calls.
        
        Returns:
            dict: Contains 'stars' (float) and 'review' (str)
        """
        try:
            huggingpt_planning = PlanningHUGGINGGPT(llm=self.llm)
            generative_memory = MemoryGenerative(llm=self.llm)
            cotsc_reasoning = ReasoningCOTSC(
                profile_type_prompt='You are a realistic user on a review platform.',
                memory=generative_memory,
                llm=self.llm
            )
            
            plan_task = f"Simulate review considering dependencies: user analysis → business analysis → review generation"
            plan = huggingpt_planning(
                task_type='Review Simulation',
                task_description=plan_task,
                feedback='',
                few_shot='sub-task 1: {"description": "Analyze user first", "reasoning instruction": "Must complete before writing review"}'
            )
            
            user_info = self.interaction_tool.get_user(user_id=self.task['user_id'])
            business_info = self.interaction_tool.get_item(item_id=self.task['item_id'])
            
            other_reviews = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            if other_reviews:
                for review in other_reviews[:15]:
                    if 'text' in review and review['text']:
                        generative_memory(f"review: {review['text']}")
            
            user_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            
            relevant_context = ""
            if user_reviews and len(user_reviews) > 0:
                sample_review = user_reviews[0].get('text', '')
                if sample_review:
                    relevant_context = generative_memory(sample_review[:200])
            
            user_summary = self._summarize_user(user_info, user_reviews)
            business_summary = self._summarize_business(business_info)
            other_reviews_summary = self._summarize_reviews(other_reviews, limit=5)
            user_style_summary = self._summarize_user_style(user_reviews, limit=3)
            
            task_description = f"""
USER PROFILE:
{user_summary}

BUSINESS/ITEM:
{business_summary}

WHAT OTHERS SAY:
{other_reviews_summary}

USER'S WRITING STYLE:
{user_style_summary}

HIGH-IMPORTANCE MEMORY CONTEXT:
{relevant_context if relevant_context else "No similar context found"}

Generate an authentic review with self-consistency:
stars: [1.0, 2.0, 3.0, 4.0, or 5.0]
review: [2-4 sentences matching the user's style]
"""
            
            result = cotsc_reasoning(task_description)
            stars, review_text = self._parse_result(result)
            
            if len(review_text) > 512:
                review_text = review_text[:512]
            
            return {"stars": stars, "review": review_text}
            
        except Exception as e:
            logging.error(f"Error in Hybrid Advanced workflow: {e}", exc_info=True)
            return {"stars": 3.0, "review": "Good experience overall."}


if __name__ == "__main__":
    task_set = "goodreads"
    data_dir = "../data_processed"
    
    simulator = Simulator(
        data_dir=data_dir,
        device="auto",
        cache=True
    )
    
    simulator.set_task_and_groundtruth(
        task_dir=f"../example/track1/{task_set}/tasks",
        groundtruth_dir=f"../example/track1/{task_set}/groundtruth"
    )
    
    simulator.set_agent(EnhancedSimulationAgent)
    simulator.set_llm(InfinigenceLLM(api_key="your_api_key_here"))
    
    outputs = simulator.run_simulation(
        number_of_tasks=10,
        enable_threading=True,
        max_workers=5
    )
    
    evaluation_results = simulator.evaluate()
    
    output_file = f'./evaluation_results_enhanced_track1_{task_set}.json'
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4)

