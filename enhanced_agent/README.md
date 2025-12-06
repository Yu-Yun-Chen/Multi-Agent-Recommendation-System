# Enhanced Agent Modifications

This package implements four key enhancements to the recommendation agent system.

## 1. Dynamic Profile Generation

**Purpose**: Automatically generate user and item profiles based on planner requirements.

**Key Components**:
- `base_agent.py` - Core agent that integrates InfoOrchestrator for profile generation
- `enhanced_recommendation_agent.py` - Main agent using dynamic profiles
- `base_agent_schema_validator.py` - Validates profile schemas
- `utils.py` - Helper functions for profile processing

**How it works**: InfoOrchestrator analyzes planner steps to detect required parameters, then uses SchemaFitterIO to dynamically generate user/item profiles with only the necessary fields. This eliminates manual profile specification.

**Usage**: Profiles are automatically generated in `workflow()` via `self.info_orchestrator(planner_steps, user_id, candidate_list)`.

## 2. Permutation Workflow Tests

**Purpose**: Systematically test different combinations of planning, memory, and reasoning modules.

**Key Components**:
- `test_recommendation_accuracy.py` - Compares multiple workflows and reports accuracy metrics
- `workflow_mixins.py` - Defines 15+ workflow combinations (Voyager, DEPS, TOT, Self-Refine, etc.)
- `enhanced_recommendation_agent.py` - Base agent that supports workflow permutations

**How it works**: Each workflow combines different planning modules (VoyagerCustom, DEPSCustom, etc.), memory modules (DILU, Generative, TP, Voyager), and reasoning modules (IO, COT, Self-Refine, TOT) to find optimal combinations.

**Usage**: Run `python test_recommendation_accuracy.py --workflows voyager self_refine openagi` to compare workflows.

## 3. Pairwise Ranking

**Purpose**: Rerank top-K candidates using pairwise comparisons for improved precision.

**Key Components**:
- `websocietysimulator/agent/modules/pairwise_modules.py` - Implements PairwiseRanker class
- `test_baseline_enhanced_agent.py` - Uses PairwiseRanker for reranking top-5 items

**How it works**: PairwiseRanker uses a "King of the Hill" algorithm: after initial ranking, it compares the top-K items head-to-head. Each challenger competes against the current winner using LLM-based pairwise comparison with rich user profiles (including review history) and detailed item profiles. Only significantly better items replace the current winner.

**Usage**: Set `USE_PAIRWISE = True` in `test_baseline_enhanced_agent.py`. Reranking happens automatically after initial reasoning via `self.pairwise_ranker.rerank(ranked_list, context)`.

## 4. Long Term Memory

**Purpose**: Store and retrieve successful recommendation trajectories for few-shot learning.

**Key Components**:
- `websocietysimulator/agent/modules/memory_modules_longterm.py` - Contains Track2TrainAgent for offline memory training
- `test_baseline_enhanced_agent.py` - Uses MemoryDILU to retrieve trajectories during inference
- `workflow_mixins.py` - Supports multiple memory types (DILU, Generative, TP, Voyager)

**How it works**: Track2TrainAgent runs offline training: it executes full recommendation workflows, evaluates against ground truth, and stores only successful trajectories (where ground truth is in top-5) into ChromaDB. During inference, similar past tasks are retrieved via similarity search and used as few-shot examples for the planner.

**Usage**: 
- **Training**: Run `memory_modules_longterm.py` to build memory from successful trajectories
- **Inference**: Memory is automatically used via `few_shot = self.memory(task_query)` in the workflow

## File Organization

- **Core**: `base_agent.py`, `enhanced_recommendation_agent.py`, `base_agent_schema_validator.py`, `utils.py`
- **Testing**: `test_baseline_enhanced_agent.py`, `test_recommendation_accuracy.py`
- **Workflows**: `workflow_mixins.py`
- **Utilities**: `google_gemini_llm.py`, `enhanced_simulation_agent.py`

