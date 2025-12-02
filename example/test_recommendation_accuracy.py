"""
Test and compare ACCURACY of different recommendation workflows.

This script runs multiple tasks through each workflow and computes actual
accuracy metrics (NDCG@10, Hit Rate@10, Precision@10) to determine which
workflow performs best.

Usage:
    pation_accuracy.py ython test_recommend                                   # Test default workflows
    python test_recommendation_accuracy.py --num-tasks 20                     # Test on 20 tasks
    python test_recommendation_accuracy.py --workflows default self_refine    # Test specific workflows
    python test_recommendation_accuracy.py --dataset amazon                   # Test on amazon data
"""

import argparse
import json
import time
import logging
from GoogleGeminiLLM import GoogleGeminiLLM
from websocietysimulator import Simulator
from EnhancedRecommendationAgent import EnhancedRecommendationAgent

logging.basicConfig(level=logging.INFO, format='%(message)s')


def create_workflow_agent(workflow_name, llm):
    """
    Create an agent class that uses a specific workflow.
    
    Args:
        workflow_name: Name of the workflow to use
        llm: LLM instance
    """
    if workflow_name == 'default':
        return EnhancedRecommendationAgent
    
    class WorkflowAgent(EnhancedRecommendationAgent):
        def workflow(self):
            # Map workflow names to methods
            workflow_map = {
                # Original workflows
                'voyager': self.workflow_with_voyager_planning,
                'self_refine': self.workflow_with_self_refine,
                'cot_sc': self.workflow_with_cot_sc,
                'voyager_memory': self.workflow_with_voyager_memory,
                'openagi': self.workflow_with_openagi_planning,
                'hybrid': self.workflow_hybrid_advanced,
                # New workflows exploring unused modules
                'tot': self.workflow_with_tot_reasoning,
                'td': self.workflow_with_td_planning,
                'deps': self.workflow_with_deps_planning,
                'all_voyager': self.workflow_all_voyager,
                'dilu_memory': self.workflow_with_dilu_memory,
                'simple': self.workflow_simple_efficient,
                'tot_memory': self.workflow_tot_with_memory,
                'deps_refine': self.workflow_deps_self_refine
            }
            
            if workflow_name in workflow_map:
                return workflow_map[workflow_name]()
            else:
                raise ValueError(f"Unknown workflow: {workflow_name}")
    
    return WorkflowAgent


def test_workflow(workflow_name, dataset='goodreads', num_tasks=10, llm_model='gemini-2.0-flash'):
    """
    Test a specific workflow and return accuracy metrics.
    
    Args:
        workflow_name: Name of the workflow to test
        dataset: Dataset to use ('yelp', 'amazon', or 'goodreads')
        num_tasks: Number of tasks to test
        llm_model: LLM model to use
    
    Returns:
        dict: Contains metrics and timing information
    """
    print(f"\n{'='*80}")
    print(f"Testing: {workflow_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Initialize LLM
        llm = GoogleGeminiLLM(model=llm_model)
        
        # Initialize Simulator
        simulator = Simulator(data_dir='../data_processed', cache=True)
        simulator.set_task_and_groundtruth(
            task_dir=f'./track2/{dataset}/tasks',
            groundtruth_dir=f'./track2/{dataset}/groundtruth'
        )
        
        # Set the agent with specific workflow
        WorkflowAgent = create_workflow_agent(workflow_name, llm)
        simulator.set_agent(WorkflowAgent)
        simulator.set_llm(llm)
        
        # Run simulation
        print(f"Running {num_tasks} tasks...")
        start_time = time.time()
        try:
            simulator.run_simulation(number_of_tasks=num_tasks)
            execution_time = time.time() - start_time

            # Evaluate
            print("Evaluating results...")
            metrics = simulator.evaluate()
        except Exception as exc:
            execution_time = time.time() - start_time
            print(f"\n✗ Error during workflow '{workflow_name}': {exc}")
            metrics = {"error": str(exc)}
            return {
                "workflow": workflow_name,
                "metrics": metrics,
                "execution_time": execution_time,
                "avg_time_per_task": execution_time / max(num_tasks, 1),
                "num_tasks": num_tasks,
                "dataset": dataset,
                "success": False,
            }
        
        # Add timing info
        result = {
            'workflow': workflow_name,
            'metrics': metrics,
            'execution_time': execution_time,
            'avg_time_per_task': execution_time / num_tasks,
            'num_tasks': num_tasks,
            'dataset': dataset,
            'success': True
        }
        
        # Print results
        print(f"\n✓ Completed in {execution_time:.1f}s ({execution_time/num_tasks:.1f}s per task)")
        print(f"\nMetrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'workflow': workflow_name,
            'error': str(e),
            'success': False
        }


def compare_workflows(workflows, dataset='goodreads', num_tasks=10, llm_model='gemini-2.0-flash'):
    """
    Compare multiple workflows and determine which performs best.
    
    Args:
        workflows: List of workflow names to compare
        dataset: Dataset to use
        num_tasks: Number of tasks to test each workflow on
        llm_model: LLM model to use
    
    Returns:
        dict: Results for all workflows
    """
    print(f"\n{'='*80}")
    print(f"COMPARING RECOMMENDATION WORKFLOWS")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Number of tasks: {num_tasks}")
    print(f"Model: {llm_model}")
    print(f"Workflows to test: {', '.join(workflows)}")
    
    results = {}
    raw_failures = {}
    
    for workflow_name in workflows:
        result = test_workflow(workflow_name, dataset, num_tasks, llm_model)
        results[workflow_name] = result
        if result and not result.get("success", False):
            raw_failures[workflow_name] = result.get("raw_output") or result["metrics"].get("error")
    
    # Print comparison summary
    print_comparison_summary(results, num_tasks)
    
    import os
    # Make result folder with same naming convention
    output_folder = f'workflow_results_{dataset}_{num_tasks}tasks'
    os.makedirs(output_folder, exist_ok=True)

    # Save results to JSON file inside the folder
    json_file = os.path.join(output_folder, f'workflow_accuracy_results_{dataset}_{num_tasks}tasks.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save readable summary to text file inside the folder
    txt_file = os.path.join(output_folder, f'workflow_summary_{dataset}_{num_tasks}tasks.txt')
    save_readable_summary(results, num_tasks, dataset, txt_file, raw_failures, raw_failures)
    
    print(f"\n{'='*80}")
    print(f"Results saved to:")
    print(f"  - JSON: {json_file}")
    print(f"  - Summary: {txt_file}")
    print(f"{'='*80}")
    
    return results


def save_readable_summary(results, num_tasks, dataset, filename, raw_failures=None, all_failures=None):
    """
    Save a readable summary to a text file.
    """
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"WORKFLOW ACCURACY TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Number of tasks tested: {num_tasks}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Filter successful results
        successful = {k: v for k, v in results.items() if v.get('success', False)}
        
        if not successful:
            f.write("No successful results.\n")
            if all_failures:
                f.write("\nRAW FAILURE OUTPUTS\n")
                f.write("="*80 + "\n")
                for workflow_name, raw_text in all_failures.items():
                    f.write(f"{workflow_name}:\n{raw_text}\n\n")
            return
        
        # Extract metrics
        first_result = next(iter(successful.values()))
        metrics_dict = first_result['metrics']
        
        if 'metrics' in metrics_dict and isinstance(metrics_dict['metrics'], dict):
            metric_names = list(metrics_dict['metrics'].keys())
            nested = True
        else:
            metric_names = list(metrics_dict.keys())
            nested = False
        
        # Write table header
        f.write("="*80 + "\n")
        f.write("RESULTS TABLE\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Workflow':<20} {'Time(s)':<10} ")
        for metric in metric_names:
            f.write(f"{metric:<15} ")
        f.write("\n" + "-"*80 + "\n")
        
        # Write each workflow's results
        for workflow_name, result in successful.items():
            metrics = result['metrics']
            time_taken = result['execution_time']
            
            f.write(f"{workflow_name:<20} {time_taken:>8.1f}s  ")
            
            # Extract nested metrics if needed
            if nested and 'metrics' in metrics:
                actual_metrics = metrics['metrics']
            else:
                actual_metrics = metrics
            
            for metric in metric_names:
                value = actual_metrics.get(metric, 0)
                if isinstance(value, (int, float)):
                    f.write(f"{value:>13.4f}  ")
                else:
                    f.write(f"{str(value):>13s}  ")
            f.write("\n")
        
        # Write best performers
        f.write("\n" + "="*80 + "\n")
        f.write("BEST PERFORMER FOR EACH METRIC\n")
        f.write("="*80 + "\n\n")
        
        for metric in metric_names:
            try:
                def get_metric_value(item):
                    metrics = item[1]['metrics']
                    if nested and 'metrics' in metrics:
                        actual_metrics = metrics['metrics']
                    else:
                        actual_metrics = metrics
                    val = actual_metrics.get(metric, 0)
                    return float(val) if isinstance(val, (int, float)) else 0
                
                best_workflow = max(successful.items(), key=get_metric_value)
                metrics = best_workflow[1]['metrics']
                if nested and 'metrics' in metrics:
                    best_value = metrics['metrics'][metric]
                else:
                    best_value = metrics[metric]
                
                if isinstance(best_value, (int, float)):
                    f.write(f"{metric:<25}: {best_workflow[0]:<15} ({best_value:.4f})\n")
                else:
                    f.write(f"{metric:<25}: {best_workflow[0]:<15} ({best_value})\n")
            except (ValueError, TypeError):
                f.write(f"{metric:<25}: Unable to compare\n")
        
        # Write recommendation
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*80 + "\n\n")
        
        # Use average_hit_rate as primary metric if available
        if 'average_hit_rate' in metric_names:
            primary_metric = 'average_hit_rate'
        else:
            primary_metric = metric_names[0]
        
        try:
            def get_primary_value(item):
                metrics = item[1]['metrics']
                if nested and 'metrics' in metrics:
                    actual_metrics = metrics['metrics']
                else:
                    actual_metrics = metrics
                val = actual_metrics.get(primary_metric, 0)
                return float(val) if isinstance(val, (int, float)) else 0
            
            best_overall = max(successful.items(), key=get_primary_value)
            fastest = min(successful.items(), key=lambda x: x[1]['execution_time'])
            
            f.write(f"🏆 Best Overall (by {primary_metric}): {best_overall[0].upper()}\n")
            
            metrics = best_overall[1]['metrics']
            if nested and 'metrics' in metrics:
                primary_value = metrics['metrics'][primary_metric]
            else:
                primary_value = metrics[primary_metric]
            
            if isinstance(primary_value, (int, float)):
                f.write(f"   {primary_metric}: {primary_value:.4f}\n")
            f.write(f"   Execution time: {best_overall[1]['execution_time']:.1f}s\n\n")
            
            f.write(f"⚡ Fastest Workflow: {fastest[0].upper()}\n")
            f.write(f"   Time: {fastest[1]['execution_time']:.1f}s\n\n")
            
            # Calculate value scores
            f.write("💡 Quality/Time Scores (higher is better):\n")
            value_scores = {}
            for name, result in successful.items():
                metrics = result['metrics']
                if nested and 'metrics' in metrics:
                    metric_val = metrics['metrics'].get(primary_metric, 0)
                else:
                    metric_val = metrics.get(primary_metric, 0)
                
                if isinstance(metric_val, (int, float)) and result['execution_time'] > 0:
                    score = metric_val / (result['execution_time'] / 60)
                    value_scores[name] = score
                    f.write(f"   {name:<20}: {score:.4f}\n")
            
        except (ValueError, TypeError):
            f.write("Unable to determine best overall\n")
        
        f.write("\n" + "="*80 + "\n")

        if raw_failures:
            f.write("RAW FAILURE OUTPUTS\n")
            f.write("="*80 + "\n")
            for workflow_name, raw_text in raw_failures.items():
                f.write(f"{workflow_name}:\n{raw_text}\n\n")


def print_comparison_summary(results, num_tasks):
    """
    Print a nice comparison table of all results.
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY ({num_tasks} tasks each)")
    print(f"{'='*80}\n")
    
    # Filter successful results
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful:
        print("No successful results to compare.")
        return
    
    # Extract nested metrics from the first result
    first_result = next(iter(successful.values()))
    metrics_dict = first_result['metrics']
    
    # Check if metrics contains nested dict
    if 'metrics' in metrics_dict and isinstance(metrics_dict['metrics'], dict):
        # Nested structure - extract the actual metrics
        metric_names = list(metrics_dict['metrics'].keys())
        nested = True
    else:
        # Flat structure
        metric_names = list(metrics_dict.keys())
        nested = False
    
    # Print header
    print(f"{'Workflow':<20} {'Time(s)':<10} ", end='')
    for metric in metric_names:
        # Shorten metric names for display
        short_name = metric.replace('_hit_rate', '').replace('_', ' ').title()[:10]
        print(f"{short_name:<12}", end='')
    print()
    print("-" * (32 + len(metric_names) * 12))
    
    # Print each workflow's results
    for workflow_name, result in successful.items():
        metrics = result['metrics']
        time_taken = result['execution_time']
        
        print(f"{workflow_name:<20} {time_taken:>8.1f}s  ", end='')
        
        # Extract nested metrics if needed
        if nested and 'metrics' in metrics:
            actual_metrics = metrics['metrics']
        else:
            actual_metrics = metrics
        
        for metric in metric_names:
            value = actual_metrics.get(metric, 0)
            if isinstance(value, (int, float)):
                print(f"{value:>10.4f}  ", end='')
            else:
                print(f"{str(value):>10s}  ", end='')
        print()
    
    # Find best for each metric
    print("\n" + "="*80)
    print("BEST PERFORMER FOR EACH METRIC")
    print("="*80)
    
    for metric in metric_names:
        # Find workflow with highest value for this metric
        try:
            def get_metric_value(item):
                metrics = item[1]['metrics']
                if nested and 'metrics' in metrics:
                    actual_metrics = metrics['metrics']
                else:
                    actual_metrics = metrics
                val = actual_metrics.get(metric, 0)
                return float(val) if isinstance(val, (int, float)) else 0
            
            best_workflow = max(successful.items(), key=get_metric_value)
            
            # Get the actual metric value
            metrics = best_workflow[1]['metrics']
            if nested and 'metrics' in metrics:
                best_value = metrics['metrics'][metric]
            else:
                best_value = metrics[metric]
            
            if isinstance(best_value, (int, float)):
                print(f"{metric:<20}: {best_workflow[0]:<15} ({best_value:.4f})")
            else:
                print(f"{metric:<20}: {best_workflow[0]:<15} ({best_value})")
        except (ValueError, TypeError):
            print(f"{metric:<20}: Unable to compare (non-numeric values)")
    
    # Overall recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Use primary metric (average_hit_rate or first available)
    if 'average_hit_rate' in metric_names:
        primary_metric = 'average_hit_rate'
    else:
        primary_metric = metric_names[0]
    
    try:
        def get_primary_value(item):
            metrics = item[1]['metrics']
            if nested and 'metrics' in metrics:
                actual_metrics = metrics['metrics']
            else:
                actual_metrics = metrics
            val = actual_metrics.get(primary_metric, 0)
            return float(val) if isinstance(val, (int, float)) else 0
        
        best_overall = max(successful.items(), key=get_primary_value)
        
        print(f"\n🏆 Best Overall: {best_overall[0].upper()}")
        
        # Get primary value
        metrics = best_overall[1]['metrics']
        if nested and 'metrics' in metrics:
            primary_value = metrics['metrics'][primary_metric]
        else:
            primary_value = metrics[primary_metric]
        
        if isinstance(primary_value, (int, float)):
            print(f"   {primary_metric}: {primary_value:.4f}")
        else:
            print(f"   {primary_metric}: {primary_value}")
        print(f"   Execution time: {best_overall[1]['execution_time']:.1f}s")
        
        # Find fastest
        fastest = min(successful.items(), key=lambda x: x[1]['execution_time'])
        print(f"\n⚡ Fastest: {fastest[0].upper()}")
        print(f"   Time: {fastest[1]['execution_time']:.1f}s")
        
        # Get fastest metric value
        fastest_metrics = fastest[1]['metrics']
        if nested and 'metrics' in fastest_metrics:
            fastest_metric_val = fastest_metrics['metrics'][primary_metric]
        else:
            fastest_metric_val = fastest_metrics[primary_metric]
        
        if isinstance(fastest_metric_val, (int, float)):
            print(f"   {primary_metric}: {fastest_metric_val:.4f}")
        else:
            print(f"   {primary_metric}: {fastest_metric_val}")
        
        # Best value recommendation
        print(f"\n💡 Best Value (quality/time): ", end='')
        value_scores = {}
        for name, result in successful.items():
            metrics = result['metrics']
            if nested and 'metrics' in metrics:
                metric_val = metrics['metrics'].get(primary_metric, 0)
            else:
                metric_val = metrics.get(primary_metric, 0)
            
            if isinstance(metric_val, (int, float)):
                value_scores[name] = metric_val / (result['execution_time'] / 60)
        
        if value_scores:
            best_value = max(value_scores.items(), key=lambda x: x[1])
            print(f"{best_value[0].upper()}")
            print(f"   Score: {best_value[1]:.4f} (higher is better)")
        else:
            print("Unable to calculate")
    except (ValueError, TypeError) as e:
        print(f"\n⚠️  Could not determine best overall")


def main():
    parser = argparse.ArgumentParser(
        description='Test and compare accuracy of different recommendation workflows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default workflows on 10 tasks
  python test_recommendation_accuracy.py
  
  # Test on 20 tasks
  python test_recommendation_accuracy.py --num-tasks 20
  
  # Test specific workflows
  python test_recommendation_accuracy.py --workflows default self_refine openagi
  
  # Test on Amazon data
  python test_recommendation_accuracy.py --dataset amazon --num-tasks 15
  
  # Test expensive workflows (COT-SC, Hybrid)
  python test_recommendation_accuracy.py --workflows hybrid cot_sc --num-tasks 5

Available workflows:

ORIGINAL WORKFLOWS:
  default        - Default workflow (StepBack + Generative Memory)
  voyager        - Voyager Planning
  self_refine    - Self-Refine (iterative improvement)
  cot_sc         - COT with Self-Consistency (expensive: 5x cost)
  voyager_memory - Voyager Memory (summarized patterns)
  openagi        - OpenAGI Planning (fast and cheap)
  hybrid         - Hybrid Advanced (expensive: 3x cost, best quality)

NEW WORKFLOWS (exploring unused modules):
  tot            - Tree of Thoughts reasoning (VERY EXPENSIVE: 8x cost)
  td             - Temporal Dependencies planning
  deps           - Multi-Hop DEPS planning (perfect for recommendations!)
  all_voyager    - Full Voyager stack (planning + reasoning + memory)
  dilu_memory    - HuggingGPT + DILU Memory
  simple         - Minimal/Fast (IO reasoning only, good baseline)
  tot_memory     - TOT + TP Memory (EXTREMELY EXPENSIVE: 8+ calls)
  deps_refine    - DEPS + Self-Refine + Memory (expensive, high quality)
"""
    )
    
    parser.add_argument(
        '--workflows',
        nargs='+',
        default=['default', 'self_refine', 'openagi'],
        choices=['default', 'voyager', 'self_refine', 'cot_sc', 'voyager_memory', 'openagi', 'hybrid',
                 'tot', 'td', 'deps', 'all_voyager', 'dilu_memory', 'simple', 'tot_memory', 'deps_refine', 'all'],
        help='Workflows to test (default: default self_refine openagi). All workflows use InfoOrchestrator for profile generation.'
    )
    
    parser.add_argument(
        '--num-tasks',
        type=int,
        default=10,
        help='Number of tasks to test each workflow on (default: 10)'
    )
    
    parser.add_argument(
        '--dataset',
        choices=['yelp', 'amazon', 'goodreads'],
        default='goodreads',
        help='Dataset to use (default: goodreads)'
    )
    
    parser.add_argument(
        '--model',
        default='gemini-2.0-flash',
        help='LLM model to use (default: gemini-2.0-flash)'
    )
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if 'all' in args.workflows:
        workflows = ['default', 'voyager', 'self_refine', 'cot_sc', 'voyager_memory', 'openagi', 'hybrid',
                     'tot', 'td', 'deps', 'all_voyager', 'dilu_memory', 'simple', 'tot_memory', 'deps_refine']
        print("\n⚠️  WARNING: Testing ALL workflows including VERY EXPENSIVE ones")
        print(f"   Total workflows: {len(workflows)}")
        print(f"   Including: TOT (8x cost), TOT_Memory (8+ calls), COT-SC (5x cost)")
        print(f"   Estimated cost: ${args.num_tasks * 0.50:.2f} or more")
        response = input("   Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    else:
        workflows = args.workflows
        
        # Warn about expensive workflows
        very_expensive = set(workflows) & {'tot', 'tot_memory'}
        expensive = set(workflows) & {'cot_sc', 'hybrid', 'deps_refine'}
        
        if very_expensive:
            print(f"\n⚠️  WARNING: Testing VERY EXPENSIVE workflows: {', '.join(very_expensive)}")
            print(f"   These use 8+ API calls per task!")
            print(f"   Estimated cost: ${args.num_tasks * 0.20:.2f}")
        elif expensive:
            print(f"\n⚠️  WARNING: Testing expensive workflows: {', '.join(expensive)}")
            print(f"   Estimated cost: ${args.num_tasks * 0.08:.2f}")
    
    # Run comparison
    results = compare_workflows(
        workflows=workflows,
        dataset=args.dataset,
        num_tasks=args.num_tasks,
        llm_model=args.model
    )
    
    print(f"\n✅ Testing complete!")
    print(f"\nNext steps:")
    print(f"1. Review the comparison summary above")
    print(f"2. Choose the best workflow for your needs")
    print(f"3. Use it in your agent for full evaluation")
    print(f"\nExample:")
    print(f"  simulator.set_agent(MyAgentWithBestWorkflow)")
    print(f"  simulator.run_simulation(number_of_tasks=400)")


if __name__ == "__main__":
    main()

