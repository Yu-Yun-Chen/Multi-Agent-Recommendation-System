"""
Test and compare ACCURACY of different recommendation workflows.

This script runs multiple tasks through each workflow and computes actual
accuracy metrics (NDCG@10, Hit Rate@10, Precision@10) to determine which
workflow performs best.

Usage:
    python test_recommendation_accuracy.py                                   # Test default workflows
    python test_recommendation_accuracy.py --num-tasks 20                     # Test on 20 tasks
    python test_recommendation_accuracy.py --workflows default self_refine    # Test specific workflows
    python test_recommendation_accuracy.py --dataset amazon                   # Test on amazon data
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google_gemini_llm import GoogleGeminiLLM
from websocietysimulator import Simulator
from enhanced_recommendation_agent import EnhancedRecommendationAgent

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Workflow mapping
WORKFLOW_MAP = {
    'voyager': 'workflow_with_voyager_planning',
    'self_refine': 'workflow_with_self_refine',
    'cot_sc': 'workflow_with_cot_sc',
    'voyager_memory': 'workflow_with_voyager_memory',
    'openagi': 'workflow_with_openagi_planning',
    'hybrid': 'workflow_hybrid_advanced',
    'tot': 'workflow_with_tot_reasoning',
    'td': 'workflow_with_td_planning',
    'deps': 'workflow_with_deps_planning',
    'all_voyager': 'workflow_all_voyager',
    'dilu_memory': 'workflow_with_dilu_memory',
    'simple': 'workflow_simple_efficient',
    'tot_memory': 'workflow_tot_with_memory',
    'deps_refine': 'workflow_deps_self_refine'
}


def get_actual_metrics(metrics_dict, nested=False):
    """Extract actual metrics from potentially nested structure."""
    if nested and 'metrics' in metrics_dict:
        return metrics_dict['metrics']
    return metrics_dict


def get_metric_names(results):
    """Extract metric names from results structure."""
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    if not successful:
        return [], False
    
    first_result = next(iter(successful.values()))
    metrics_dict = first_result['metrics']
    
    nested = 'metrics' in metrics_dict and isinstance(metrics_dict['metrics'], dict)
    if nested:
        return list(metrics_dict['metrics'].keys()), nested
    return list(metrics_dict.keys()), nested


def get_metric_value(item, metric_name, nested=False):
    """Extract metric value from result item."""
    metrics = item[1]['metrics']
    actual_metrics = get_actual_metrics(metrics, nested)
    val = actual_metrics.get(metric_name, 0)
    return float(val) if isinstance(val, (int, float)) else 0


def create_workflow_agent(workflow_name, llm):
    """Create an agent class that uses a specific workflow."""
    if workflow_name == 'default':
        return EnhancedRecommendationAgent
    
    class WorkflowAgent(EnhancedRecommendationAgent):
        def workflow(self):
            """Execute workflow based on workflow_name."""
            if workflow_name not in WORKFLOW_MAP:
                raise ValueError(f"Unknown workflow: {workflow_name}")
            
            method_name = WORKFLOW_MAP[workflow_name]
            method = getattr(self, method_name)
            return method()
    
    return WorkflowAgent


def run_simulation_and_evaluate(simulator, num_tasks, workflow_name):
    """Run simulation and return metrics or error."""
    start_time = time.time()
    try:
        simulator.run_simulation(number_of_tasks=num_tasks)
        execution_time = time.time() - start_time
        metrics = simulator.evaluate()
        return metrics, execution_time, None
    except Exception as exc:
        execution_time = time.time() - start_time
        return {"error": str(exc)}, execution_time, str(exc)


def test_workflow(workflow_name, dataset='goodreads', num_tasks=10, llm_model='gemini-2.0-flash'):
    """Test a specific workflow and return accuracy metrics."""
    print(f"\n{'='*80}")
    print(f"Testing: {workflow_name.upper()}")
    print(f"{'='*80}")
    
    try:
        llm = GoogleGeminiLLM(model=llm_model)
        simulator = Simulator(data_dir='../data_processed', cache=True)
        simulator.set_task_and_groundtruth(
            task_dir=f'../example/track2/{dataset}/tasks',
            groundtruth_dir=f'../example/track2/{dataset}/groundtruth'
        )
        
        WorkflowAgent = create_workflow_agent(workflow_name, llm)
        simulator.set_agent(WorkflowAgent)
        simulator.set_llm(llm)
        
        print(f"Running {num_tasks} tasks...")
        metrics, execution_time, error = run_simulation_and_evaluate(simulator, num_tasks, workflow_name)
        
        if error:
            print(f"\n✗ Error during workflow '{workflow_name}': {error}")
            return {
                "workflow": workflow_name,
                "metrics": metrics,
                "execution_time": execution_time,
                "avg_time_per_task": execution_time / max(num_tasks, 1),
                "num_tasks": num_tasks,
                "dataset": dataset,
                "success": False,
            }
        
        result = {
            'workflow': workflow_name,
            'metrics': metrics,
            'execution_time': execution_time,
            'avg_time_per_task': execution_time / num_tasks,
            'num_tasks': num_tasks,
            'dataset': dataset,
            'success': True
        }
        
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
        traceback.print_exc()
        return {
            'workflow': workflow_name,
            'error': str(e),
            'success': False
        }


def save_results(results, dataset, num_tasks):
    """Save results to JSON and text files."""
    output_folder = f'workflow_results_{dataset}_{num_tasks}tasks'
    os.makedirs(output_folder, exist_ok=True)
    
    json_file = os.path.join(output_folder, f'workflow_accuracy_results_{dataset}_{num_tasks}tasks.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    txt_file = os.path.join(output_folder, f'workflow_summary_{dataset}_{num_tasks}tasks.txt')
    return json_file, txt_file


def write_header(f, dataset, num_tasks):
    """Write header section to file."""
    f.write("="*80 + "\n")
    f.write(f"WORKFLOW ACCURACY TEST RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {dataset}\n")
    f.write(f"Number of tasks tested: {num_tasks}\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")


def write_results_table(f, results, metric_names, nested):
    """Write results table to file."""
    f.write("="*80 + "\n")
    f.write("RESULTS TABLE\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Workflow':<20} {'Time(s)':<10} ")
    for metric in metric_names:
        f.write(f"{metric:<15} ")
    f.write("\n" + "-"*80 + "\n")
    
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    for workflow_name, result in successful.items():
        metrics = result['metrics']
        time_taken = result['execution_time']
        actual_metrics = get_actual_metrics(metrics, nested)
        
        f.write(f"{workflow_name:<20} {time_taken:>8.1f}s  ")
        for metric in metric_names:
            value = actual_metrics.get(metric, 0)
            if isinstance(value, (int, float)):
                f.write(f"{value:>13.4f}  ")
            else:
                f.write(f"{str(value):>13s}  ")
        f.write("\n")


def write_best_performers(f, results, metric_names, nested):
    """Write best performers section to file."""
    f.write("\n" + "="*80 + "\n")
    f.write("BEST PERFORMER FOR EACH METRIC\n")
    f.write("="*80 + "\n\n")
    
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    for metric in metric_names:
        try:
            best_workflow = max(successful.items(), 
                              key=lambda x: get_metric_value(x, metric, nested))
            metrics = best_workflow[1]['metrics']
            actual_metrics = get_actual_metrics(metrics, nested)
            best_value = actual_metrics.get(metric, 0)
            
            if isinstance(best_value, (int, float)):
                f.write(f"{metric:<25}: {best_workflow[0]:<15} ({best_value:.4f})\n")
            else:
                f.write(f"{metric:<25}: {best_workflow[0]:<15} ({best_value})\n")
        except (ValueError, TypeError):
            f.write(f"{metric:<25}: Unable to compare\n")


def write_recommendations(f, results, metric_names, nested):
    """Write recommendations section to file."""
    f.write("\n" + "="*80 + "\n")
    f.write("RECOMMENDATION\n")
    f.write("="*80 + "\n\n")
    
    primary_metric = 'average_hit_rate' if 'average_hit_rate' in metric_names else metric_names[0]
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    
    try:
        best_overall = max(successful.items(),
                         key=lambda x: get_metric_value(x, primary_metric, nested))
        fastest = min(successful.items(), key=lambda x: x[1]['execution_time'])
        
        metrics = best_overall[1]['metrics']
        actual_metrics = get_actual_metrics(metrics, nested)
        primary_value = actual_metrics.get(primary_metric, 0)
        
        f.write(f"🏆 Best Overall (by {primary_metric}): {best_overall[0].upper()}\n")
        if isinstance(primary_value, (int, float)):
            f.write(f"   {primary_metric}: {primary_value:.4f}\n")
        f.write(f"   Execution time: {best_overall[1]['execution_time']:.1f}s\n\n")
        
        f.write(f"⚡ Fastest Workflow: {fastest[0].upper()}\n")
        f.write(f"   Time: {fastest[1]['execution_time']:.1f}s\n\n")
        
        f.write("💡 Quality/Time Scores (higher is better):\n")
        for name, result in successful.items():
            metrics = result['metrics']
            actual_metrics = get_actual_metrics(metrics, nested)
            metric_val = actual_metrics.get(primary_metric, 0)
            
            if isinstance(metric_val, (int, float)) and result['execution_time'] > 0:
                score = metric_val / (result['execution_time'] / 60)
                f.write(f"   {name:<20}: {score:.4f}\n")
    except (ValueError, TypeError):
        f.write("Unable to determine best overall\n")


def save_readable_summary(results, num_tasks, dataset, filename, raw_failures=None):
    """Save a readable summary to a text file."""
    with open(filename, 'w') as f:
        write_header(f, dataset, num_tasks)
        
        successful = {k: v for k, v in results.items() if v.get('success', False)}
        if not successful:
            f.write("No successful results.\n")
            if raw_failures:
                f.write("\nRAW FAILURE OUTPUTS\n")
                f.write("="*80 + "\n")
                for workflow_name, raw_text in raw_failures.items():
                    f.write(f"{workflow_name}:\n{raw_text}\n\n")
            return
        
        metric_names, nested = get_metric_names(results)
        write_results_table(f, results, metric_names, nested)
        write_best_performers(f, results, metric_names, nested)
        write_recommendations(f, results, metric_names, nested)
        
        f.write("\n" + "="*80 + "\n")
        if raw_failures:
            f.write("RAW FAILURE OUTPUTS\n")
            f.write("="*80 + "\n")
            for workflow_name, raw_text in raw_failures.items():
                f.write(f"{workflow_name}:\n{raw_text}\n\n")


def print_results_table(results, metric_names, nested):
    """Print results table to console."""
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    
    print(f"{'Workflow':<20} {'Time(s)':<10} ", end='')
    for metric in metric_names:
        short_name = metric.replace('_hit_rate', '').replace('_', ' ').title()[:10]
        print(f"{short_name:<12}", end='')
    print()
    print("-" * (32 + len(metric_names) * 12))
    
    for workflow_name, result in successful.items():
        metrics = result['metrics']
        time_taken = result['execution_time']
        actual_metrics = get_actual_metrics(metrics, nested)
        
        print(f"{workflow_name:<20} {time_taken:>8.1f}s  ", end='')
        for metric in metric_names:
            value = actual_metrics.get(metric, 0)
            if isinstance(value, (int, float)):
                print(f"{value:>10.4f}  ", end='')
            else:
                print(f"{str(value):>10s}  ", end='')
        print()


def print_best_performers(results, metric_names, nested):
    """Print best performers to console."""
    print("\n" + "="*80)
    print("BEST PERFORMER FOR EACH METRIC")
    print("="*80)
    
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    for metric in metric_names:
        try:
            best_workflow = max(successful.items(),
                              key=lambda x: get_metric_value(x, metric, nested))
            metrics = best_workflow[1]['metrics']
            actual_metrics = get_actual_metrics(metrics, nested)
            best_value = actual_metrics.get(metric, 0)
            
            if isinstance(best_value, (int, float)):
                print(f"{metric:<20}: {best_workflow[0]:<15} ({best_value:.4f})")
            else:
                print(f"{metric:<20}: {best_workflow[0]:<15} ({best_value})")
        except (ValueError, TypeError):
            print(f"{metric:<20}: Unable to compare (non-numeric values)")


def print_recommendations(results, metric_names, nested):
    """Print recommendations to console."""
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    primary_metric = 'average_hit_rate' if 'average_hit_rate' in metric_names else metric_names[0]
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    
    try:
        best_overall = max(successful.items(),
                         key=lambda x: get_metric_value(x, primary_metric, nested))
        fastest = min(successful.items(), key=lambda x: x[1]['execution_time'])
        
        metrics = best_overall[1]['metrics']
        actual_metrics = get_actual_metrics(metrics, nested)
        primary_value = actual_metrics.get(primary_metric, 0)
        
        print(f"\n🏆 Best Overall: {best_overall[0].upper()}")
        if isinstance(primary_value, (int, float)):
            print(f"   {primary_metric}: {primary_value:.4f}")
        else:
            print(f"   {primary_metric}: {primary_value}")
        print(f"   Execution time: {best_overall[1]['execution_time']:.1f}s")
        
        print(f"\n⚡ Fastest: {fastest[0].upper()}")
        print(f"   Time: {fastest[1]['execution_time']:.1f}s")
        
        fastest_metrics = fastest[1]['metrics']
        fastest_actual = get_actual_metrics(fastest_metrics, nested)
        fastest_metric_val = fastest_actual.get(primary_metric, 0)
        
        if isinstance(fastest_metric_val, (int, float)):
            print(f"   {primary_metric}: {fastest_metric_val:.4f}")
        else:
            print(f"   {primary_metric}: {fastest_metric_val}")
        
        print(f"\n💡 Best Value (quality/time): ", end='')
        value_scores = {}
        for name, result in successful.items():
            metrics = result['metrics']
            actual_metrics = get_actual_metrics(metrics, nested)
            metric_val = actual_metrics.get(primary_metric, 0)
            
            if isinstance(metric_val, (int, float)):
                value_scores[name] = metric_val / (result['execution_time'] / 60)
        
        if value_scores:
            best_value = max(value_scores.items(), key=lambda x: x[1])
            print(f"{best_value[0].upper()}")
            print(f"   Score: {best_value[1]:.4f} (higher is better)")
        else:
            print("Unable to calculate")
    except (ValueError, TypeError):
        print(f"\n⚠️  Could not determine best overall")


def print_comparison_summary(results, num_tasks):
    """Print a comparison table of all workflow results."""
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY ({num_tasks} tasks each)")
    print(f"{'='*80}\n")
    
    successful = {k: v for k, v in results.items() if v.get('success', False)}
    if not successful:
        print("No successful results to compare.")
        return
    
    metric_names, nested = get_metric_names(results)
    print_results_table(results, metric_names, nested)
    print_best_performers(results, metric_names, nested)
    print_recommendations(results, metric_names, nested)


def compare_workflows(workflows, dataset='goodreads', num_tasks=10, llm_model='gemini-2.0-flash'):
    """Compare multiple workflows and determine which performs best."""
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
    
    print_comparison_summary(results, num_tasks)
    
    json_file, txt_file = save_results(results, dataset, num_tasks)
    save_readable_summary(results, num_tasks, dataset, txt_file, raw_failures)
    
    print(f"\n{'='*80}")
    print(f"Results saved to:")
    print(f"  - JSON: {json_file}")
    print(f"  - Summary: {txt_file}")
    print(f"{'='*80}")
    
    return results


def get_all_workflows():
    """Return list of all available workflows."""
    return ['default', 'voyager', 'self_refine', 'cot_sc', 'voyager_memory', 'openagi', 'hybrid',
            'tot', 'td', 'deps', 'all_voyager', 'dilu_memory', 'simple', 'tot_memory', 'deps_refine']


def check_workflow_costs(workflows, num_tasks):
    """Check and warn about expensive workflows."""
    very_expensive = set(workflows) & {'tot', 'tot_memory'}
    expensive = set(workflows) & {'cot_sc', 'hybrid', 'deps_refine'}
    
    if very_expensive:
        print(f"\n⚠️  WARNING: Testing VERY EXPENSIVE workflows: {', '.join(very_expensive)}")
        print(f"   These use 8+ API calls per task!")
        print(f"   Estimated cost: ${num_tasks * 0.20:.2f}")
    elif expensive:
        print(f"\n⚠️  WARNING: Testing expensive workflows: {', '.join(expensive)}")
        print(f"   Estimated cost: ${num_tasks * 0.08:.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test and compare accuracy of different recommendation workflows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_recommendation_accuracy.py
  python test_recommendation_accuracy.py --num-tasks 20
  python test_recommendation_accuracy.py --workflows default self_refine openagi
  python test_recommendation_accuracy.py --dataset amazon --num-tasks 15
  python test_recommendation_accuracy.py --workflows hybrid cot_sc --num-tasks 5

Available workflows:
  default        - Default workflow (StepBack + Generative Memory)
  voyager        - Voyager Planning
  self_refine    - Self-Refine (iterative improvement)
  cot_sc         - COT with Self-Consistency (expensive: 5x cost)
  voyager_memory - Voyager Memory (summarized patterns)
  openagi        - OpenAGI Planning (fast and cheap)
  hybrid         - Hybrid Advanced (expensive: 3x cost, best quality)
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
    
    if 'all' in args.workflows:
        workflows = get_all_workflows()
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
        check_workflow_costs(workflows, args.num_tasks)
    
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
