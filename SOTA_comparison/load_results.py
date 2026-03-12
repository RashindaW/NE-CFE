import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def load_and_visualize(results_path):
    """Load saved results and create visualizations."""
    # Create directory for visualizations
    os.makedirs("visualizations", exist_ok=True)

    # Load the PyTorch results file
    results_data = torch.load(results_path, weights_only=False)

    # Extract main components of the results
    dataset = results_data["dataset"]
    methods = results_data["methods"]
    results = results_data["results"]
    summary = results_data["summary"]

    # Print basic information
    print(f"Dataset: {dataset}")
    print(f"Methods evaluated: {methods}")
    print("\nSummary statistics:")
    print("=" * 80)

    # Convert summary to a more readable format
    for method in methods:
        print(f"\nMethod: {method}")
        for metric, value in summary[method].items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    # Create visualizations
    plt.figure(figsize=(10, 6))
    success_rates = [summary[method]['success_rate'] for method in methods]
    plt.bar(methods, success_rates, color=['blue', 'green', 'red'])
    plt.title(f'Success Rates on {dataset}')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    plt.savefig('visualizations/success_rates.png')
    print("\nSaved success rate visualization to visualizations/success_rates.png")

    # Compare feature changes and time
    fig, ax1 = plt.figure(figsize=(10, 6)), plt.subplot(111)
    ax2 = ax1.twinx()

    feature_changes = [summary[method]['avg_feature_changes'] for method in methods]
    times = [summary[method]['avg_time'] for method in methods]

    ax1.bar(methods, feature_changes, color='skyblue', alpha=0.7, label='Feature Changes')
    ax1.set_ylabel('Avg. Feature Changes', color='blue')
    ax1.tick_params(axis='y', colors='blue')

    ax2.plot(methods, times, 'ro-', linewidth=2, label='Time (s)')
    ax2.set_ylabel('Avg. Time (s)', color='red')
    ax2.tick_params(axis='y', colors='red')

    plt.title(f'Feature Changes vs. Time on {dataset}')
    plt.tight_layout()
    plt.savefig('visualizations/features_vs_time.png')
    print("Saved feature changes vs time visualization to visualizations/features_vs_time.png")

    # Save summary as CSV
    summary_df = pd.DataFrame({method: summary[method] for method in methods})
    summary_df.to_csv("summary_results.csv")
    print("\nSummary saved to summary_results.csv")

    # Show detailed results for each method
    print("\nExtracting detailed per-node results...")
    for method in methods:
        print(f"\nDetailed results for {method}:")
        # Convert to dataframe for easier viewing
        df = pd.DataFrame(results[method])

        # Show success rate per node
        print(f"Success rate: {df['success'].mean()*100:.2f}%")

        # Show average metrics for successful nodes
        success_df = df[df['success'] == True]
        if not success_df.empty:
            print(f"Average feature changes: {success_df['feature_changes'].mean():.2f}")
            print(f"Average edge changes: {success_df['total_edge_changes'].mean():.2f}")
            print(f"Average time: {success_df['time_seconds'].mean():.4f} seconds")

        # Save detailed results
        df.to_csv(f"{method}_detailed_results.csv", index=False)
        print(f"Detailed results saved to {method}_detailed_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and visualize counterfactual evaluation results")
    parser.add_argument("results_path", type=str, help="Path to the .pt results file")
    args = parser.parse_args()

    load_and_visualize(args.results_path)
