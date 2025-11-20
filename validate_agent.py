import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from frogger_env import FroggerEnv
from frogger_policy import PolicyNet

def validate_policy(checkpoint_path, num_episodes=10000, max_steps=200, use_greedy=True):
    """
    Validate a trained policy over many episodes.
    
    Args:
        checkpoint_path: path to the policy checkpoint
        num_episodes: number of episodes to run for validation
        max_steps: maximum steps per episode
        use_greedy: if True, use greedy action selection (argmax), else sample (NOT stochastic for testing)
        
    Returns:
        dict containing validation metrics
    """
    
    # Load environment and policy
    env = FroggerEnv()
    obs_dim = env._get_obs().shape[0]
    n_actions = 5
    
    policy = PolicyNet(input_dim=obs_dim, hidden_dim=128, n_actions=n_actions)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    policy.load_state_dict(state_dict)
    policy.eval()
    
    # Metrics to track
    episode_returns = []
    episode_lengths = []
    episode_success = []  # 1 if reached goal, 0 otherwise...
    
    print(f"Starting validation over {num_episodes} episodes...")
    print(f"Policy: {checkpoint_path}")
    print(f"Action selection: {'Greedy (argmax)' if use_greedy else 'Stochastic (sampling)'}")
    print("-" * 80)
    
    # Run validation episodes
    for _ep in tqdm(range(num_episodes), desc="Validation"):
        state = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done and step_count < max_steps:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            with torch.no_grad():
                logits = policy(state_tensor)
                
                if use_greedy:
                    action = logits.argmax(dim=-1).item()
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1
        
        # Record metrics
        episode_returns.append(total_reward)
        episode_lengths.append(step_count)
        episode_success.append(1 if total_reward >= 4.0 else 0) # since success is +5 reward, but may take many steps (just safeguard for backwards compat.)
    
    # Compute stats
    returns_array = np.array(episode_returns)
    lengths_array = np.array(episode_lengths)
    success_array = np.array(episode_success)
    
    metrics = {
        "num_episodes": num_episodes,
        "checkpoint": checkpoint_path,
        "greedy": use_greedy,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        
        # Return metrics
        "mean_return": float(returns_array.mean()),
        "std_return": float(returns_array.std()),
        "median_return": float(np.median(returns_array)),
        "min_return": float(returns_array.min()),
        "max_return": float(returns_array.max()),
        
        # Success metrics
        "success_rate": float(success_array.mean()),
        "num_successes": int(success_array.sum()),
        
        # Episode length metrics
        "mean_length": float(lengths_array.mean()),
        "std_length": float(lengths_array.std()),
        "median_length": float(np.median(lengths_array)),
        
        # Confidence ints. (95%)
        "return_95_ci": (
            float(returns_array.mean() - 1.96 * returns_array.std() / np.sqrt(num_episodes)),
            float(returns_array.mean() + 1.96 * returns_array.std() / np.sqrt(num_episodes))
        ),
        "success_95_ci": (
            float(success_array.mean() - 1.96 * success_array.std() / np.sqrt(num_episodes)),
            float(success_array.mean() + 1.96 * success_array.std() / np.sqrt(num_episodes))
        ),
        
        # Raw data
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_success": episode_success,
    }
    
    return metrics

def print_validation_summary(metrics):
    """Print a formatted summary of validation metrics."""
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Policy: {metrics['checkpoint']}")
    print(f"Episodes: {metrics['num_episodes']}")
    print(f"Action Selection: {'Greedy' if metrics['greedy'] else 'Stochastic'}")
    print(f"Timestamp: {metrics['timestamp']}")
    print("-" * 80)
    
    print("\nSUCCESS RATE:")
    print(f"  Overall: {metrics['success_rate']:.4f} ({metrics['success_rate']*100:.2f}%)")
    print(f"  95% CI: [{metrics['success_95_ci'][0]:.4f}, {metrics['success_95_ci'][1]:.4f}]")
    print(f"  Successes: {metrics['num_successes']} / {metrics['num_episodes']}")
    
    print("\nEPISODE RETURN:")
    print(f"  Mean: {metrics['mean_return']:.4f} ± {metrics['std_return']:.4f}")
    print(f"  95% CI: [{metrics['return_95_ci'][0]:.4f}, {metrics['return_95_ci'][1]:.4f}]")
    print(f"  Median: {metrics['median_return']:.4f}")
    print(f"  Range: [{metrics['min_return']:.4f}, {metrics['max_return']:.4f}]")
    
    print("\nEPISODE LENGTH:")
    print(f"  Mean: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    print(f"  Median: {metrics['median_length']:.0f}")
    
    print("=" * 80 + "\n")

def plot_validation_results(metrics, save_dir="evaluation"):
    """Create and save validation plots."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    returns = np.array(metrics['episode_returns'])
    success = np.array(metrics['episode_success'])
    lengths = np.array(metrics['episode_lengths'])
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Validation Results: {os.path.basename(metrics["checkpoint"])}', fontsize=14, fontweight='bold')
    
    # 1. Return distribution
    ax = axes[0, 0]
    ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(metrics['mean_return'], color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["mean_return"]:.2f}')
    ax.axvline(metrics['median_return'], color='orange', linestyle='--', linewidth=2, label=f'Median: {metrics["median_return"]:.2f}')
    ax.set_xlabel('Episode Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Return Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Rolling success rate (window of 1000)
    ax = axes[0, 1]
    window = min(1000, len(success) // 10)
    rolling_success = np.convolve(success, np.ones(window)/window, mode='valid')
    ax.plot(rolling_success, linewidth=1.5)
    ax.axhline(metrics['success_rate'], color='red', linestyle='--', linewidth=2, label=f'Overall: {metrics["success_rate"]:.4f}')
    ax.fill_between(range(len(rolling_success)), 
                     metrics['success_95_ci'][0], metrics['success_95_ci'][1],
                     alpha=0.2, color='red', label='95% CI')
    ax.set_xlabel(f'Episode (rolling window={window})')
    ax.set_ylabel('Success Rate')
    ax.set_title('Rolling Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 3. Episode length distribution
    ax = axes[1, 0]
    ax.hist(lengths, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(metrics['mean_length'], color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["mean_length"]:.1f}')
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Frequency')
    ax.set_title('Episode Length Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Return vs Episode Length scatter
    ax = axes[1, 1]
    successful = returns >= 4.0
    ax.scatter(lengths[~successful], returns[~successful], alpha=0.3, s=10, c='red', label='Failed')
    ax.scatter(lengths[successful], returns[successful], alpha=0.3, s=10, c='green', label='Success')
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Episode Return')
    ax.set_title('Return vs. Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"validation_{os.path.basename(metrics['checkpoint']).replace('.pt', '')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Validation plots saved to: {filepath}")
    
    plt.show()

def save_validation_results(metrics, save_path="evaluation/validation_results.json"):
    """Save validation results to JSON file."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create a copy without raw data for summary file
    summary = {k: v for k, v in metrics.items() 
               if k not in ['episode_returns', 'episode_lengths', 'episode_success']}
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Validation summary saved to: {save_path}")
    
    # Also save full results with raw data
    full_path = save_path.replace('.json', '_full.json')
    with open(full_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Full validation data saved to: {full_path}")

if __name__ == "__main__":
    # Configuration
    CHECKPOINT_PATH = "checkpoints/frogger_policy_0.98.pt"
    NUM_EPISODES = 20000  # 20k episodes for robust statistics
    USE_GREEDY = True     # Use greedy policy for evaluation
    
    # Run validation
    metrics = validate_policy(
        checkpoint_path=CHECKPOINT_PATH,
        num_episodes=NUM_EPISODES,
        use_greedy=USE_GREEDY
    )
    
    # Print summary
    print_validation_summary(metrics)
    
    # Save results
    save_validation_results(metrics)
    
    # Plot results
    plot_validation_results(metrics)

