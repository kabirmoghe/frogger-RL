from frogger_env import FroggerEnv
from frogger_policy import PolicyNet
from render_utils import render_game, clear_screen, print_title
import torch
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt

def compute_returns(rewards, gamma=0.99):
    """
    Compute returns for each step in the episode.
    Using gamma for temporal credit assignment (i.e., reward received t steps ago is worth gamma^t times the reward received today).
    - rewards: list of r_t
    - gamma: discount factor
    - returns: list of G_t
    """

    # rewards: list of r_t
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def train_reinforce(
    num_episodes=2000,
    gamma=0.99,
    beta_range=(0.1, 0.02),
    alpha_b = 0.01,
    lr=1e-3,
    render_every=1000, # how often to watch an episode
    render_slowdown=0.5,  # seconds between steps when watching
    use_ascii=False,
    log_file="training_logs.txt"
):
    """
    Train the policy using the "REINFORCE" algorithm.

    Args:
        num_episodes: number of episodes to train for
        gamma: discount factor
        beta_range: range of beta values for entropy regularization
        lr: learning rate
        render_every: how often to watch an episode (0 to disable)
        render_slowdown: seconds between steps when watching
        use_ascii: whether to use ASCII mode for watching
        log_file: file to write training logs to
    """

    env = FroggerEnv()
    obs_dim = env._get_obs().shape[0]
    n_actions = 5

    policy = PolicyNet(input_dim=obs_dim, hidden_dim=128, n_actions=n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Open log file
    log = open(log_file, 'w')
    log.write(f"Training started: {num_episodes} episodes\n")
    log.write(f"Learning rate: {lr}, Gamma: {gamma}, Beta range: {beta_range}\n")
    log.write("-" * 80 + "\n")
    log.flush()

    # Track measures for evaluation post-training
    episode_returns = []
    episode_lengths = []
    episode_losses = []
    episode_success = []  # 1 if reached goal, else 0

    # Baseline smoothing
    baseline = 0.0

    for episode in range(num_episodes):
        # ---------- 1. Roll out one episode ----------
        state = env.reset()
        log_probs = []
        entropies = []
        rewards = []
        done = False

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # (1, obs_dim)

            logits = policy(state_tensor)  # forward pass (policy(state_tensor))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            entropy = dist.entropy()

            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            entropies.append(entropy)

            next_state, reward, done, _ = env.step(action.item())
            rewards.append(reward)

            state = next_state

        # update baseline
        ep_return = sum(rewards)
        baseline = (1 - alpha_b) * baseline + alpha_b * ep_return

        # ---------- 2. Compute returns ----------
        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize for stability
        normalize=False
        if normalize:
            if len(returns) > 1:
                std = returns.std(unbiased=False)
                if std > 0:
                    returns = (returns - returns.mean()) / (std + 1e-8)
            else:
                # single-step episode: just zero-center
                returns = returns - returns.mean()

        # ---------- 3. Policy gradient loss ----------
        advantages = returns - baseline # use advantages rather than just raw return

        loss = 0
        decay = episode / num_episodes
        beta = beta_range[0] + (beta_range[1] - beta_range[0]) * decay
        for log_prob, A_t, entropy in zip(log_probs, advantages, entropies):

            # negative because we use gradient descent for loss minimization
            loss += -(log_prob * A_t + beta * entropy)

        # ---------- 4. Backprop + update ----------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---------- 5. Logging ----------
        if (episode + 1) % 50 == 0:
            ep_return = sum(rewards)
            ep_len = len(rewards)
            log_msg = f"Episode {episode+1}/{num_episodes} | return = {ep_return:.3f} | loss = {loss.item():.6f} | len = {ep_len}\n"
            log.write(log_msg)
            log.flush()

        # ---------- 6. Optional: watch the agent play ----------
        if render_every > 0 and (episode + 1) % render_every == 0:
            log.write(f"\n=== Watching policy after episode {episode+1} ===\n")
            log.flush()
            watch_episode(policy, env, slowdown=render_slowdown, use_ascii=use_ascii, episode=episode)

        ep_return = sum(rewards)
        ep_len = len(rewards)
        success = 1 if ep_return > 0 else 0  # marking success as pos. return

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        episode_losses.append(loss.item())
        episode_success.append(success)

    # Close log file
    log.write("-" * 80 + "\n")
    log.write(f"Training completed: {num_episodes} episodes\n")
    log.close()
    
    print(f"\nTraining complete. Logs saved to {log_file}")
    print(f"  Final avg return (last 100): {np.mean(episode_returns[-100:]):.3f}")
    print(f"  Success rate (last 100): {np.mean(episode_success[-100:]):.3f}")
    
    plot_results(episode_returns, episode_losses, episode_success)

    # Approximate final success rate for naming checkpoint
    final_success_rate = np.mean(episode_success[-100:])
    checkpoint_name = f"frogger_policy_{final_success_rate:.2f}.pt"

    # Save policy
    torch.save(policy.state_dict(), f"checkpoints/{checkpoint_name}")
    print(f"Policy saved to checkpoints/{checkpoint_name}")

def watch_episode(policy, env, slowdown=0.5, use_ascii=False, max_steps=200, greedy_evaluation=True, episode=None):
    """Watch the policy play one episode with clean rendering"""
    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    while not done and step_count < max_steps:
        # Clear and render current state
        clear_screen()
        print_title()
        print(f"Training Checkpoint @ Ep. {episode+1} - Watching Agent")
        print(f"Step {step_count} | Reward: {total_reward:.3f}")
        print("─" * 35)
        print()
        render_game(env, use_ascii=use_ascii)
        print()
        time.sleep(slowdown)

        # Policy action (greedy)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            if greedy_evaluation:
                logits = policy(state_tensor)
                action = logits.argmax(dim=-1).item()
            else:
                logits = policy(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        step_count += 1

    # Final render after episode ends
    clear_screen()
    print_title()
    print(f"Training Checkpoint @ Ep. {episode+1} - Episode Complete!")
    print("─" * 35)
    print()
    render_game(env, use_ascii=use_ascii)
    print()
    
    if total_reward >= 4.0:
        print("✓ Agent reached the goal!")
    elif total_reward < -0.5:
        print("✗ Agent failed to reach the goal.")
    
    print(f"\nEpisode Reward: {total_reward:.3f}")
    print("\nResuming training in 2 seconds...")
    time.sleep(2.0)

def smooth(x, k=50):
    if len(x) < k:
        return x
    x = np.array(x)
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")

def plot_results(episode_returns, episode_losses, episode_success):
    # Returns
    plt.figure()
    plt.plot(episode_returns, alpha=0.3, label="raw return")
    plt.plot(smooth(episode_returns), label="smoothed return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.title("Episode Return")
    plt.show()

    # Loss
    plt.figure()
    plt.plot(episode_losses, alpha=0.3, label="loss")
    plt.plot(smooth(episode_losses), label="smoothed loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Policy Loss")
    plt.show()

    # Success rate (moving average)
    success_ma = smooth(episode_success, k=200)
    plt.figure()
    plt.plot(success_ma)
    plt.xlabel("Episode")
    plt.ylabel("Success rate (moving avg)")
    plt.title("Success Rate")
    plt.show()

if __name__ == "__main__":
    train_reinforce(
        num_episodes=40000,
        gamma=0.99,
        lr=1e-3,
        render_every=6000,  # watch every 6000 training episodes (0 to disable)
        render_slowdown=0.5,   # seconds between steps when watching
        use_ascii=True, # use ASCII mode for watching
        log_file="training_logs.txt"
    )
