from frogger_env import FroggerEnv
from frogger_policy import PolicyNet
import torch
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt

def compute_returns(rewards, gamma=0.99):
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
    lr=1e-3,
    render_every=200,     # how often to watch an episode
    render_slowdown=0.5,  # seconds between steps when watching
    use_ascii=True       # False = emoji mode if you wired that in
):
    env = FroggerEnv()
    obs_dim = env._get_obs().shape[0]
    n_actions = 5

    policy = PolicyNet(input_dim=obs_dim, hidden_dim=128, n_actions=n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Track measures
    episode_returns = []
    episode_lengths = []
    episode_losses = []
    episode_success = []  # 1 if reached goal, else 0

    # Baseline
    baseline = 0.0
    alpha_b = 0.01  # baseline smoothing

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
        advantages = returns - baseline # use advantages instead of raw return

        loss = 0
        decay = episode / num_episodes
        beta = beta_range[0] + (beta_range[1] - beta_range[0]) * decay
        for log_prob, A_t, entropy in zip(log_probs, advantages, entropies):

            # negative because we use gradient *descent*
            loss += -(log_prob * A_t + beta * entropy)

        # ---------- 4. Backprop + update ----------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---------- 5. Logging ----------
        if (episode + 1) % 50 == 0:
            ep_return = sum(rewards)
            ep_len = len(rewards)
            print(f"Episode {episode+1}/{num_episodes} | return = {ep_return:.3f} | loss = {loss.item():.6f} | len = {ep_len}")

        # ---------- 6. Optional: watch the agent play ----------
        if (episode + 1) % render_every == 0:
            print(f"\n=== Watching policy after episode {episode+1} ===\n")
            # watch_episode(policy, env, slowdown=render_slowdown, use_ascii=use_ascii)

        ep_return = sum(rewards)
        ep_len = len(rewards)
        success = 1 if ep_return > 0 else 0  # crude success flag (since goal≈+5-steps)

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        episode_losses.append(loss.item())
        episode_success.append(success)

    plot_results(episode_returns, episode_losses, episode_success)

    # Save the policy
    torch.save(policy.state_dict(), "checkpoints/frogger_policy.pt")

def watch_episode(policy, env, slowdown=0.2, use_ascii=False, max_steps=200, greedy_evaluation=True):
    state = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    while not done and step_count < max_steps:
        # Render current state
        env.render(use_ascii=use_ascii)
        print(f"Step: {step_count}, Total reward so far: {total_reward:.3f}")
        time.sleep(slowdown)

        # Policy action
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    with torch.no_grad():
        if greedy_evaluation:
            logits = policy(state_tensor)
            action = logits.argmax(dim=-1).item()   # now action is a Python int
        else:
            logits = policy(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()           # also convert to int here

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        step_count += 1

    # Final render after episode ends
    env.render(use_ascii=use_ascii)
    print(f"Episode finished. Total reward = {total_reward:.3f}\n")
    time.sleep(1.0)

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
        num_episodes=30000,
        gamma=0.99,
        lr=1e-3,
        render_every=200,      # watch every 200 training episodes
        render_slowdown=0.15,  # slower / faster animation
        use_ascii=False        # emoji mode if you added it
    )

