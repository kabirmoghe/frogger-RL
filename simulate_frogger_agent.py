import torch
from frogger_env import FroggerEnv
from frogger_policy import PolicyNet
import time
import torch
import os

def load_policy(checkpoint_path: str):
    env = FroggerEnv()
    obs_dim = env._get_obs().shape[0]
    n_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY (whatever you used)

    policy = PolicyNet(input_dim=obs_dim, hidden_dim=128, n_actions=n_actions)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, env

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def watch_policy(policy, env, episodes=10, greedy=True, slowdown=0.2, use_ascii=False, max_steps=200):
    cumulative_reward = 0.0
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < max_steps:
            clear_screen()                        # 👈 clear previous frame
            print(f"Episode {ep+1} | Cumulative reward: {cumulative_reward:.3f}")
            env.render(use_ascii=use_ascii)       # 👈 draw the grid
            print(f"Step: {step_count}, Return: {total_reward:.3f}")
            time.sleep(slowdown)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
                if greedy:
                    action = logits.argmax(dim=-1).item()
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()

            state, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1

        # update cumulative reward
        cumulative_reward += total_reward

        # final frame
        clear_screen()
        print(f"Episode {ep+1} finished.")
        env.render(use_ascii=use_ascii)
        print(f"Total return = {total_reward:.3f}")
        time.sleep(1.0)

def evaluate_policy(policy, env, n_episodes=200, greedy=True):
    total_return = 0.0
    successes = 0
    lengths = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
                if greedy:
                    action = logits.argmax(dim=-1).item()
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
            state, reward, done, _ = env.step(action)
            ep_ret += reward
            ep_len += 1

        total_return += ep_ret
        lengths.append(ep_len)
        if ep_ret > 0:   # success in your reward scheme
            successes += 1

    print(f"Avg return: {total_return / n_episodes:.3f}")
    print(f"Success rate: {successes / n_episodes:.3f}")
    print(f"Avg length: {sum(lengths) / n_episodes:.2f}")

if __name__ == "__main__":
    policy_path = "checkpoints/frogger_policy_0.89.pt"
    policy, env = load_policy(policy_path)
    evaluate_policy(policy, env, n_episodes=200, greedy=True)
    time.sleep(2.0)
    watch_policy(policy, env, episodes=20, greedy=True, slowdown=0.5, use_ascii=True)