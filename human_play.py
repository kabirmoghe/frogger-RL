import os
import time
from frogger_env import FroggerEnv

def clear_screen():
    # Works on Unix/macOS; on Windows, you can use 'cls'
    os.system('clear' if os.name != 'nt' else 'cls')


def human_play():
    env = FroggerEnv()
    episode = 0

    while True:
        state = env.reset()
        episode += 1
        done = False

        total_reward = 0.0

        while not done:
            clear_screen()
            print(f"Episode {episode}")
            env.render(use_ascii=False)
            print("Controls: w=up, s=down, a=left, d=right, e=stay, q=quit")
            action_key = input("Your move: ").strip().lower()

            if action_key == 'q':
                print("Quitting.")
                return

            # Map keys to actions
            if action_key == 'w':
                action = 0  # UP
            elif action_key == 's':
                action = 1  # DOWN
            elif action_key == 'a':
                action = 2  # LEFT
            elif action_key == 'd':
                action = 3  # RIGHT
            elif action_key == 'e' or action_key == '':
                action = 4  # STAY
            else:
                # Invalid key: treat as STAY
                action = 4

            _, reward, done, _ = env.step(action)
            total_reward += reward

            clear_screen()
            env.render(use_ascii=False)
            print(f"Last reward: {reward:.2f}   Total reward: {total_reward:.2f}")
            time.sleep(0.2)

            if done:
                if reward > 0:
                    print("You reached the goal! 🎉")
                elif reward < 0:
                    print("You got hit by a car 💀")
                else:
                    print("Episode ended (step limit).")
                print(f"Episode return: {total_reward:.2f}")
                cont = input("Play another episode? (y/n): ").strip().lower()
                if cont != 'y':
                    print("Bye!")
                    return
                break


if __name__ == "__main__":
    human_play()