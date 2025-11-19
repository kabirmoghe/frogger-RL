#!/usr/bin/env python3
"""
Minimal, aesthetic Frogger UI
Allows users to play or watch the RL agent with ASCII or emoji rendering
"""
import os
import sys
import time
import torch
import select
from frogger_env import FroggerEnv
from frogger_policy import PolicyNet

# Platform-specific imports for real-time key capture
if os.name == 'nt':  # Windows
    import msvcrt
else:  # Unix/Linux/MacOS
    import tty
    import termios


def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_raw(*args, **kwargs):
    """Print that works properly in raw terminal mode (adds carriage returns)"""
    # In raw mode, we need \r\n instead of just \n for proper line breaks
    text = ' '.join(str(arg) for arg in args)
    end = kwargs.get('end', '\n')
    if os.name != 'nt' and '\n' in (text + end):
        text = text.replace('\n', '\r\n')
        end = end.replace('\n', '\r\n')
    sys.stdout.write(text + end)
    sys.stdout.flush()


def getch():
    """Get a single character from standard input without echo"""
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8', errors='ignore')
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def kbhit():
    """Check if a key has been pressed (non-blocking)"""
    if os.name == 'nt':
        return msvcrt.kbhit()
    else:
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        return dr != []


def getch_nonblocking():
    """Get a character if available, otherwise return None (non-blocking)"""
    if os.name == 'nt':
        if msvcrt.kbhit():
            return msvcrt.getch().decode('utf-8', errors='ignore')
        return None
    else:
        # Terminal should already be in raw mode when this is called
        if kbhit():
            ch = sys.stdin.read(1)
            return ch
        return None


class RawTerminal:
    """Context manager for raw terminal mode on Unix/Linux/MacOS"""
    def __init__(self):
        if os.name != 'nt':
            self.fd = sys.stdin.fileno()
            self.old_settings = None
    
    def __enter__(self):
        if os.name != 'nt':
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        return self
    
    def __exit__(self, type, value, traceback):
        if os.name != 'nt' and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def print_title():
    """Print the game title"""
    print("╔═══════════════════════════════════╗")
    print("║         F R O G G E R             ║")
    print("║     Reinforcement Learning        ║")
    print("╚═══════════════════════════════════╝")
    print()


def show_menu():
    """Display the main menu and return user choices"""
    clear_screen()
    print_title()
    
    # Choose rendering mode
    print("Choose rendering mode:")
    print("  [1] ASCII  (simple characters)")
    print("  [2] Emoji  (🐸🚗🏆)")
    print()
    
    while True:
        choice = input("Select mode (1 or 2): ").strip()
        if choice == '1':
            use_ascii = True
            break
        elif choice == '2':
            use_ascii = False
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Choose play mode
    clear_screen()
    print_title()
    mode_str = "ASCII" if use_ascii else "Emoji"
    print(f"Mode: {mode_str}\n")
    print("Choose game mode:")
    print("  [1] Play yourself")
    print("  [2] Watch RL agent")
    print("  [q] Quit")
    print()
    
    game_speed = None
    while True:
        choice = input("Select mode (1, 2, or q): ").strip().lower()
        if choice == '1':
            mode = 'human'
            # Choose game speed for human play
            clear_screen()
            print_title()
            print(f"Mode: {mode_str}\n")
            print("Choose game speed:")
            print("  [1] Fast   (0.75s per step)")
            print("  [2] Medium (1s per step)")
            print("  [3] Slow   (1.25s per step)")
            print()
            
            while True:
                speed_choice = input("Select speed (1, 2, or 3): ").strip()
                if speed_choice == '1':
                    game_speed = 0.75
                    break
                elif speed_choice == '2':
                    game_speed = 1
                    break
                elif speed_choice == '3':
                    game_speed = 1.25
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            
            return mode, use_ascii, game_speed
        elif choice == '2':
            return 'agent', use_ascii, None
        elif choice == 'q':
            return None, None, None
        else:
            print("Invalid choice. Please enter 1, 2, or q.")


def _render_game(env, use_ascii):
    """Render game using print_raw for proper formatting in raw terminal mode"""
    # Choose symbols based on mode
    if use_ascii:
        goal_symbol = 'G'
        frog_symbol = 'F'
        lane_symbol_right = '>'
        lane_symbol_left = '<'
        car_symbol_1 = 'C'
        car_symbol_2 = 'C'
        empty_symbol = '.'
    else:
        goal_symbol = '🏆'
        frog_symbol = '🐸'
        lane_symbol_right = '→'
        lane_symbol_left = '←'
        car_symbol_1 = '🚙'
        car_symbol_2 = '🚘'
        empty_symbol = '·'

    # Base grid filled with "empty" symbol
    grid = [[empty_symbol for _ in range(env.W)] for _ in range(env.H)]

    # Goal row
    for c in range(env.W):
        grid[0][c] = goal_symbol

    # Lane rows with directional fill
    for row, d in zip(env.lane_rows, env.lane_dirs):
        lane_fill = lane_symbol_right if d == 1 else lane_symbol_left
        for col in range(env.W):
            grid[row][col] = lane_fill

        # Place cars as car_symbol (overwriting arrows)
        for c in env.cars[row]:
            if row % 2 == 0:
                grid[row][c] = car_symbol_1
            else:
                grid[row][c] = car_symbol_2

    # Frog (overwrites whatever is underneath)
    grid[env.frog_row][env.frog_col] = frog_symbol

    # Print grid using print_raw
    for r in range(env.H):
        print_raw(' '.join(grid[r]))


def human_play(use_ascii=True, game_speed=0.3):
    """Human player mode with real-time game updates"""
    env = FroggerEnv()
    episode = 0
    
    # Track the last action for continuous input
    current_action = 4  # Default to STAY
    quit_flag = False
    
    # Use raw terminal mode for the entire play session
    with RawTerminal():
        while True:
            state = env.reset()
            episode += 1
            done = False
            total_reward = 0.0
            step_count = 0
            
            while not done:
                # Display current state
                clear_screen()
                print_raw("╔═══════════════════════════════════╗")
                print_raw("║         F R O G G E R             ║")
                print_raw("║     Reinforcement Learning        ║")
                print_raw("╚═══════════════════════════════════╝")
                print_raw()
                print_raw(f"Episode {episode} | Step {step_count} | Reward: {total_reward:.2f}")
                print_raw("─" * 35)
                print_raw()
                
                # Render the game grid manually to use print_raw
                _render_game(env, use_ascii)
                
                print_raw()
                print_raw("Controls: W=up, S=down, A=left, D=right, Space=stay, Q=quit")
                print_raw(f"Game Speed: {game_speed}s/step")
                print_raw()
                
                # Wait for game_speed duration while checking for input
                start_time = time.time()
                current_action = 4  # Default to STAY
                
                while (time.time() - start_time) < game_speed:
                    # Check for input during the wait period
                    key = getch_nonblocking()
                    
                    if key is not None:
                        key_lower = key.lower()
                        
                        if key_lower == 'q':
                            quit_flag = True
                            break
                        
                        # Map keys to actions (WASD only)
                        action_map = {
                            'w': 0,  # UP
                            's': 1,  # DOWN
                            'a': 2,  # LEFT
                            'd': 3,  # RIGHT
                            ' ': 4,  # STAY
                        }
                        
                        current_action = action_map.get(key_lower, 4)
                    
                    # Small sleep to prevent busy-waiting
                    time.sleep(0.01)
                    
                    if quit_flag:
                        break
                
                if quit_flag:
                    break
                
                # Take action (either user input or STAY)
                _, reward, done, _ = env.step(current_action)
                total_reward += reward
                step_count += 1
            
            if quit_flag:
                return
            
            # Episode ended
            clear_screen()
            print_raw("╔═══════════════════════════════════╗")
            print_raw("║         F R O G G E R             ║")
            print_raw("║     Reinforcement Learning        ║")
            print_raw("╚═══════════════════════════════════╝")
            print_raw()
            print_raw(f"Episode {episode} Complete!")
            print_raw("─" * 35)
            print_raw()
            
            # Render final state
            _render_game(env, use_ascii)
            
            print_raw()
            
            if total_reward >= 4.0:  # Successfully reached goal
                print_raw("🎉 SUCCESS! You reached the goal!")
            elif total_reward < -0.5:  # Hit by car
                print_raw("💥 CRASH! You got hit by a car.")
            else:
                print_raw("⏱️  Time's up!")
            
            print_raw(f"\r\nFinal Reward: {total_reward:.2f}")
            print_raw()
            print_raw("Press any key to play again, or 'q' to quit...")
            
            key = getch_nonblocking()
            while key is None:
                time.sleep(0.01)
                key = getch_nonblocking()
            
            if key.lower() == 'q':
                return


def watch_agent(use_ascii=True, episodes=5):
    """Watch the RL agent play"""
    # Load policy
    checkpoint_path = "checkpoints/frogger_policy.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find policy at {checkpoint_path}")
        print("Press any key to return to menu...")
        getch()
        return
    
    env = FroggerEnv()
    obs_dim = env._get_obs().shape[0]
    n_actions = 5
    
    policy = PolicyNet(input_dim=obs_dim, hidden_dim=128, n_actions=n_actions)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    policy.load_state_dict(state_dict)
    policy.eval()
    
    cumulative_reward = 0.0
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done and step_count < 200:
            clear_screen()
            print_title()
            print(f"Episode {ep+1}/{episodes} | Step {step_count} | Cumulative: {cumulative_reward:.2f}")
            print("─" * 35)
            print()
            env.render(use_ascii=use_ascii)
            print()
            print(f"Episode Reward: {total_reward:.2f}")
            print()
            print("(Press Ctrl+C to stop)")
            
            time.sleep(0.5)  # Slower pace for watching
            
            # Get action from policy
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor)
                action = logits.argmax(dim=-1).item()
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1
        
        cumulative_reward += total_reward
        
        # Final frame
        clear_screen()
        print_title()
        print(f"Episode {ep+1}/{episodes} Complete!")
        print("─" * 35)
        print()
        env.render(use_ascii=use_ascii)
        print()
        
        if total_reward >= 4.0:
            print("✓ Agent reached the goal!")
        elif total_reward < -0.5:
            print("✗ Agent got hit by a car.")
        else:
            print("○ Episode ended (time limit).")
        
        print(f"\nEpisode Reward: {total_reward:.2f}")
        print(f"Cumulative Reward: {cumulative_reward:.2f}")
        print()
        
        if ep < episodes - 1:
            print("Next episode in 2 seconds...")
            time.sleep(2.0)
    
    print()
    print(f"All {episodes} episodes complete!")
    print(f"Average reward: {cumulative_reward/episodes:.2f}")
    print()
    print("Press any key to return to menu...")
    getch()


def main():
    """Main entry point"""
    try:
        while True:
            mode, use_ascii, game_speed = show_menu()
            
            if mode is None:
                clear_screen()
                print("Thanks for playing! 🐸")
                break
            elif mode == 'human':
                human_play(use_ascii=use_ascii, game_speed=game_speed)
            elif mode == 'agent':
                watch_agent(use_ascii=use_ascii, episodes=5)
    
    except KeyboardInterrupt:
        clear_screen()
        print("\nThanks for playing! 🐸")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

