"""
Shared rendering utilities for Frogger game display in CLI
"""
import sys
import os

def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_with_carriage_return(text, end='\n'):
    """Print with proper line endings for both normal and raw terminal modes"""
    sys.stdout.write(text + end)
    sys.stdout.flush()


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


def render_game(env, use_ascii=True, use_carriage_returns=False):
    """
    Render the game grid with consistent formatting.
    
    Args:
        env: FroggerEnv instance
        use_ascii: True for ASCII mode, False for emoji mode
        use_carriage_returns: True when in raw terminal mode (for human play)
    """
    # Choose symbols based on mode
    if use_ascii:
        goal_symbol = '$'
        frog_symbol = 'F'
        lane_symbol_right = '»'
        lane_symbol_left = '«'
        car_symbol_1 = 'C'
        car_symbol_2 = 'C'
        empty_symbol = '.'
    else:
        
        goal_symbol = '💰'
        frog_symbol = '🐸'
        lane_symbol_right = '»»'
        lane_symbol_left = '««'
        car_symbol_1 = '🚙'
        car_symbol_2 = '🚘'
        empty_symbol = '· '

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

    # Print grid with appropriate spacing
    separator = ' '
    indent = '\t    '  # tab for centering
    
    for r in range(env.H):
        line = separator.join(grid[r])
        if use_carriage_returns:
            # In raw terminal mode, use print_raw
            print_raw(indent + line)
        else:
            print(indent + line)


def print_title():
    """Print the game title"""
    print("╔═══════════════════════════════════╗")
    print("║         F R O G G E R             ║")
    print("║   (via Reinforcement Learning)    ║")
    print("╚═══════════════════════════════════╝")
    print()
