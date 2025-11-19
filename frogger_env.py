import random
import numpy as np
from render_utils import render_game

class FroggerEnv():
    """
    (Simplified) Frogger environment.
    - Cars either move left or right in their respective lanes.
    - Frog can move up, down, left, right, or stay still.
    - Goal is to reach the top row.
    
    Reward Surface:
    - Reward += -0.02 per step penalty to avoid "dithering"
    - Reward += -1.0 for collision (want to avoid death but not discourage exploration)
    - Reward += 5.0 for reaching the goal (want to encourage reaching the goal)
    - Reward += -10.0 for exceeding the maximum number of steps (agent occasionally learns strategy to not move if penalty for exceeding max steps is too low)
    
    Done = True when the frog reaches the goal or exceeds the maximum number of steps.
    """

    def __init__(self, height: int=6, width: int=5, max_steps: int=50):
        self.H = height
        self.W = width
        self.max_steps = 50

        # Initialize lanes in env
        self.lane_rows = list(range(1, self.H - 1)) # [...start... | | | | ...end...]
        self.lane_dirs = [1, -1, 1, -1][:len(self.lane_rows)] # cars flow either left (-1) or right (1)

        # Have 2 cars per lane
        self.cars_per_lane = 2

        self.reset()

    def reset(self):
        # Ensure frog starts at bottom middle
        self.frog_row = self.H - 1
        self.frog_col = self.W // 2

        # Initialize cars to be in random positions in each lane
        self.cars = {row: [] for row in self.lane_rows}
        for row, _d in zip(self.lane_rows, self.lane_dirs):
            positions = random.sample(range(self.W), self.cars_per_lane)
            self.cars[row] = positions

        # Frog / game vars
        self.steps = 0
        self.done = False

        return self._get_obs()

    
    def _get_obs(self):
        """
        Get observation of the environment and turn into flattened input vector.
        - 3 input channels per grid cell: frog, cars, goal
        - frog: 1.0 if frog is in the cell, 0.0 otherwise
        - cars: 1.0 if a car is in the cell, 0.0 otherwise
        - goal: 1.0 if the goal is in the cell, 0.0 otherwise
        """

        grid = np.zeros((3, self.H, self.W), dtype=np.float32)
        
        # Frog (1st channel, get position there)
        grid[0, self.frog_row, self.frog_col] = 1.0
        
        # Cars (2nd channel, get all car positions)
        for row in self.lane_rows:
            for c in self.cars[row]:
                grid[1, row, c] = 1.0
        
        # Goal row (2nd channel, get goal row; i.e., row 0)
        grid[2, 0, :] = 1.0

        return grid.flatten()  # shape is (3 * H * W,)

    def _move_cars(self):
        """
        Move cars in their respective lanes (making them wrap around the edges of the grid).
        """

        new_cars = {}
        for row, d in zip(self.lane_rows, self.lane_dirs):
            new_positions = []
            for c in self.cars[row]:
                nc = (c + d) % self.W  # wrap around by rotating array
                new_positions.append(nc)
            new_cars[row] = new_positions
        self.cars = new_cars

    def step(self, action):
        """
        Take a step in frogger env.
        - action: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT, 4 = STAY
        - reward: reward for the step
        - done: True if episode done, False o/w
        - info: dictionary of additional information (currently unused)
        """

        if self.done:
            raise RuntimeError("Call reset() before step() after done=True.")

        self.steps += 1

        # --- move frog ---
        if action == 0: # UP
            self.frog_row = max(0, self.frog_row - 1) # acct for goal row @ idx=0
        elif action == 1: # DOWN
            self.frog_row = min(self.H - 1, self.frog_row + 1) # acct for starting row @ idx=H-1
        elif action == 2: # LEFT
            self.frog_col = max(0, self.frog_col - 1)
        elif action == 3: # RIGHT
            self.frog_col = min(self.W - 1, self.frog_col + 1)
        elif action == 4: # STAY
            pass

        # --- move cars ---
        self._move_cars()

        # --- compute reward & done ---
        reward = -0.02  # time penalty to avoid "dithering"
        done = False

        # Collision? (frog shares cell with a car)
        if self.frog_row in self.lane_rows:
            if self.frog_col in self.cars[self.frog_row]:
                reward = -1.0
                done = True # dead --> done

        # Reached goal row
        if self.frog_row == 0:
            reward = 5.0
            done = True # won --> done

        # Max steps
        if self.steps >= self.max_steps:
            done = True # exceeded max_steps --> done

            # Apply terminal penalty
            if reward == -0.02:  # i.e., still in "dithering" mode, want to really discourage this
                reward = -10.0 

        self.done = done
        obs = self._get_obs()
        info = {}
        return obs, reward, done, info

    def render(self, use_ascii: bool = True):
        """
        Render the game state.
        Delegates to render_utils.render_game() for consistent rendering across all modes.
        
        Args:
            use_ascii: True for ASCII mode, False for emoji mode
        """
        render_game(self, use_ascii=use_ascii, use_carriage_returns=False)
        print()  # Add blank line after rendering for spacing
