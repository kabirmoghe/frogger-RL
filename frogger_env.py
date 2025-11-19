import random
import numpy as np

class FroggerEnv():
    def __init__(self, height: int=6, width: int=5, max_steps: int=50):
        self.H = height
        self.W = width
        self.max_steps = 50

        # Initialize lanes in env
        self.lane_rows = list(range(1, self.H - 1)) # [...start... | | | | ...end...]
        self.lane_dirs = [1, -1, 1, -1][:len(self.lane_rows)] # cars flow either left (-1) or right (1)

        # Cars per lane = 2
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
        # Create 3 input channels per grid cell: frog, cars, goal
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
        new_cars = {}
        for row, d in zip(self.lane_rows, self.lane_dirs):
            new_positions = []
            for c in self.cars[row]:
                nc = (c + d) % self.W  # wrap around by rotating array
                new_positions.append(nc)
            new_cars[row] = new_positions
        self.cars = new_cars

    def step(self, action):
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
            if reward == -0.02:  # i.e., still in "dithering" mode
                reward = -10.0 

        self.done = done
        obs = self._get_obs()
        info = {}
        return obs, reward, done, info

    def render(self, use_ascii: bool = True):
        """
        Rendering rules:

        If use_ascii is True:
            - Goal row: 'G'
            - Lanes with cars:
                - 'C' where cars are
                - '>' or '<' in empty spaces, depending on lane direction
            - Start row (bottom): '.'
            - Frog: 'F'

        If use_ascii is False (emoji mode):
            - Goal row: '🎯'
            - Lanes with cars:
                - '🚗' where cars are
                - '→' or '←' in empty spaces, depending on lane direction
            - Start row (bottom): '·'
            - Frog: '🐸'
        """
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
        grid = [[empty_symbol for _ in range(self.W)] for _ in range(self.H)]

        # Goal row
        for c in range(self.W):
            grid[0][c] = goal_symbol

        # Lane rows with directional fill
        for row, d in zip(self.lane_rows, self.lane_dirs):
            lane_fill = lane_symbol_right if d == 1 else lane_symbol_left
            for col in range(self.W):
                grid[row][col] = lane_fill

            # Place cars as car_symbol (overwriting arrows)
            for c in self.cars[row]:
                if row % 2 == 0:
                    grid[row][c] = car_symbol_1
                else:
                    grid[row][c] = car_symbol_2

        # Frog (overwrites whatever is underneath)
        grid[self.frog_row][self.frog_col] = frog_symbol

        # Print grid
        for r in range(self.H):
            print(' '.join(grid[r]))
        print()
