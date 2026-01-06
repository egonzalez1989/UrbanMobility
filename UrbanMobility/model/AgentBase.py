import math, agentpy as ap, numpy as np
from utils.UrbanUtils import weighted_astar, euclidean, is_behind


#######################################
# Intended as an interface fo agents in the city
#######################################
class UrbanAgent(ap.Agent):
    def setup(self):
        self.agent_type = "Urban"   # Agent type. Override fo specific agent type when needed
        self.city = self.model.city     # Reference to ciy environment
        self.spawn_time = self.model.t      # Birth time of agent
        self.next_action = None     # Next action of agent

    # Position of the agent in the environment grid (integers)
    def grid_position(self):
        return self.city.positions[self]

    ###########################################################
    # Basic agent's behavioral methods
    def sense(self):
        pass

    def react(self):
        pass

    def choose_action(self):
        pass

    def execute(self):
        pass
    ###########################################################



#######################################
# A GENERAL CLASS FOR URBAN MOBILE AGENTS (PEDESTRIANS, CARS)
#######################################
class MobileAgent(UrbanAgent):
    SPEED_LIMIT = 60    # Static maximum possible speed for any agent
    ACTIONS = ['N', 'S', 'E', 'W', 'V']     # Set of actions
    DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]     # Directions mapped from actions. (0,0) included to allow stay when goal is reached
    ACTION_MAP = dict(zip(ACTIONS, DIRECTIONS))        # Maps actions to direction
    DIRECTION_MAP = dict(zip(DIRECTIONS, ACTIONS))      # Maps directions from actions

    '''
    Initializing agent's attributes. 
    '''
    def setup(self, start, goal, direction, speed, visibility, awareness):
        super().setup()
        self.start = (float(start[0]), float(start[1]))     # (float, float): Initial position
        self.goal = (int(goal[0]), int(goal[1]))    # (int, int): Target cell
        self.direction = direction  # (int, int): Integer direction (may change to floating point)
        self.obstacles = set()
        self.speed = self.max_speed = speed         # float: Maximum speed for this agent.
        self.active = True                          # bool: Agents are active by default. They are inactive when collisions or goal reached
        self.removal_time = math.inf                # int: Number of steps before removing from environmnet.
        self.conflicts = []
        self.position = start                        # (float, float): Floating point coordinate
        self.next_positions = []
        self.visibility = visibility                # int: Number of neighbor cells of visibility (semi-accesible environment)
        self.awareness = awareness
        self.future_steps = 10

    # Agent initialization. Override this method to perform customized initializations
    def initialize_agent(self):
        pass

    def grid_position(self, center = None, dir = None):
        center = self.position if center is None else center
        dir = self.direction if dir is None else dir
        rear = (center[0] - (self.width * dir[0]) / 2,
                center[1] - (self.width * dir[1]) / 2)
        return (int(rear[0]), int(rear[1]))

    ###########################################################
    # Sensing and reacting logic. Default behaviour:
    # - Return all agents found in the path and visibility of self
    def sense(self):
        # Get all agents in the visible neighborhood
        neighbors = self.city.neighbors(self, self.visibility)
        #   A conflict is found if trajectories are in risk of collision.
        conflicts = list(filter(lambda x: x, [self.determine_conflict(x) for x in neighbors]))
        # Sort conflicts by distance
        self.conflicts = sorted(conflicts, key=lambda x: euclidean(self.position, x[1].position))
        # Focus on conflicts ahead
        self.conflicts = list(filter(lambda x: not is_behind(self.direction, self.grid_position(), x[1].grid_position()), self.conflicts))
        return self.conflicts

    # Determines if there is a possible collision with other agent. Default ignores any interaction
    def determine_conflict(self, other):
        return False

    def intent_move(self, position, dir):
        current_cell = self.grid_position(position, dir)    # Position within the grid
        hjump, wjump = [dir[i] * self.speed / MobileAgent.SPEED_LIMIT for i in (0, 1)]   # Relative step-jump
        next_position = [position[0] + hjump, position[1] + wjump]    # Next real position into the grid
        # Rear and front coordinates of the car
        rear = (next_position[0] - (self.width * dir[0]) / 2,
                next_position[1] - (self.width * dir[1]) / 2)
        front = (next_position[0] + (self.width * dir[0]) / 2,
                next_position[1] + (self.width * dir[1]) / 2)
        next_cell = self.grid_position(next_position, dir)
        offset = [front[0] - (next_cell[0] + max(0, dir[0]) - self.model.random.uniform(0, .1*self.width) * dir[0]),
                  front[1] - (next_cell[1] + max(0, dir[1]) - self.model.random.uniform(0, .1*self.width) * dir[1])]
        if (int(rear[0]), int(rear[1])) == current_cell:
            next_cell =  current_cell
        if (int(rear[0]), int(rear[1])) == (int(front[0]), int(front[1])):
            offset = [0, 0]


        if self.instructions[next_cell] != self.instructions[current_cell]:
            if dir in [(0, 1), (0, -1)]:
                next_position[1] = next_position[1] - offset[1]
            else:
                next_position[0] = next_position[0] - offset[0]
            next_cell = self.grid_position(next_position, dir)

        return tuple(next_position), tuple(next_cell)

    ################################################################################
    def choose_action(self):
        if self.active:
            self.next_action = self.instructions[self.grid_position()]

    def execute(self):
        # If agent is active, execute action
        if self.active:
            self.move()

        # Otherwise check for removal
        else:
            if self.removal_time == 0:
                self.model.schedule_removal(self)
            elif self.removal_time > 0:
                self.removal_time -= 1
            else:
                raise ValueError("Removal time is negative! Verify implementation")

    # Actual movement according to speed
    def move(self):
        # Choose action according to current route plan
        self.direction = MobileAgent.ACTION_MAP[self.next_action]
        self.position, grid_position = self.intent_move(self.position, self.direction)
        self.city.move_to(self, grid_position)

        # If goal is reached, remove from environment
        if self.grid_position() == self.goal:
            self.deactivate(removal_time=0)
            return

        # Update visibility
        self.set_next_positions()

    ################################################################################
    # Get next cost of moving to a specific cell
    def step_cost(self, position):
        cell_type = self.city.city_grid[position][0]
        return self.cost_map[cell_type]

    def step_risk(self, action, position):
        return 0

    # Deactivating the agent. Is not functional anymore and should be removed.
    def deactivate(self, removal_time = np.inf):
        self.active = False
        self.removal_time = removal_time

    # Default mobile agent reactions for accelerating and slowdown
    def react(self, conflicts = None):
        conflicts = self.conflicts if conflicts is None else conflicts
        if len(conflicts) == 0:
            cur_speed = self.speed
            self.speedup()
            acc = self.speed - cur_speed
            # Check if speedup does nos involve new collisions
            self.set_next_positions()
            new_conflicts = self.sense()
            if new_conflicts:
                if not new_conflicts[0][1].active:
                    self.reroute(lambda x: not x.active)
                else:
                    self.speed = self.speed - acc
                self.set_next_positions()
        else:
            self.slowdown()

    def slowdown(self):
        self.speed = 0

    def speedup(self):
        self.speed = self.max_speed
    #########################################################





class AstarAgent(MobileAgent):
    def setup(self, start, goal, direction, speed, visibility, awareness, weight):
        super().setup(start, goal, direction, speed, visibility, awareness)
        self.weight = weight

    # Get a list of instructions self.instructions to achieve the goal. A small list of next positions will update at each step to simulate visibility
    def find_route(self, grid = None):
        #route = weighted_astar_agent(self, self.grid_position(), self.goal, MobileAgent.INSTRUCTION_MAP, grid=grid, heuristic=manhattan, weight=self.weight)
        self.align_to_cell()
        grid = grid if grid is not None else self.city.city_grid
        route = weighted_astar(self.grid_position(), self.goal, grid=grid, step_cost=self.step_cost, step_risk=self.step_risk, weight=self.weight)
        dir_list = [self.DIRECTION_MAP[(route[i+1][0] - route[i][0], route[i+1][1] - route[i][1])] for i in range(len(route) - 1)]
        self.instructions = dict(zip(route[:-1], dir_list))
        self.instructions[self.goal] = 'V'
        self.set_next_positions()

    def set_next_positions(self):
        # Initializing next near positions (visibility)
        pos = self.position
        self.next_positions, self.next_real = [], []
        if self.grid_position() == self.goal:
            return
        new_grid = self.grid_position()
        for i in range(self.future_steps):
            dir = self.ACTION_MAP[self.instructions[new_grid]]
            new_pos, new_grid = self.intent_move(pos, dir)
            self.next_positions.append(new_grid)
            self.next_real.append(new_pos)
            if euclidean(new_pos, self.grid_position()) > self.visibility or len(set(self.next_positions)) >= self.visibility:
                break
            if new_grid != self.goal:
                pos = new_pos

    # Compute a new route, considering obstacles
    def reroute(self, is_obstacle = lambda x: False):
        # Get blocking agents within the visibility field. Consider function is_obstacle to determine blocking agents
        # and update the city_grid
        neighbors = self.city.neighbors(self, self.visibility)
        blocking_agents = list(filter(is_obstacle, neighbors))
        #self_grid = self.city.city_grid[:, :]
        for blocking_agent in blocking_agents:
            self.obstacles.add(blocking_agent.grid_position())
        self.find_route()
        self.obstacles = set()
#        for blocking_agent in blocking_agents:
#            if blocking_agent.grid_position() in self.obstacles:
#                self.obstacles.remove(blocking_agent.grid_position())

    # Check immediate route positions of agents to verify possible collisions
    def determine_conflict(self, other):
        n = min(len(self.next_positions), len(other.next_positions))
        if self.next_positions and (self.next_positions[0] == other.grid_position()):
            return other.grid_position(), other
        for i in range(1, n-1):
            for j in range(i-1, i+2):
                if euclidean(self.next_real[i], other.next_real[j]) < .7:
                    return other.next_positions[j], other

            #area, j = set([other.grid_position()] if i == 0 else [other.next_positions[i-1]]), i
            #while j < len(other.next_positions) and len(area) < 3:
            #    area.add(other.next_positions[j])
            #    j += 1
            #if self.next_positions[i] in area:
        return False

    def align_to_cell(self):
        dir, pos = self.direction, self.position
        pos0, pos1, cell = pos[0], pos[1], (int(pos[0]), int(pos[1]))
        pad = {'l': cell[0] + 1.1 * self.width/2,
               'r': cell[0] + 1 - 1.1 * self.width/2,
               't': cell[1] + 1.1 * self.width/2,
               'b': cell[1] + 1 - 1.1 * self.width/2,}
        # Verify if point is in addmisible zone
        if pos[0] < pad['l']:
            pos0 = pad['l']
        if pos[0] > pad['r']:
            pos0 = pad['r']
        if pos[1] < pad['t']:
            pos1 = pad['t']
        if pos[1] > pad['b']:
            pos1 = pad['b']

        self.position = (pos0, pos1)