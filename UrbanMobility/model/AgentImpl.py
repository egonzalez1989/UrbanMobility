import math
import numpy as np
from model.AgentBase import AstarAgent, MobileAgent
from utils.UrbanUtils import euclidean, is_right_turn, is_left_turn, manhattan


#######################################
# CLASS FOR CARS
#######################################
class AstarDriver(AstarAgent):
    #Initializing Car agent elements:
    def setup(self, start, goal, direction, speed, visibility = 5, awareness = 1, weight = 1):
        # Dimensions of agent
        self.width = .7
        self.length = .7
        super().setup(start, goal, direction, speed, visibility, awareness, weight)
        self.agent_type = 'driver'
        self.cost_map = self.model.drive_cost
        self.risk_map = self.model.drive_risk
        # Waiting times before reroute
        self.max_wait = 10
        self.wait = 0

    def initialize_agent(self):
        if self.sense():
            # If prbability of collision, initialize with low speed
            self.speed = 10 + self.model.random.random() * 10
            self.set_next_positions()

    def step_cost(self, position):
        if position in self.city.obstacles or position in self.obstacles:
            cost = self.cost_map['o']
        elif position in self.city.potholes:
            cost = self.cost_map['ph']
        else:
            cost = super().step_cost(position)
        return cost

    #Simulates the risk of a step according to allowed movements
    def step_risk(self, action, position):
        cell_code = self.city.city_grid[position]
        # Check type of movement: forward, backward, turning left, right, lane change
        move_type = self.move_type(action, cell_code)
        return self.risk_map[move_type]

    def move_type(self, action, cell_code):
        # Last move for parking.
        if action == 'V':
            result = 'fw'
        # Backward moves
        elif ((action, cell_code[1]) in [('N', 'S'), ('S', 'N')]) or ((action, cell_code[2]) in [('W', 'E'), ('E', 'W')]):
            result = 'bw'
        # Forward move or turns
        elif ((action, cell_code[1]) in [('N', 'N'), ('S', 'S')]) or ((action, cell_code[2]) in [('W', 'W'), ('E', 'E')]):
            new_dir = self.ACTION_MAP[action]
            # Forward movement (maintains direction).
            if self.direction == new_dir and cell_code[0]:
                result = 'fw'
            # If a right turn is performed and it is allowed
            elif is_right_turn(self.direction, new_dir) and cell_code[0] == 't':
                result = 'rt'
            # If a left turn is performed and it is allowed
            elif is_left_turn(self.direction, new_dir) and cell_code[0] == 'l':
                result = 'lt'
            # Otherwise, consider a turn on the direction and not in corners as lane incorporation
            elif ' ' in cell_code:
                result = 'fw'
            else:
                result = 'ft'
        # Any other option is a turn in the wrong direction is considered a lane change
        else:
            result = 'ch'
        return result

    #######################################################################00
    # Actions for reactive decisions
    def slowdown(self, factor = .9, min_speed = 0):
        self.accelerate(min_speed = min_speed, factor = factor)

    def speedup(self, factor = 1.1, max_speed = MobileAgent.SPEED_LIMIT):
        self.accelerate(min_speed = 10, factor = factor, max_speed = self.max_speed)

    def accelerate(self, factor, min_speed = 0, max_speed = MobileAgent.SPEED_LIMIT):
        if self.speed <= 10 and factor < 1 and min_speed == 0:
            self.speed = 0
        speed = self.speed * factor
        self.speed = min(max(speed, min_speed), min(max_speed, self.max_speed))
        self.set_next_positions()

    def move(self):
        super().move()
        if self.grid_position() == self.goal and self.goal in self.city.parking_spaces:
            street_dir = self.city.city_grid[self.grid_position()][1:]
            self.direction == self.ACTION_MAP[street_dir[0] if street_dir[0] != ' ' else street_dir[1]]
            self.removal_time = math.inf

    # A car is parking if inactive and in a parking spot (or without removal scheduled)
    def is_parking(self):
        return (not self.active) and self.removal_time != math.inf

    # Sense for possible collisions. Consider agent's route
    def sense(self):
        if self.goal in self.next_positions:
            agents = self.city.get_agents_at(self.goal)
            if agents:
                agent = agents[0]
                if agent.agent_type == 'driver' and agent.is_parking():
                    self.find_parking()

        # Default conflict sensing
        self.conflicts = super().sense()
        return self.conflicts


    # This method is used when the parking space is occupied by another driver.
    def find_parking(self):
        # Check if there is a nearby parking space (in visibility space)
        x, y = self.grid_position()
        h, w = self.city.city_grid.shape
        dx, dy = self.direction
        if dx == 1:
            near_grid = self.city.city_grid[x: min(h-1, x + self.visibility), y: max(0, y - self.visibility)]
        elif dx == -1:
            near_grid = self.city.city_grid[x: max(0, x - self.visibility), y: min(w-1, y - self.visibility)]
        elif dy == 1:
            near_grid = self.city.city_grid[x: min(h-1, x + self.visibility), y: min(w-1, y - self.visibility)]
        else:
            near_grid = self.city.city_grid[x: max(0, x - self.visibility), y: max(0, y - self.visibility)]
        # If no parking is found, move randomly to another nearby parking cell. A new route will be traced
        near_grid = np.array([near_grid[x, y][0] for x in range(len(near_grid)) for y in range(len(near_grid[0]))]).reshape(near_grid.shape)
        near_spaces = np.argswhere(near_grid == 'p')
        if near_spaces:
            self.goal = near_spaces[0]
        else:
            self.goal = self.random.choice(self.city.parking_spaces)
        self.reroute(lambda x: not x.active)

    def react(self, conflicts = []):
        # If a list of conflicts is received, use it, el pre-stored conflicts
        conflicts = self.conflicts

        if self.speed == 0:
            if self.wait >= self.max_wait:
                self.wait = 0
                self.reroute(is_obstacle=lambda x: not x.active or x.speed == 0 or x in self.city.neighbors(self))
            else:
                self.wait += 1
                # If no conflicts found proceed as default
        else:
            self.wait = 0
        # Verify if conflicts ahead
        if len(conflicts) == 0:
            super().react()
        # Otherwise, check traffic rules and possible actions
        else:
            nearest_conflict, nearest_agent = conflicts[0]
            # React accordingly
            if nearest_agent.agent_type == 'driver':
                self.react_driver(nearest_agent, nearest_conflict)

            # If pedestrian is a conflict
            else:
                # Pedestrian have priority in z-cross
                self.react_walker(nearest_agent, nearest_conflict)
        self.set_next_positions()

    def react_walker(self, walker, location):
        pos = walker.grid_position()
        conflict_dist = euclidean(self.position, pos)
        # Deaccelerate
        if (self.city.city_grid[location][0] == 'z' and euclidean(self.position, location) > 1) or (pos in self.next_positions):
            factor = max(.6, math.sqrt(conflict_dist / self.visibility))
            self.slowdown(factor=factor)
        else:
            self.conflicts.pop(0)
            self.react()

    def react_driver(self, other, location):
        # Conflict distance is the minimum from agents location and possible collision location
        conflict_dist = min(euclidean(self.position, location),
                        euclidean(self.position, other.position))
        factor = max(.6, (conflict_dist / self.visibility) ** 2) + (self.model.random.random() - .5) * .05 # Slightly randomize acceleration increment
        # Emergency break if an agent is near and possible collision
        if other.grid_position() in self.next_positions[:3] or manhattan(other.position, self.position) < 3:# and other.speed <= .6 * self.speed:
            self.slowdown(factor=factor)
        # If intersection, check right of way and avoid collisions with other cars on lane change
        elif (self.city.is_intersection(location) and self.right_of_way(other)) or (not self.city.is_intersection(location) and self.has_lane_change()): # and other.speed > 10)\
            self.conflicts.pop(0)     # If no conflicts with current agent, perfom an inspection with susequent conflicts
            self.react()
        # Otherwise adjust speed accordingly
        else:
            min_speed = 0
            # Adjust speed if I'm behind a car
            if other.direction == self.direction and other.grid_position() in self.next_positions:
                    min_speed = other.speed * .9
            self.accelerate(factor=factor, min_speed=min_speed, max_speed=self.max_speed)

    def has_lane_change(self):
        np, n = self.next_positions, len(self.next_positions)
        next_directions = set([(np[i][0] - np[i+1][0], np[i][1] - np[i+1][1]) for i in range(n-1)])
        if (0, 0) in next_directions:
            next_directions.remove((0, 0))
        return len(next_directions) > 1

    def determine_conflict(self, other):
        result = super().determine_conflict(other)         # If no collisions detected on future interactions, verify if a pedestrian is close
        if not result:
            # Add inactive agents and pedestrians only if they are on the way
            if (not other.active or other.agent_type == 'walker') and other.grid_position() in self.next_positions[:3]:
                result = other.grid_position(), other
        return result

    # Checks if a right, left, or no turn is expected from the agent.
    def turn_type(self):
        turn_side, turn_cell = 'straight', None
        curr_dir = self.direction
        nextpos = self.next_positions
        # Get all next directions to observe a change
        for i in range(len(self.next_positions) - 1):
            next_dir = (nextpos[i+1][0] - nextpos[i][0],  nextpos[i+1][1] - nextpos[i][1])
            # If direction does not change, cotinue
            if next_dir == (0, 0) or next_dir == curr_dir:
                continue
            # else, check possible outcomes
            if curr_dir == (0, 1):      # East
                turn_side, turn_cell = 'right' if next_dir == (1, 0) else 'left', nextpos[i]
                break
            elif curr_dir  == (0, -1):    # West
                turn_side, turn_cell = 'right' if next_dir == (-1, 0) else 'right', nextpos[i]
                break
            elif curr_dir == (1, 0):     # South
                turn_side, turn_cell = 'right' if next_dir == (0, -1) else 'left', nextpos[i]
                break
            elif curr_dir == (-1, 0):    # North
                turn_side, turn_cell = 'right' if next_dir == (0, 1) else 'left', nextpos[i]
                break
        return turn_side, turn_cell

    # Determines if self has right of way over agent other at an intersection.
    def right_of_way(self, other):
        d1, d2 = self.direction, other.direction
        tt1, tc1 = self.turn_type()     # Turn type and turn cell
        tt2, tc2 = other.turn_type()

        # Rule 1: Right agent has priority
        if d1 == (-d2[1], d2[0]):
            return True
#        if d2 == (-d1[1], d1[0]):
#            return False

        # Rule 2: Right turn has priority (uncommon)
        if tt1 == 'right' and tt2 != 'right':
            return True
#        if t2 == 'right' and t1 != 'right':
#            return False

        # Rule 3: Straight has priority over left turn
        if tt1 == 'straight' and tt2 == 'left':
            return True
#        if t2 == 'straight' and t1 == 'left':
#            return False

        # Rule 4: if both agents turn left, nearest has right of way:
        if tt1 == tt2 == 'left' and euclidean(self.grid_position(), tc1) < euclidean(other.grid_position(), tc2):
            return True
        # If both go straight, right hand agent has priority
        #if t1 == 'straight' and t2 == 'straight' and d1 == (-d2[1], d2[0]):
        #    return True

        # Default: no clear priority
        return False

#######################################
# CLASS FOR PEDESTRIAN
#######################################
class AstarWalker(AstarAgent):
    '''
    Initializing agent elements:
    '''
    def setup(self, start, goal, direction, speed, visibility = 3, awareness = 1, weight = 1):
        self.width = .1
        self.length = .1
        super().setup(start, goal, direction, speed, visibility, awareness, weight)
        self.cost_map = self.model.walk_cost
        self.agent_type = 'walker'

    def step_cost(self, position):
        return self.cost_map['o'] if position in self.city.obstacles else super().step_cost(position)

    def determine_conflict(self, other):
        # Float and int positions, and directions of agents
        for i in range(min(len(self.next_positions), len(other.next_positions))):
            if self.next_positions[i] == other.next_positions[i] and other.agent_type == 'driver':
                return self.next_positions[i], other
        return False

    def sense(self):
        conflicts = super().sense()
        self.conflicts = list(filter(lambda x: x[1].agent_type == 'driver', conflicts))
        return conflicts

    def react(self):
        conflicts = self.conflicts
        if conflicts:
            nearest_collision, nearest_agent = conflicts[0]
            t1 = self.city.city_grid[self.grid_position()][0]
            t2 = self.city.city_grid[nearest_collision][0]
            # Pedestrians have preference on zebra cross, continue walking unless an obstacle is recognized.
            if t1 == 'z' or t2 == 'z' and all(x[1].grid_position() not in self.next_positions for x in conflicts):
                pass
            else:
                super().react()
        else:
            super().react()

