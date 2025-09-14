import heapq, math
######################################################
# HEURISTICS
######################################################
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1])**2)

######################################################
# A weighted A* implementation considering an urban codification with a grid.
# Cell types have an inherent cost according to the cell type and action.
######################################################
def weighted_astar(agent, start, goal, action_dir, grid, heuristic = manhattan, weight=1):
    # Custom grid. It may be modified to add dynamic environment elemnts, e.g., potholes or damaged cars
    if grid is None:
        grid = agent.city.city_grid
    rows, cols = grid.shape

    # Veirifies if the action is within the square
    def in_bounds(x, y):
        return 0 <= x < rows and 0 <= y < cols

    # Initializing A* algorithm. The heap stores visited current and estimated distance (f, g) and current node
    open_set = []
    heapq.heappush(open_set, (0 + weight * heuristic(start, goal), 0, start))  # (f, g, node)

    came_from = {}
    cost_so_far = {start: (0, agent.direction)}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            # Get instructions from path
            route = [current]
            while current in came_from:
                cfrom = came_from[current]
                route.append(cfrom)
                current = came_from[current]
            return route[::-1]

        for a in action_dir.keys():
            dx, dy = action_dir[a]
            next_node = current[0] + dx, current[1] + dy
            if in_bounds(next_node[0], next_node[1]):
                c, agent.direction = cost_so_far[current]
                new_cost = c + agent.step_cost(next_node) + agent.step_risk(a, current)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node][0]:
                    cost_so_far[next_node] = (new_cost, (dx, dy))
                    priority = new_cost + weight * heuristic(next_node, goal)
                    heapq.heappush(open_set, (priority, new_cost, next_node))
                    came_from[next_node] = current


    return None  # No path found

def is_right_turn(dir1, dir2):
    return dir2 == (dir1[1], -dir1[0])

def is_left_turn(dir1, dir2):
    return is_right_turn(dir2, dir1)
############################################################
# CITY AND BLOCK GENERATION (ENVIRONMENT)
############################################################
import random, numpy as np
"""
Function for the generation of a single city block with the following characteristics: 
- 4 cols and rows for lanes (2 for each direction)
- 2 cols and rows for parking places (if required) 
- 2 cols and rows for sidewalk
- 8 cells for zebra cross
- remaining inner cells are buildings 
:param block_shape: Shape of the required block in the form of tuple (int, int)
:param street_parking: Dictionary allowing parking on the directions with positive value. If boolean all values assume 0 or 1 
:return: A city block of the aforemention characteristics
"""
def gen_block(block_shape = (25, 25), street_parking = False):
    # Create a block using measures of given elements:
    # - 2 sidewalks of each side
    sw = dict(zip(['N', 'S', 'E', 'W'], [1] * 4))
    # - 2 lanes for each direction
    ln = dict(zip(['N', 'S', 'E', 'W'], [2] * 4))
    shape = (block_shape[0], block_shape[1], 3)
    block = np.full(shape, [' ', ' ', ' '])
    # Parking
    val = 1 if street_parking else 0
    sp = dict(zip(['N', 'S', 'E', 'W'], [val] * 4))

    # Borders. It allows to add sidewalks and zebra crosses easily
    # Buildings. Road and sidewalk elements will overwrite accordingly
    block[:, :, 0] = 'b'

    # Roads.
    block[:ln['N']+sw['N'], :, 0] = block[-ln['S'] - 1:, :, 0] = block[:, :ln['E'], 0] = block[:, -ln['W'] - 1:, 0] = 'r'
    # Parking places
    block[(ln['N'], -ln['S']-1), ln['W']+sw['N']+1:-ln['E']-sw['E']-1, 0] =(
        block)[ln['N']+sw['N']+1:-ln['S']-sw['S']-1, (ln['W'], -ln['E']-1),  0] = 'p'
    # Sidewalks
    block[ln['N']+sp['N']:-ln['S']-sp['S'], (ln['W']+sp['W'], -ln['E']-sp['E']-1), 0] = 's'
    block[(ln['N']+sp['N'], -ln['S']-sp['S']-1), ln['E']+sp['E']:-ln['W']-sp['W'], 0] = 's'
    # Zebra crosses
    block[(ln['N']+sp['N'], -ln['S']-sp['S']-1), :ln['W']+sp['W'], 0] = block[(ln['N']+sp['N'], -ln['S']-sp['S']-1), -ln['E']-sp['E']:, 0] = 'z'
    block[:ln['N']+sp['N'], (ln['W']+sp['W'], -ln['E']-sp['E']-1), 0] = block[-ln['S']-sp['S']:, (ln['W']+sp['W'], -ln['E']-sp['E']-1), 0] = 'z'
    # Directions
    block[:ln['N']+sp['N'], :, 2] = 'E'
    block[-ln['S']-sp['S']:, :, 2] = 'W'
    block[:, :ln['W']+sp['W'], 1] = 'N'
    block[:, -ln['E']-sp['E']:, 1] = 'S'
    #Turns
    block[1, 1, 0] = block[1, -2, 0] = block[-2, 1, 0] = block[-2, -2, 0] = 't'
    block[0, 0, 0] = block[0, -1, 0] = block[-1, 0, 0] = block[-1, -1, 0] = 'l'


    block = np.array([[''.join(cell) for cell in row] for row in block])
    return block

"""
Function for the generation of a city of simple blocks.
:param block_shape: Shape of the required block in the form of tuple (int, int)
:param city_shape: Create an array of complete city blocks in the form (int, int)
:param obstacles: Number of obstacles in sidewalks
:param potholes: Number of obstacles in the road
:return: A city encodingtogether with the locations of obstacles, potholes, agent's sources and destinations
"""
def gen_city(block_shape = (25, 25), city_shape = (2, 2), street_parking = False, crop = 0):
    city_block = gen_block(block_shape, street_parking)
    city_grid = np.tile(city_block, city_shape)
    if crop:
        city_grid = city_grid[crop:-crop, crop:-crop]
    return city_grid

def gen_obstacles(city_grid, obstacles = .05, potholes = .05):
    # If obstacles is int, generate that number of obstacles
    city_grid = np.array(city_grid)
    H, W = city_grid.shape
    sidewalk_cells = np.argwhere(city_grid == 's  ')
    road_cells = np.argwhere(np.array([city_grid[i, j][0] in ('r', 't') for i in range(H) for j in range(W)]).reshape((H, W)))
    if type(obstacles) is not int:
        if obstacles >= 0 and obstacles <= 1:
            obstacles = int(obstacles) * len(sidewalk_cells)
        else:
            raise ValueError('obtacles must be integer or float between 0 and 1')

    if type(potholes) is not int:
        if potholes >= 0 and potholes <= 1:
            potholes = int(potholes) * len(road_cells)
        else:
            raise ValueError('potholes must be integer or float between 0 and 1')

    obstacle_cells = set([tuple(random.choice(sidewalk_cells)) for _ in range(obstacles)])
    potholes_cells = set([tuple(random.choice(road_cells)) for _ in range(potholes)])
    return obstacle_cells, potholes_cells


def get_walker_endpoints(city_grid):
    # Pedestrians can be spawned from any sidewalk cell
    H, W = city_grid.shape
    endpoints = [tuple(x) for x in np.argwhere(
        np.array([city_grid[i, j][0] == 's' for i in range(H) for j in range(W)]).reshape((H, W)))]
    return endpoints, endpoints[:]

def get_parking_spots(city_grid):
    H, W = city_grid.shape
    parking_spaces = [tuple(x) for x in np.argwhere(
        np.array([city_grid[i, j][0] == 'p' for i in range(H) for j in range(W)]).reshape((H, W)))]
    return parking_spaces

def get_driver_endpoints(city_grid):
    # Car will be spawned from street intersections on edges. These will also work as goals
    H, W = city_grid.shape
    driver_sources = []
    for j in range(W):
        if 'S' in city_grid[0, j]:
            driver_sources.append((0, j))
        if 'N' in city_grid[H-1, j]:
            driver_sources.append((H-1, j))
    for i in range(H):
        if 'E' in city_grid[i, 0]:
            driver_sources.append((i, 0))
        if 'W' in city_grid[i, W-1]:
            driver_sources.append((i, W-1))

    driver_goals = driver_sources[:]
    driver_goals.extend(get_parking_spots(city_grid))
    return driver_sources, driver_goals

