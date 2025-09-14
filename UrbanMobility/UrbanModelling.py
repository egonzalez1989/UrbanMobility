import AgentBase
from AgentImpl import *
from UrbanUtils import *
import agentpy as ap, numpy as np
from collections import defaultdict

#######################################
# CLASS FOR THE SYSTEM (Ag, Env)
#######################################
class CityModel(ap.Model):
    def setup(self):
        self.city = City(self, shape = self.p.city_grid.shape)
        self.random = self.model.random
        #TODO: Default removal times, maybe allow modifying from p.
        self.removal_times = {AstarDriver: 20, AstarWalker: 5}
        self.scheduled_removals = []
        # Initilize environment
        self.setup_grid_labels()
        self.metrics = {}
        #TODO: Riks penalization for agent. Also may change with a p parameter
        #Initializing agents (walkers & drivers)
        self.max_walker_spawn = self.p.max_walker_spawn if 'max_walker_spawn' in self.p else 0
        self.max_driver_spawn = self.p.max_driver_spawn if 'max_driver_spawn' in self.p else 0
        self.random_weight_walker = self.p.random_weight_walker if 'random_weight_walker' in self.p else lambda: 1
        self.random_speed_walker = self.p.random_weight_walker if 'random_weight_walker' in self.p else lambda: self.random.randint(5, 10)
        self.random_weight_driver = self.p.random_weight_driver if 'random_weight_driver' in self.p else lambda: 1
        self.random_speed_driver = self.p.random_weight_driver if 'random_weight_driver' in self.p else lambda: self.random.randint(MobileAgent.SPEED_LIMIT // 2, MobileAgent.SPEED_LIMIT)
        self.setup_initial_agents()


    # Initializes labels and values for walker and driver agents.
    def setup_grid_labels(self):
        walk_terrain = ['s', 'z', 'p', 'r', 't', 'l', 'b', 'o']  # street, zcross, road, building, obstacle
        self.walk_cost = dict(zip(walk_terrain, [1, 1, 2, 5, 10, 20, 1000, 1000]))
        drive_terrain = ['s', 'z', 'p', 'r', 't', 'l', 'b', 'o', 'ph']
        self.drive_cost = dict(zip(drive_terrain, [1000, 2, 3, 1, 1, 1, 1000, 1001, 20]))
        drive_move = ['fw', 'rt', 'lt', 'ch', 'ft', 'bw']
        self.drive_risk = dict(zip(drive_move, [0, 1, 2, 3, 5, 10]))

    # Initialize agents according to parameter. If a full description for initial agents is given, it adds these agents.
    # Otherwise, a random configuration is created
    def setup_initial_agents(self):
        agents, positions = [], []
        if 'walkers' in self.p:
            default_vals = {'max_speed' : 10, 'visibility': 3, 'awareness': 1, 'weight': 1}
            for walker in self.p.walkers:
                s, v, w, a = [walker[key] if key in walker else default_vals[key] for key in ('max_speed', 'visibility', 'weight', 'awareness')]
                agent = AstarWalker(self, walker['start'], walker['goal'], speed = s, visibility = v, awareness = a, weight = w)
                positions.append(agent.start), agents.append(agent)
        else:
            nwalkers = self.p.initial_driver_count if 'initial_driver_count' in self.p else 0
            self.spawn_walkers(nwalkers)
        if 'drivers' in self.p:
            default_vals = {'max_speed' : 60, 'visibility': 5, 'awareness': 1, 'weight': 1}
            for driver in self.p.drivers:
                s, v, w, a = [driver[key] if key in driver else default_vals[key] for key in ('max_speed', 'visibility', 'weight', 'awareness')]
                agent = AstarDriver(self, driver['start'], driver['goal'], speed = s, visibility = v, awareness = a, weight = w)
                positions.append(agent.start), agents.append(agent)
        else:
            ndrivers = self.p.initial_driver_count if 'initial_driver_count' in self.p else 0
            self.spawn_drivers(ndrivers, sources = self.city.road_cells)
        self.city.add_agents(agents, positions=positions)
        [agent.initialize_agent() for agent in agents]

    # Pedestrians will be spawned from sidewalks, simulating leaving a building
    def spawn_walkers(self, n):
        # List of pedestrians and its positions
        agents, pos = [], []
        for i in range(n):
            # Random start and goal in any position with a sidewalk
            start = self.random.choice(self.city.walker_sources)
            goal = self.random.choice(self.city.walker_goals)
            weight = self.random_weight_walker()
            speed = self.random_speed_walker()#self.random.randint(5, 10)
            agent = AstarWalker(self, start, goal, speed=speed, weight=weight)
            pos.append(start)
            agents.append(agent)
            self.city.add_agents(agents, positions = pos)
        [agent.initialize_agent() for agent in agents]

    # Cars will be spawned from street edges
    def spawn_drivers(self, n, sources = None):
        # List of cars and its positions
        agents, pos = [], []
        # I f no sources give, use default given in parameters
        sources = self.city.driver_sources[:] if sources is None else sources
        sources = list(set(sources).difference(self.city.positions.values()))
        for i in range(n):
            # Random start and goal in any position with a sidewalk
            start = self.random.choice(sources)
            goal = self.random.choice(self.city.driver_goals)
            weight = self.random_weight_driver()
            speed = self.random_speed_driver()  # self.random.randint(5, 10)
            #weight = self.random.randint(1, 10)
            #speed = self.random.randint(MobileAgent.SPEED_LIMIT // 2, MobileAgent.SPEED_LIMIT)
            agent = AstarDriver(self, start, goal, speed=speed, weight=weight)
            pos.append(start)
            agents.append(agent)
            sources.remove(start)
        self.city.add_agents(agents, positions=pos)
        [agent.initialize_agent() for agent in agents]


    def update(self):
        # For collecting metrics at time t
        # Update metrics
        self.metrics[self.t] = {}

        # Remove any agents scheduled for removal
        if self.scheduled_removals:
            self.city.remove_agents(self.scheduled_removals)
            self.scheduled_removals = []


        # Check for collisions. If any, involved agents remain still and are removed after some steps
        collisions = self.get_collisions()
        self.manage_collisions(collisions)

        # Check for jaywalking
        self.jaywalking()

        # Spawning walkers and drivers per step
        n = self.random.randint(0, self.max_walker_spawn)
        self.spawn_walkers(n)
        n = self.random.randint(0, self.max_driver_spawn)
        self.spawn_drivers(n)


        # Manages goal reaching agents.
        self.goal_agents()

        # Sense-react-decide cycle.
        self.city.agents.sense()
        self.city.agents.react()
        self.city.agents.choose_action()

    def goal_agents(self):
        self.metrics[self.t]['arrival_time'] = {}
        for agent in self.city.agents:
            pos = agent.get_position()
            if pos == agent.goal:
                self.metrics[self.t]['arrival_time'][(agent.start, agent.goal)] = (agent.speed, self.t - agent.spawn_time)
                if pos not in self.city.parking_spaces:
                    agent.deactivate(removal_time=0)
                else:
                    agent.deactivate(removal_time=np.inf)

    def jaywalking(self):
        jw = {}
        for agent in self.city.agents:
            pos = agent.get_position()
            if agent.agent_type == 'walker' and pos in self.city.road_cells:
                if pos not in jw.keys():
                    jw[pos] = [agent.id]
                else:
                    jw[pos].append(agent.id)
        self.metrics[self.t]['jaywalking'] = jw

    def step(self):
        # Compute every agent's next action
        self.city.agents.execute()

    def end(self):
        for key in self.log.keys():
            print(key + ': ', self.log[key][-1])
#####################################################################
# Managing collisions between agents
    def manage_collisions(self, collisions):
        self.metrics[self.t]['collisions'] = []
        self.metrics[self.t]['runovers'] = []
        for key in collisions.keys():
            agents = collisions[key]
            if all(map(lambda x: x.agent_type == 'driver', agents)):
                 self.metrics[self.t]['collisions'].append(key)
            else:
                self.metrics[self.t]['runovers'].append(key)
            for agent in agents:
                if agent.active:
                    agent.deactivate(self.removal_times[type(agent)])

    def get_collisions(self):
        # Accumulate agents sharing same position
        collisions = defaultdict(list)
        for key, val in self.city.positions.items():
            collisions[val].append(key)
        # Remove positions with a single agent or only pedestrians (no collision)
        to_remove = []
        for key, val in collisions.items():
            if len(val) == 1 or all([type(x) == AstarWalker for x in val]) or all([not x.active for x in val]):
                to_remove.append(key)

        for key in to_remove:
            collisions.pop(key)

        return collisions

    # Agents scheduled for removal will be deleted from city next update
    def schedule_removal(self, agent):
        self.scheduled_removals.append(agent)
#####################################################################

#######################################
# CLASS FOR ENVIRONMENT
#######################################
'''
A grid of costs is generated for each agent type
'''
class City(ap.Grid):
    def setup(self):
        self.city_grid = self.p.city_grid
        grid, (h, w) = self.p.city_grid, self.p.city_grid.shape
        self.parking_spaces = self.parking_cells() # Parking spaces are encoded in city_grid
        # Default attributes
        self.obstacles = set(self.p.obstacles) if 'obstacles' in self.p else set([])
        self.potholes = set(self.p.potholes) if 'potholes' in self.p else set([])
        self.driver_sources, self.driver_goals = self.p.driver_endpoints if 'driver_endpoints' in self.p else self.driver_endpoints()
        self.walker_sources, self.walker_goals = self.p.walker_endpoints if 'driver_endpoints' in self.p else self.walker_endpoints()
        self.road_cells = [tuple(x) for x in np.argwhere(np.array([grid[i, j][0] in ('r', 'z', 'p') for i in range(h) for j in range(w)]).reshape((h, w)))]

    def is_intersection(self, cell):
        return not ' ' in self.city_grid[cell]

#    def cell_type(self, cell):
#        return self.city_grid[cell]

    def get_agents_at(self, position):
        return list(filter(lambda x: x.get_position == position, self.agents))

    def walker_endpoints(self):
        # Pedestrians can be spawned from any sidewalk cell
        return get_walker_endpoints(self.city_grid)

    def parking_cells(self):
        return get_parking_spots(self.city_grid)

    def driver_endpoints(self):
        # Car will be spawned from street intersections on edges. These will also work as goals
        return get_driver_endpoints(self.city_grid)


