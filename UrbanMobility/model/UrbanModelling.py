from model.AgentImpl import *
from utils.UrbanUtils import *
from collections import defaultdict
import agentpy as ap, numpy as np

#######################################
# CLASS FOR THE SYSTEM (Ag, Env)
#######################################
class CityModel(ap.Model):

    def setup(self):
        self.p.city_grid = np.array(self.p.city_grid)
        self.city = City(self, shape = self.p.city_grid.shape)

        #Default removal times after incidents, maybe allow modifying from p.
        self.removal_times = {AstarDriver: 20, AstarWalker: 5}
        self.scheduled_removals = []
        # Initilize environment
        self.setup_grid_labels()
        self.metrics = {}
        # Agent history. Stores initialization data of each agent
        self.spawned_agents = {}
        self.repopulate = False if 'repopulate' not in self.p else self.p.repopulate

        #TODO: Riks penalization for agent. Also may change with a p parameter
        #Initializing agents (walkers & drivers)
        self.walker_weight = self.p.walker_weight if 'walker_weight' in self.p else lambda: 1
        self.driver_weight = self.p.driver_weight if 'driver_weight' in self.p else lambda: 1
        self.walker_maxspeed = self.p.walker_maxspeed if 'walker_maxspeed' in self.p else lambda: 10
        self.driver_maxspeed = self.p.driver_maxspeed if 'driver_maxspeed' in self.p else lambda: MobileAgent.SPEED_LIMIT
        self.setup_initial_agents()

        # Heatmap statistics for: Main walking areas and speed
        h, w = self.p.city_grid.shape
        self.walker_speed_hm = np.zeros((h, w))
        self.driver_speed_hm = np.zeros((h, w))
        self.walker_count_hm = [[set() for j in range(w)] for i in range (h)]
        self.driver_count_hm = [[set() for j in range(w)] for i in range (h)]


        # Initializes labels and values for walker and driver agents.
    def setup_grid_labels(self):
        walk_terrain = ['s', 'z', 'p', 'r', 't', 'l', 'b', 'o']  # street, zcross, road, building, obstacle
        self.walk_cost = dict(zip(walk_terrain, [1, 1, 2, 5, 10, 20, 1000, 1000]))
        drive_terrain = ['s', 'z', 'p', 'r', 't', 'l', 'b', 'o', 'ph']
        self.drive_cost = dict(zip(drive_terrain, [1000, 2, 3, 1, 1, 1, 1000, 1001, 20]))
        drive_move = ['fw', 'rt', 'lt', 'ch', 'ft', 'bw']
        self.drive_risk = dict(zip(drive_move, [0, 1, 2, 10, 10, 15]))

    # Initialize agents according to parameter. If a full description for initial agents is given, it adds these agents.
    # Otherwise, a random configuration is created
    def setup_initial_agents(self):
        if 'walkers' not in self.p:
            self.nwalkers = self.p.initial_walker_count if 'initial_walker_count' in self.p else 0
            self.spawn_walkers(self.nwalkers)
        else:
            agents = []
            self.nwalkers = len(self.p.walkers)
            default_vals = {'max_speed' : 10, 'visibility': 3, 'awareness': 1, 'weight': 1,
                            'direction': self.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])}
            for walker in self.p.walkers:
                d, s, v, w, a = [walker[key] if key in walker else default_vals[key] for key in ('direction', 'max_speed', 'visibility', 'weight', 'awareness')]
                start = walker['start']
                position = tuple(start[i] + self.random.uniform(.1, .9) for i in (0, 1))
                agent = AstarWalker(self, position, walker['goal'], direction = d, speed = s, visibility = v, awareness = a, weight = w)
                agents.append(agent)
                self.city.add_agents([agent], positions = [start])
                agent.find_route()
                self.spawned_agents[agent.id] = {'start': agent.start, 'goal': agent.goal, 'type': agent.type,
                                                 'spawn_time': self.t, 'max_speed': agent.max_speed}
            [agent.initialize_agent() for agent in agents]

        if 'drivers' not in self.p:
            self.ndrivers = self.p.initial_driver_count if 'initial_driver_count' in self.p else 0
            self.spawn_drivers(self.ndrivers, sources = self.city.road_cells)
        else:
            agents = []
            self.ndrivers = len(self.p.drivers)
            default_vals = {'max_speed' : 60, 'visibility': 5, 'awareness': 1, 'weight': 1}
            for driver in self.p.drivers:
                s, v, w, a = [driver[key] if key in driver else default_vals[key] for key in ('max_speed', 'visibility', 'weight', 'awareness')]
                start = driver['start']
                if 'direction' not in driver.keys():
                    ways, dict = self.city.city_grid[start][1:].strip(), MobileAgent.ACTION_MAP
                    driver['direction'] = dict[self.random.choice('NSEW')] if ways == '' else dict[self.random.choice(ways)]
                position = tuple(start[i] + self.random.uniform(.4, .6) for i in (0, 1))
                agent = AstarDriver(self, position, driver['goal'], driver['direction'], speed = s, visibility = v, awareness = a, weight = w)
                agents.append(agent)
                self.city.add_agents([agent], positions=[start])
                #positions.append(start), agents.append(agent)
                agent.find_route()
                self.spawned_agents[agent.id] = {'start': agent.start, 'goal': agent.goal, 'type': agent.type,
                                                 'spawn_time': self.t, 'max_speed': agent.max_speed}
            [agent.initialize_agent() for agent in agents]


    # Pedestrians will be spawned from sidewalks, simulating leaving a building
    def spawn_walkers(self, n):
        # List of pedestrians and its positions
        agents, pos = [], []
        for i in range(n):
            # Random start and goal in any position with a sidewalk
            start = list(self.random.choice(self.city.walker_sources))
            position = (start[0] + self.random.uniform(.1, .9), start[1] + self.random.uniform(.1, .9))
            goal = self.random.choice(self.city.walker_goals)
            direction = self.random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            weight = self.walker_weight()
            speed = self.walker_maxspeed()
            agent = AstarWalker(self, position, goal, direction=direction, speed=speed, weight=weight)
            pos.append(start)
            agents.append(agent)
        self.city.add_agents(agents, positions = pos)
        [agent.find_route() for agent in agents]
        [agent.initialize_agent() for agent in agents]
        for agent in agents:
            self.spawned_agents[agent.id] = {'start': agent.start, 'goal': agent.goal, 'type': agent.type,
                                         'spawn_time': self.t, 'max_speed': agent.max_speed}

    # Cars will be spawned from street edges
    def spawn_drivers(self, n, sources = None):
        # List of cars and its positions
        agents, pos, rnd, city = [], [], self.random, self.city
        # I f no sources give, use default given in parameters
        sources = city.driver_sources[:] if sources is None else sources
        sources = list(set(sources).difference(city.positions.values()))
        for i in range(n):
            # Random start and goal in any position with a sidewalk
            start = rnd.choice(sources)
            goal = rnd.choice(city.driver_goals)
            pos.append(start)
            position = (start[0] + .5, start[1] + .5)
            # Avoid very short distances
            while manhattan(goal, start) < 10:
                goal = rnd.choice(city.driver_goals)
            weight = self.driver_weight()
            speed = self.driver_maxspeed()
            ways, dict = city.city_grid[start][1:].strip(), MobileAgent.ACTION_MAP
            direction = dict[rnd.choice('NSEW')] if ways == '' else dict[rnd.choice(ways)]
            agent = AstarDriver(self, position, goal, direction=direction, speed=speed, weight=weight)
            agents.append(agent)
            sources.remove(start)
        city.add_agents(agents, positions=pos)
        [agent.find_route() for agent in agents]
        [agent.initialize_agent() for agent in agents]
        for agent in agents:
            self.spawned_agents[agent.id] = {'start': agent.start, 'goal': agent.goal, 'type': agent.type,
                                         'spawn_time': self.t, 'max_speed': agent.max_speed}


    def update(self):
        # Update and collect metrics at time t
        self.metrics[self.t] = {}

        # Remove any agents scheduled for removal
        if self.scheduled_removals:
            self.city.remove_agents(self.scheduled_removals)
            self.scheduled_removals = []

        # Spawning walkers and drivers to match initial count
        if self.repopulate:
            active_walkers = filter(lambda x: x.agent_type == 'walker' and x.active, self.city.agents)
            self.spawn_walkers(self.nwalkers - len(list(active_walkers)))
            active_drivers = filter(lambda x: x.agent_type == 'driver' and x.active, self.city.agents)
            self.spawn_drivers(self.ndrivers - len(list(active_drivers)))

        # Manages goal reaching agents.
        self.goal_agents()

        # Sense-react-decide cycle.
        self.city.agents.sense()
        self.city.agents.react()
        self.city.agents.choose_action()

        # Monitoring metrics
        self.report_metrics()


    def goal_agents(self):
        arrivals = {}
        for agent in self.city.agents:
            if euclidean(agent.position, agent.goal) < .5:
                arrivals[agent.id] = self.t
                if agent.goal not in self.city.parking_spaces:
                    agent.deactivate(removal_time=0)
                else:
                    agent.deactivate(removal_time=np.inf)
        self.metrics[self.t]['arrivals'] = arrivals


    def step(self):
        # Compute every agent's next action
        self.city.agents.execute()
        # Stop execution when every agent is inactive and removed
        if (len(self.city.agents) == 0 and 'steps' not in self.p.keys()):
            self.end()
            self.stop()


    def end(self):
        self.report('metrics', self.metrics)
        self.report('spawned_agents', self.spawned_agents)




    #####################################################################
    def report_metrics(self):
        # Check for collisions. If any, involved agents remain still and are removed after some steps
        collisions = self.get_collisions()
        self.manage_collisions(collisions)
        # Check for jaywalking
        self.manage_jaywalking()
        self.manage_active_agents()
        self.compute_average_speed()
        self.update_heatmaps()

    def update_heatmaps(self):
        active_walkers = list(filter(lambda x: x.agent_type == 'walker' and x.active, self.city.agents))
        active_drivers = list(filter(lambda x: x.agent_type == 'driver' and x.active, self.city.agents))
        for w in active_walkers:
            pos = self.city.positions[w]
            self.walker_count_hm[pos[0]][pos[1]].add(w.id)
            n = len(self.walker_count_hm[pos[0]][pos[1]])
            self.walker_speed_hm[pos] = ((n-1) * self.walker_speed_hm[pos] + w.speed) / n
        for d in active_drivers:
            pos = self.city.positions[d]
            self.driver_count_hm[pos[0]][pos[1]].add(d.id)
            n = len(self.driver_count_hm[pos[0]][pos[1]])
            self.driver_speed_hm[pos] = ((n-1) * self.driver_speed_hm[pos] + d.speed) / n



# Managing collisions and metrics of agents
    def manage_collisions(self, collisions):
        self.metrics[self.t]['collisions'] = []
        self.metrics[self.t]['runovers'] = []
        for key in collisions.keys():
            agents = collisions[key]
            # Only consider collisions among active agents to avoid duplicates. Agents remain in the environment until removed
            if any(x.active for x in agents):
                if all(x.agent_type == 'driver' for x in agents):
                        self.metrics[self.t]['collisions'].append(key)
                else:
                    self.metrics[self.t]['runovers'].append(key)
            for agent in agents:
                if agent.active:
                    agent.deactivate(self.removal_times[type(agent)])

    def get_collisions(self):
        # Accumulate agents sharing area of influence (collisions).
        collisions = defaultdict(list)
        for agent in self.city.agents:
            for neigh in self.city.neighbors(agent, distance=1):
                if (euclidean(agent.position, neigh.position) < (agent.width + neigh.width) / 2
                        and (agent.agent_type == 'driver' or neigh.agent_type == 'driver')):
                    pos = agent.grid_position()
                    pos = (int(pos[0]), int(pos[1]))
                    collisions[pos].extend([agent, neigh])
        return collisions

    def manage_jaywalking(self):
        jw = defaultdict(list)
        for agent in self.city.agents:
            pos = agent.grid_position()
            if agent.agent_type == 'walker' and pos in (set(self.city.road_cells) - set(self.city.zebras))\
                    and pos not in jw[agent.id]:
                jw[agent.id].append(tuple(pos))
        self.metrics[self.t]['jaywalking'] = jw

    def compute_average_speed(self):
        walkers = list(filter(lambda x: x.agent_type == 'walker', self.city.agents))
        drivers = list(filter(lambda x: x.agent_type == 'driver', self.city.agents))
        self.metrics[self.model.t]['avg_speed_walkers'] = np.mean([x.speed for x in walkers]) if len(walkers) > 0 else 0
        self.metrics[self.model.t]['avg_speed_drivers'] = np.mean([x.speed for x in drivers]) if len(drivers) > 0 else 0

    def manage_active_agents(self):
        walkers = list(filter(lambda x: x.agent_type == 'walker', self.city.agents))
        active_walkers = list(filter(lambda x: x.active, walkers))
        drivers = list(filter(lambda x: x.agent_type == 'driver', self.city.agents))
        active_drivers = list(filter(lambda x: x.active, drivers))
        parked_drivers = list(filter(lambda x: not x.active and x.grid_position() in self.city.parking_cells(), drivers))
        self.metrics[self.model.t]['env_walkers'] = len(walkers)
        self.metrics[self.model.t]['active_walkers'] = len(active_walkers)
        self.metrics[self.model.t]['env_drivers'] = len(drivers)
        self.metrics[self.model.t]['active_drivers'] = len(active_drivers)
        self.metrics[self.model.t]['parked_drivers'] = len(parked_drivers)


    # Agents scheduled for removal will be deleted from city next update
    def schedule_removal(self, agent):
        self.scheduled_removals.append(agent)















####################################################################
def animation_plot(model, ax):
    plot_city(model.city, ax, alpha = '80')
    plot_agents(model.city, ax)
    plot_collisions(model.metrics, ax)
    '''
    for t in range(max(0, model.t - 10), model.t):
        if 'collisions' in model.metrics[t].keys():
            for collision in model.metrics[t]['collisions']:
                ax.text(collision[1] - 1.25, collision[0], "\u2739", fontsize=30, color='red')
        if 'runovers' in model.metrics[t].keys():
            for runover in model.metrics[t]['runovers']:
                ax.text(runover[1] - 1.25, runover[0], "\u2739", fontsize=30, color='red')

    ncollisions = sum([len(model.metrics[t]['collisions']) for t in range(0, model.t)])
    nrunovers = sum([len(model.metrics[t]['runovers']) for t in range(0, model.t)])
    ax.set_title('t: {}, runovers: {}, collisions: {}'.format(model.t, nrunovers, ncollisions))'''

def plot_collisions(metrics, ax):
    T = max(metrics.keys())
    for t in range(max(0, T - 10), T):
        if 'collisions' in metrics[t].keys():
            for collision in metrics[t]['collisions']:
                ax.text(collision[1] - 1.25, collision[0], "\u2739", fontsize=30, color='red')
        if 'runovers' in metrics[t].keys():
            for runover in metrics[t]['runovers']:
                ax.text(runover[1] - 1.25, runover[0], "\u2739", fontsize=30, color='red')

    ncollisions = sum([len(metrics[t]['collisions']) for t in range(0, T)])
    nrunovers = sum([len(metrics[t]['runovers']) for t in range(0, T)])
    ax.set_title('t: {}, runovers: {}, collisions: {}'.format(T, nrunovers, ncollisions))


def plot_agents(city, ax):
    agents = city.agents
    fig_w, fig_h = ax.figure.get_size_inches()
    h, w = city.city_grid.shape
    marker_dict = {(-1, 0): '^', (1, 0): 'v', (0, 1): '>', (0, -1): '<', (0, 0): 'o'} # Directions of agents
    for agent in agents:
        pos, dir = agent.position, agent.direction
        (color, size) = ('black', 30) if agent.agent_type == 'walker' else ('blue', 60) if agent.active else ('white', 60)
        msize = .9 * size*fig_w / w
        ax.plot(pos[1]-.5, pos[0]-.5, marker_dict[dir], markersize=msize, color=color)     # Agent plot

        # Plotting future positions
        n = len(agent.next_real)
        for i in range(n):
            np = agent.next_real[i]
            ax.plot([np[1]-.5], [np[0]-.5], marker='o', color = color, markersize = msize / 6, alpha = (n-i) / n)


def plot_city(city, ax, alpha='ff'):
    city_grid, obstacles, potholes = city.city_grid, city.obstacles, city.potholes

    h, w = city_grid.shape
    st, gl, b, o, p, r, s, z, ph = -1, -2, 1000, 100, 9, 10, 1, 2, 5
    values = dict(zip(['b', 'p', 'r', 't', 'l', 's', 'z'], [b, p, r, r, r, s, z]))
    grid = np.array([values[city_grid[i, j][0]] for i in range(h) for j in range(w)]).reshape((h, w))

    for obst in obstacles:
        ax.plot([obst[1]], [obst[0]], 'H', markersize=6, markerfacecolor='#636363', markeredgecolor='r',
                markeredgewidth=1)

    for pth in potholes:
        ax.plot([pth[1]], [pth[0]], 's', markersize=4, markerfacecolor='#795c34', markeredgecolor='#795c34',
                markeredgewidth=1)

    parking_cells = [tuple(x) for x in np.argwhere(
        np.array([city_grid[i, j][0] == 'p' for i in range(h) for j in range(w)]).reshape((h, w)))]
    poffset = 1 if parking_cells else 0
    for pc in parking_cells:
        ax.text(pc[1] - .25, pc[0] + .25, 'P', color='white', fontsize=10, fontweight='bold')
        # ax.plot([pc[1]], [pc[0]], 's', markersize=4, markerfacecolor='#795c34', markeredgecolor='#795c34',markeredgewidth=1)

    # Plotting turns
    rt_cells = [tuple(x) for x in
                np.argwhere(np.array([city_grid[i, j][0] == 't' for i in range(h) for j in range(w)]).reshape((h, w)))]
    ar_offset1 = 2.5 + poffset
    ar_offset2 = 2.25 + poffset
    for cell in rt_cells:
        if city_grid[cell][1:] == 'NE':
            if cell[0] > 0:
                if city_grid[cell[0] - 1, cell[1]][1:] == 'NE':
                    ax.arrow(cell[1] - 1, cell[0] + ar_offset1, 0, -.5, width=.15, head_width=.3, head_length=.25,
                             fc='white', ec='white')
                else:
                    ax.arrow(cell[1] - 1, cell[0] + ar_offset1 - .75, 0, .5, width=.15, head_width=.3, head_length=.25,
                             fc='white', ec='white')
            ax.arrow(cell[1], cell[0] + ar_offset1, 0, -.5, width=.15, head_width=.3, head_length=.25, fc='white',
                     ec='white')
            ax.arrow(cell[1], cell[0] + ar_offset2, .25, 0, width=.15, head_width=.3, head_length=.25, fc='white',
                     ec='white')
        if city_grid[cell][1:] == 'NW':
            if cell[1] < w - 1:
                if city_grid[cell[0], cell[1] + 1][1:] == 'NW':
                    ax.arrow(cell[1] + ar_offset1, cell[0] + 1, -.5, 0, width=.15, head_width=.3, head_length=.25,
                             fc='white', ec='white')
                else:
                    ax.arrow(cell[1] + ar_offset1 - .75, cell[0] + 1, .5, 0, width=.15, head_width=.3, head_length=.25,
                             fc='white', ec='white')
            ax.arrow(cell[1] + ar_offset1, cell[0], -.5, 0, width=.15, head_width=.3, head_length=.25, fc='white',
                     ec='white')
            ax.arrow(cell[1] + ar_offset2, cell[0], 0, -.25, width=.15, head_width=.3, head_length=.25, fc='white',
                     ec='white')
        if city_grid[cell][1:] == 'SE':
            if cell[1] > 0:
                if city_grid[cell[0], cell[1] - 1][1:] == 'SE':
                    ax.arrow(cell[1] - ar_offset1, cell[0] - 1, .5, 0, width=.15, head_width=.3, head_length=.25,
                             fc='white', ec='white')
                else:
                    ax.arrow(cell[1] - ar_offset1 + .75, cell[0] - 1, -.5, 0, width=.15, head_width=.3, head_length=.25,
                             fc='white', ec='white')
            ax.arrow(cell[1] - ar_offset1, cell[0], .5, 0, width=.15, head_width=.3, head_length=.25, fc='white',
                     ec='white')
            ax.arrow(cell[1] - ar_offset2, cell[0], 0, .25, width=.15, head_width=.3, head_length=.25, fc='white',
                     ec='white')
        if city_grid[cell][1:] == 'SW':
            if cell[0] < h - 1:
                if city_grid[cell[0] + 1, cell[1]][1:] == 'SE':
                    ax.arrow(cell[1] + 1, cell[0] - ar_offset1, 0, .5, width=.15, head_width=.3, head_length=.25,
                             fc='white', ec='white')
                else:
                    ax.arrow(cell[1] + 1, cell[0] - ar_offset1 + .75, 0, -.5, width=.15, head_width=.3, head_length=.25,
                             fc='white', ec='white')
            ax.arrow(cell[1], cell[0] - ar_offset1, 0, .5, width=.15, head_width=.3, head_length=.25, fc='white',
                     ec='white')
            ax.arrow(cell[1], cell[0] - ar_offset2, -.25, 0, width=.15, head_width=.3, head_length=.25, fc='white',
                     ec='white')

    ax.set_xlim(-.5, w - .5)
    ax.set_ylim(h - .5, -.5)
    # Colors: black = edge, white = floor, green = goal, blue = agent
    # color_dict = {s: '#63636380', z: '#ffde2180', b: '#aa4a4480', r:'#82828280', o:'#ff000080', p:'#57a0d280'}
    color_dict = {s: '#AAAAAA' + alpha, z: '#FFFFFF' + alpha, b: '#708238' + alpha, r: '#333333' + alpha,
                  p: '#57a0d2' + alpha}

    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)



####################################################################











#####################################################################

#######################################
# CLASS FOR RANDOM AGENT GENERATION -- FOR SIMULATION PURPOSES
#######################################
class PoisonCityModel(CityModel):
    def update(self):
        # Update and collect metrics at time t
        self.metrics[self.t] = {}

        # Remove any agents scheduled for removal
        if self.scheduled_removals:
            self.city.remove_agents(self.scheduled_removals)
            self.scheduled_removals = []

        # Spawning walkers and drivers per step
        cur_nwalkers = len(list(filter(lambda x: x.agent_type=='walker', self.city.agents)))
        cur_ndrivers = len(list(filter(lambda x: x.agent_type=='driver', self.city.agents)))
        N = self.p.initial_walker_count - cur_nwalkers
        n = 0 if N < 0 else max(0, np.random.poisson(N))
        self.spawn_walkers(n)
        N = self.p.initial_driver_count - cur_ndrivers
        n = 0 if N < 0 else max(0, np.random.poisson(N))
        self.spawn_drivers(n)

        # Manages goal reaching agents.
        self.goal_agents()

        # Sense-react-decide cycle.
        self.city.agents.sense()
        self.city.agents.react()
        self.city.agents.choose_action()

        # Monitoring metrics
        self.report_metrics()

        # Verifies if every driver has stopped and finish simulation
        # if self.metrics[self.t]['avg_speed_drivers'] < 1:
        #     from matplotlib import pyplot as plt
        #     fig = plt.figure(figsize=(10, 10))
        #     ax = fig.add_subplot(111)
        #     animation_plot(self, ax)
        #     plt.show()
        #     self.stop()

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
        self.road_cells = set([tuple(x) for x in np.argwhere(np.array([grid[i, j][0] in ('r', 'z', 'p') for i in range(h) for j in range(w)]).reshape((h, w)))])
        self.zebras = set([tuple(x) for x in np.argwhere(np.array([grid[i, j][0] == 'z' for i in range(h) for j in range(w)]).reshape((h, w)))])

    def is_intersection(self, cell):
        return not ' ' in self.city_grid[cell]

    def get_agents_at(self, position):
        return list(filter(lambda x: x.grid_position == position, self.agents))

    def walker_endpoints(self):
        # Pedestrians can be spawned from any sidewalk cell
        return get_walker_endpoints(self.city_grid)

    def parking_cells(self):
        return get_parking_spots(self.city_grid)

    def driver_endpoints(self):
        # Car will be spawned from street intersections on edges. These will also work as goals
        return get_driver_endpoints(self.city_grid)


