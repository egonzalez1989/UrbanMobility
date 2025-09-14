from PlotUtils import *
from UrbanUtils import *
import IPython

#%% md
# Reacting, drivers and pedestrians
#%%
from PlotUtils import *
from UrbanUtils import *
city_grid = gen_city(block_shape = (15, 15), city_shape=(5, 5), street_parking=True, crop = 4)
parameters = {
    'seed': 0,
    'debug': False,
    'city_grid': city_grid,
    'initial_walker_count': 200,
    'initial_driver_count': 100,
    'max_walker_spawn': 1,
    'max_driver_spawn': 1,
    'steps': 50,
    'display': False
}

model = CityModel(parameters)
model.run()
print(model.metrics)


'''
random.seed(0)
block_shape, city_shape = (15, 15), (2, 2)
city_grid = gen_city(block_shape, city_shape, street_parking=True, crop=3)
obstacles, potholes = gen_obstacles(city_grid, 10, 10)
walker_sources = walker_goals = get_walker_endpoints(city_grid)
driver_sources, driver_goals = get_driver_endpoints(city_grid)
parking_spaces = get_parking_spots(city_grid)

parameters = {
    'seed': 1,
    'debug': False,
    'city_grid': city_grid,
    #'grid_graph': grid_graph,
    'obstacles': set(obstacles),
    'potholes': potholes,
    'walker_initial_count': 10,
    'driver_initial_count': 10,
    'driver_sources': driver_sources,
    'driver_goals': driver_goals,
    'walker_sources': walker_sources,
    'walker_goals': walker_goals,
    'parking_spaces': parking_spaces,
    'walker_max_spawn': 1,
    'driver_max_spawn': 1,
    'steps': 300,
    'display': False
}
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
model = CityModel(parameters)
model.run()
'''




'''
def plot_route(city_grid, ax, agents, cmap = []):
    if not cmap:
        cmap = ['green'] * len(agents)

    st, gl, b, o, p, r, s, z, ph  = -1, -2, 1000, 100, 9, 10, 1, 2, 5
    values = dict(zip(['b', 'p', 'r', 't', 's', 'z'], [b, p, r, r, s, z]))
    H, W = city_grid.shape
    grid = np.array([values[city_grid[i, j][0]] for i in range(H) for j in range(W)]).reshape((H, W))
    for obst in model.p.obstacles:
        grid[obst] = o

    for pth in model.p.potholes:
        grid[pth] = ph

    dir_to_vector = {'N': (-1, 0), 'S': (1,0), 'E': (0, 1), 'W': (0, -1)}

    ax.arrow(0, 0, 0, 0, width=0, head_width=0, head_length=0, fc='#ffff9f', ec='#ffff9f', label='z-cross')
    ax.arrow(0, 0, 0, 0, width=0, head_width=0, head_length=0, fc='#a0a0a0', ec='#a0a0a0', label='sidewalk')
    ax.arrow(0, 0, 0, 0, width=0, head_width=0, head_length=0, fc='#636363', ec='#636363', label='road')

    for i in range(len(agents)):
        agent = agents[i]
        ax.plot([agent.start[1]], [agent.start[0]], 'ko', markersize=10)
        ax.plot([agent.goal[1]], [agent.goal[0]], 'k*', markersize=10)
        current = agent.start

        while current != agent.goal:
            (row, col), dir = current, agent.instructions[current]
            dr, dc = dir_to_vector[dir]
            if i > 0:
                ax.arrow(col + (-1)**i * .1, row , dc * .8, dr * .8, width=.1, head_width=0, head_length=0, fc=cmap[i], ec=cmap[i])
            else:
                ax.arrow(col, row, dc * .8, dr * .8, width=.1, head_width=0, head_length=0, fc=cmap[i], ec=cmap[i])
            #ax.arrow(col, row, dc * .8, dr * .8, width=.1, head_width=0, head_length=0, fc=cmap[i], ec=cmap[i])
            current = (row+dr, col+dc)
        if i > 0:
            ax.arrow(col + (-1)**i * .1, row, dc/2, dr/2, width=.1, head_width=.5, head_length=.5, fc=cmap[i], ec=cmap[i])
            ax.axline((-1,0), (0,-1), linewidth=2, linestyle='--', color=cmap[i], label='w = {}'.format(agents[i].weight))
        else:
            ax.arrow(col, row, dc/2, dr/2, width=.1, head_width=.5, head_length=.5, fc=cmap[i], ec=cmap[i])
            ax.axline((-1,0), (0,-1), linewidth=2, linestyle='--', color=cmap[i], label='w = {}'.format(agents[i].weight))

    ax.legend(loc='upper left')
    #plt.axis('off')
    color_dict = {st:'#ed2939', gl: '#028a0f', b: '#BC4A3C', o:'#ff0000', p:'#636363', s: '#a0a0a0', z: '#ffff9f', ph:'#4b371c', r:'#636363'}
    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)
    #ax.set_title('time-step: {}'.format(model.t))

random.seed(0)
city_grid, obstacles, potholes, walker_sources, walker_goals, driver_sources, driver_goals, parking_spaces = gen_city(block_shape = (12, 15), city_shape=(3,3), obstacles = 0, potholes = 0)
parameters = {
    'seed': 0,
    'debug': False,
    'city_grid': city_grid,
    #'grid_graph': grid_graph,
    'obstacles': set(obstacles),
    'potholes': [],
    'walker_initial_count': 0,
    'driver_initial_count': 0,
    'driver_sources': [],
    'driver_goals': [],
    'walker_sources': [],
    'walker_goals': [],
    'parking_spaces': [],
    'walker_max_spawn': 0,
    'driver_max_spawn': 0,
    'steps': 20,
    'display': False
}
model = CityModel(parameters)
model.setup()
drivers = [{'start': (10, 6), 'goal':(12, 16), 'max_speed':60, 'weight': i} for i in (1, 3, 10)]

agents = []
for driver in drivers:
    agent = AstarDriver(model, start = driver['start'], goal = driver['goal'], speed = driver['max_speed'], weight = driver['weight'])
    model.city.add_agents([agent], positions = [driver['start']])
    agent.initialize_agent()
    agents.append(agent)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
plot_route(city_grid, ax, agents, cmap=['#1c6638', '#fd6a02', 'red'])
plt.show()
'''