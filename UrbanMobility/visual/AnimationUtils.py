from matplotlib import pyplot as plt
from model.UrbanModelling import CityModel, PoisonCityModel
from utils.UrbanUtils import *
import agentpy as ap

def animation_plot(model, ax, alpha = '80'):
    plot_city(model.city, ax, alpha = alpha)
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

###################################
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


def plot_agents(city, ax, show_speed = False, show_id = False):
    agents = city.agents
    fig_w, fig_h = ax.figure.get_size_inches()
    h, w = city.city_grid.shape
    marker_dict = {(-1, 0): '^', (1, 0): 'v', (0, 1): '>', (0, -1): '<', (0, 0): 'o'} # Directions of agents
    for agent in agents:
        pos, dir = agent.position, agent.direction
        (color, size) = ('green', 30) if agent.agent_type == 'walker' else ('blue', 60) if agent.active else ('white', 60)
        msize = .9 * size*fig_w / w
        ax.plot(pos[1]-.5, pos[0]-.5, marker_dict[dir], markersize=msize, color=color)     # Agent plot
        if show_speed:
            ax.text(pos[1]-1, pos[0]-.25, str(int(agent.speed)), color="red", fontsize=6)   # Speed label
        if show_id:
            ax.text(pos[1]-1, pos[0], str(agent.id), color="red", fontsize=8)            # Id label

        # Plotting future positions
        n = len(agent.next_real)
        for i in range(n):
            np = agent.next_real[i]
            ax.plot([np[1]-.5], [np[0]-.5], marker='o', color = color, markersize = msize / 6, alpha = (n-i) / n)





###################################
def plot_city(city, ax, alpha='ff'):
    city_grid, obstacles, potholes = city.city_grid, city.obstacles, city.potholes

    h, w = city_grid.shape
    st, gl, b, o, p, r, s, z, ph = -1, -2, 1000, 100, 9, 10, 1, 2, 5
    values = dict(zip(['b', 'p', 'r', 't', 'l', 's', 'z'], [b, p, r, r, r, s, z]))
    grid = np.array([values[city_grid[i, j][0]] for i in range(h) for j in range(w)]).reshape((h, w))

    for obst in obstacles:
        ax.plot([obst[1]], [obst[0]], 'H', markersize=6, markerfacecolor='#636363', markeredgecolor='r',markeredgewidth=1, alpha=int(alpha, 16) / 256.)

    for pth in potholes:
        ax.plot([pth[1]], [pth[0]], 's', markersize=4, markerfacecolor='#795c34', markeredgecolor='#795c34',markeredgewidth=1, alpha=int(alpha, 16)/ 256.)

    parking_cells  = [tuple(x) for x in np.argwhere(np.array([city_grid[i, j][0] == 'p' for i in range(h) for j in range(w)]).reshape((h, w)))]
    poffset = 1 if parking_cells else 0
    for pc in parking_cells:
        ax.text(pc[1]-.25, pc[0]+.25, 'P', color='white', fontsize=10, fontweight='bold')
        #ax.plot([pc[1]], [pc[0]], 's', markersize=4, markerfacecolor='#795c34', markeredgecolor='#795c34',markeredgewidth=1)

    # Plotting arrows
    rt_cells = [tuple(x) for x in np.argwhere(np.array([city_grid[i, j][0] == 't' for i in range(h) for j in range(w)]).reshape((h, w)))]
    for cell in rt_cells:
        if city_grid[cell][1:] == 'NE':
            # Draw right turns
            draw_arrow(ax, city_grid[cell][1:], (cell[0]+2, cell[1]))
            if cell[0] + 2 < h and cell[1] > 0:
                pos = (cell[0] + 2, cell[1] - 1)
                draw_arrow(ax, city_grid[pos][1:], pos)
            #if cell[1] + 2 < w:
            #    pos = (cell[0], cell[1]+2)
            #    draw_arrow(ax, city_grid[pos][1:], pos)

        if city_grid[cell][1:] == 'NW':
            draw_arrow(ax, city_grid[cell][1:], (cell[0], cell[1]+2))
            if  cell[0] + 1 < h and cell[1] + 2 < w:
                pos = (cell[0] + 1, cell[1] + 2)
                draw_arrow(ax, city_grid[pos][1:], pos)
            #if cell[0] > 1:
            #    pos = (cell[0]-2, cell[1])
            #    draw_arrow(ax, city_grid[pos][1:], pos)

        if city_grid[cell][1:] == 'SE':
            draw_arrow(ax, city_grid[cell][1:], (cell[0], cell[1]-2))
            if cell[0] > 0 and cell[1] > 1:
                pos = (cell[0] - 1, cell[1] - 2)
                draw_arrow(ax, city_grid[pos][1:], pos)
            #if cell[0] + 2 < h:
            #    pos = (cell[0] + 2, cell[1])
            #    draw_arrow(ax, city_grid[pos][1:], pos)

        if city_grid[cell][1:] == 'SW':
            draw_arrow(ax, city_grid[cell][1:], (cell[0]-2, cell[1]))
#            ax.arrow(cell[1], cell[0] - ar_offset1, 0, .5, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
#            ax.arrow(cell[1], cell[0] - ar_offset2, -.25, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
            if cell[0] > 1 and cell[1] + 1 < w:
                pos = (cell[0] - 2, cell[1] + 1)
                draw_arrow(ax, city_grid[pos][1:], pos)
#            if cell[1] > 1:
#                pos = (cell[0], cell[1] - 2)
#                draw_arrow(ax, city_grid[pos][1:], pos)


    ax.set_xlim(-.5, w-.5)
    ax.set_ylim(h-.5, -.5)
    ax.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    # Colors: black = edge, white = floor, green = goal, blue = agent
    # Original color dict
    color_dict = {s: '#AAAAAA' + alpha, z: '#FFFFFF'+ alpha, b: '#708238'+ alpha, r: '#333333'+ alpha, p: '#57a0d2'+ alpha}
    #color_dict = {s: '#ffffff'+ alpha, z: '#ffffff'+ alpha, b: '#ffffff'+ alpha, r: '#ffffff'+ alpha, p: '#ffffff' + alpha}

    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)


def draw_arrow(ax, dir, pos):
    if dir == 'NE':
        ax.arrow(pos[1], pos[0] + .35, 0, -.5, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
        ax.arrow(pos[1], pos[0] + .05, .25, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    if dir == 'NW':
        ax.arrow(pos[1] + .35, pos[0], -.5, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
        ax.arrow(pos[1] + .05, pos[0], 0, -.25, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    if dir == 'SE':
        ax.arrow(pos[1] - .35, pos[0], .5, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
        ax.arrow(pos[1] - .05, pos[0], 0, .25, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    if dir == 'SW':
        ax.arrow(pos[1], pos[0] - .35, 0, .5, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
        ax.arrow(pos[1], pos[0] - .05, -.25, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    elif dir == 'N ':
        ax.arrow(pos[1], pos[0] + .35, 0, -.5, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    elif dir == 'S ':
        ax.arrow(pos[1], pos[0] - .35, 0, .5, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    elif dir == ' E':
        ax.arrow(pos[1] - .35, pos[0], .5, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')
    elif dir == ' W':
        ax.arrow(pos[1] + .35, pos[0], -.5, 0, width=.15, head_width=.3, head_length=.25, fc='w', ec='w')



def plot_stamp(model, ax):
    plot_city(model.city, ax, alpha='80')

def plot_route(city, ax, agents, cmap = []):
    if not cmap:
        cmap = ['green'] * len(agents)

    plot_city(city, ax, alpha='aa')
    city_grid = city.city_grid
    h, w = city_grid.shape
    dir_to_vector = {'N': (-1, 0), 'S': (1,0), 'E': (0, 1), 'W': (0, -1)}

    # Start and goal
    ax.plot([agents[0].start[1]], [agents[0].start[0]], 's', markersize=20, markerfacecolor='#EC5353',
            markeredgecolor='r', markeredgewidth=2, zorder=1)
    ax.plot([agents[0].goal[1]], [agents[0].goal[0]], 's', markersize=20, markerfacecolor='#50C878',
            markeredgecolor='g', markeredgewidth=2, zorder=1)

    for i in range(len(agents)):
        agent = agents[i]
        current = agent.start
        ax.plot([agent.start[1]], [agent.start[0]], 'o', ms=10, color=cmap[i])

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
            ax.axline((w,h), (w, 0), linewidth=2, linestyle='--', color=cmap[i], label='w = {}'.format(agents[i].weight))
        else:
            ax.arrow(col, row, dc/2, dr/2, width=.1, head_width=.5, head_length=.5, fc=cmap[i], ec=cmap[i])
            ax.axline((w, h), (w,0), linewidth=2, linestyle='--', color=cmap[i], label='w = {}'.format(agents[i].weight))


    ax.set_xlim(0, w-1)
    ax.set_ylim(h, 0)
    ax.legend(loc='upper left', framealpha=1)
    plt.axis('off')
    #ax.set_title('time-step: {}'.format(model.t))

def plot_next(city_grid, ax, agents, cmap=[]):
    plot_city(city_grid, ax)
    ax.plot([11], [11], 'v', markersize=20, markerfacecolor='y',
         markeredgecolor='r',markeredgewidth=2)

    for i in range(len(agents)):
        agent = agents[i]
        cur_grid = agent.start
        cur_pos = agent.real_position
        if i == 0:
            m = 'v'
        else:
            m = '>'
        ax.plot([cur_pos[1]], [cur_pos[0]], cmap[i] + m, markersize=10)
        for l in range(len(agent.next_positions)):
            dir = agent.ACTION_MAP[agent.instructions[cur_grid]]
            new_pos, new_grid = agent.intent_move(cur_pos, dir)
            ax.plot([new_pos[1]], [new_pos[0]], cmap[i] + m, markersize=5)
            cur_pos, cur_grid = new_pos, new_grid
        ax.plot([cur_pos[1]], [cur_pos[0]], cmap[i] + m, markersize=5)

if __name__ == '__main__':
    seed = 0
    wspawn = lambda n: max(0, np.random.poisson(n))
    dspawn = lambda n: max(0, np.random.poisson(n))
    random.seed(seed)
    city_grid = gen_city(block_shape=(15, 15), city_shape=(5, 5), street_parking=False, two_way=True, crop=0)
    #obstacles, potholes = gen_obstacles(city_grid, obstacles=0, potholes=0)
    obstacles = [(16 + i, 28) for i in random.sample(range(13), 8)]
    obstacles.extend([(31 + i, 28) for i in random.sample(range(13), 7)])
    obstacles.extend([(16 + i, 31) for i in random.sample(range(13), 5)])
    obstacles.extend([(31 + i, 31) for i in random.sample(range(13), 8)])

    parameters = {
        'seed': seed,
        'debug': False,
        'display': False,
        'city_grid': city_grid,
        'initial_walker_count': 0,
        'initial_driver_count': 20,
        'walker_spawn_function': lambda: wspawn(0),
        'driver_spawn_function': lambda: dspawn(20),
        'obstacles': obstacles,
        #    'random_weight_driver': lambda : random.randint(1, 10),
        #    'random_weight_walker': lambda : random.randint(1, 10),
        'steps': 2,
        'display': False
    }
    model = PoisonCityModel(parameters)
    model.setup()
    #model.run()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plot_city(model.city, ax, alpha='88')
    plt.show()