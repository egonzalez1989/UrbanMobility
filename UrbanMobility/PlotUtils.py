from matplotlib import pyplot as plt
from UrbanModelling import CityModel
from UrbanUtils import *
import agentpy as ap

def animation_plot(model, ax):
    fig_w, fig_h = ax.figure.get_size_inches()
    city_grid = model.city.city_grid
    plot_city(city_grid, ax, model.city.obstacles, model.city.potholes, alpha = '80')
    h, w = city_grid.shape

    marker_dict = {(-1, 0): '^', (1, 0): 'v', (0, 1): '>', (0, -1): '<', (0, 0): 'o'} # Directions of agents
    for agent in model.city.agents:
        pos, dir = agent.real_position, agent.direction
        (color, size) = ('white', 20) if agent.agent_type == 'walker' else ('black', 40) if agent.active else ('red', 40)
        msize = size*fig_w / w
        ax.plot(pos[1], pos[0], marker_dict[dir], markersize=msize, color=color)
        ax.text(pos[1]+.5, pos[0], str(int(agent.speed)), color="red", fontsize=6)
        ax.text(pos[1]-1, pos[0], str(agent.id), color="red", fontsize=8)

        # Plotting future positions
        n = len(agent.next_real)
        for i in range(n):
            np = agent.next_real[i]
            ax.plot([np[1]], [np[0]], marker='o', color = color, markersize = msize / 6, alpha = (n-i) / n)
        #if agent.next_positions and agent.agent_type == 'driver':
        #    last = agent.next_positions[-1]
        #    ax.arrow(pos[1], pos[0], last[1]-pos[1], last[0]-pos[0], width=.01, head_width=.5,
        #         head_length=.5, fc=color, ec=color, linestyle='--')

    for t in range(max(0, model.t - 10), model.t):
        if t in model.collisions.keys():
            for collision in model.collisions[t]:
                ax.text(collision[1] - .75, collision[0] + .5, "\u2739", fontsize=30, color='red')
        if t in model.runovers.keys():
            for runover in model.runovers[t]:
                ax.text(runover[1] - .75, runover[0] + .5, "\u2739", fontsize=30, color='red')

    ax.set_title('t: {}, runovers: {}, collisions: {}'.format(model.t, len(model.runovers), len(model.collisions)))



###################################
def plot_city(city_grid, ax, obstacles = [], potholes=[], alpha = 'ff'):
    h, w = city_grid.shape
    st, gl, b, o, p, r, s, z, ph = -1, -2, 1000, 100, 9, 10, 1, 2, 5
    values = dict(zip(['b', 'p', 'r', 't', 'l', 's', 'z'], [b, p, r, r, r, s, z]))
    grid = np.array([values[city_grid[i, j][0]] for i in range(h) for j in range(w)]).reshape((h, w))

    for obst in obstacles:
        ax.plot([obst[1]], [obst[0]], 'H', markersize=6, markerfacecolor='#636363', markeredgecolor='r',markeredgewidth=1)

    for pth in potholes:
        ax.plot([pth[1]], [pth[0]], 's', markersize=4, markerfacecolor='#795c34', markeredgecolor='#795c34',markeredgewidth=1)

    parking_cells  = [tuple(x) for x in np.argwhere(np.array([city_grid[i, j][0] == 'p' for i in range(h) for j in range(w)]).reshape((h, w)))]
    poffset = 1 if parking_cells else 0
    for pc in parking_cells:
        ax.text(pc[1]-.25, pc[0]+.25, 'P', color='white', fontsize=10, fontweight='bold')
        #ax.plot([pc[1]], [pc[0]], 's', markersize=4, markerfacecolor='#795c34', markeredgecolor='#795c34',markeredgewidth=1)

    # Plotting turns
    rt_cells = [tuple(x) for x in np.argwhere(np.array([city_grid[i, j][0] == 't' for i in range(h) for j in range(w)]).reshape((h, w)))]
    ar_offset1 = 2.5 + poffset
    ar_offset2 = 2.25 + poffset
    for cell in rt_cells:
        if city_grid[cell][1:] == 'NE':
            ax.arrow(cell[1] - 1, cell[0] + ar_offset1, 0, -.5, width=.15, head_width=.3, head_length=.25, fc='white',ec='white')
            ax.arrow(cell[1], cell[0] + ar_offset1, 0, -.5, width=.15, head_width=.3, head_length=.25, fc='white', ec='white')
            ax.arrow(cell[1], cell[0] + ar_offset2, .25, 0, width=.15, head_width=.3, head_length=.25, fc='white', ec='white')
        if city_grid[cell][1:] == 'NW':
            ax.arrow(cell[1] + ar_offset1, cell[0] + 1, -.5, 0, width=.15, head_width=.3, head_length=.25, fc='white',ec='white')
            ax.arrow(cell[1] + ar_offset1, cell[0], -.5, 0, width=.15, head_width=.3, head_length=.25, fc='white', ec='white')
            ax.arrow(cell[1] + ar_offset2, cell[0], 0, -.25, width=.15, head_width=.3, head_length=.25, fc='white', ec='white')
        if city_grid[cell][1:] == 'SE':
            ax.arrow(cell[1] - ar_offset1, cell[0] - 1, .5, 0, width=.15, head_width=.3, head_length=.25, fc='white',ec='white')
            ax.arrow(cell[1] - ar_offset1, cell[0], .5, 0, width=.15, head_width=.3, head_length=.25, fc='white', ec='white')
            ax.arrow(cell[1] - ar_offset2, cell[0], 0, .25, width=.15, head_width=.3, head_length=.25, fc='white', ec='white')
        if city_grid[cell][1:] == 'SW':
            ax.arrow(cell[1] + 1, cell[0] - ar_offset1, 0, .5, width=.15, head_width=.3, head_length=.25, fc='white',ec='white')
            ax.arrow(cell[1], cell[0] - ar_offset1, 0, .5, width=.15, head_width=.3, head_length=.25, fc='white', ec='white')
            ax.arrow(cell[1], cell[0] - ar_offset2, -.25, 0, width=.15, head_width=.3, head_length=.25, fc='white', ec='white')

    # Colors: black = edge, white = floor, green = goal, blue = agent
    #color_dict = {s: '#63636380', z: '#ffde2180', b: '#aa4a4480', r:'#82828280', o:'#ff000080', p:'#57a0d280'}
    color_dict = {s: '#AAAAAA' + alpha, z: '#ffde21'+ alpha, b: '#aa4a44'+ alpha, r: '#707070'+ alpha, o: '#ff0000'+ alpha, p: '#57a0d2'+ alpha}

    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)


if __name__ == '__main__':
    city_grid = gen_city((15, 15), (2, 2), True, 0)
    #print(city_grid)
    drivers = [{'start': (1, 4), 'goal': (13, 9), 'max_speed': 45, 'weight': 1},
               {'start': (1, 5), 'goal': (13, 9), 'max_speed': 45, 'weight': 1},
               ]
    parameters = {'debug': False, 'city_grid': city_grid, 'drivers': drivers}
    model = CityModel(parameters)
    model.setup()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    animation_plot(model, ax)
    plt.show()