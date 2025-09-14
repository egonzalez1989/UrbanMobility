from UrbanModelling import CityModel
from UrbanUtils import *
from agentpy import Experiment
import numpy as np

#
city_grid = gen_city(block_shape = (15, 15), city_shape=(3, 3), street_parking=True, crop = 4)
# Focusing on pedestrians. Increase number of obstacles in sidewalks
parameters = [{
        'seed': 0,
        'debug': False,
        'city_grid': city_grid,
        'initial_walker_count': 10,
        'initial_driver_count': 10,
        'obstacles': gen_obstacles(city_grid, obstacles = o)[0],
        'max_walker_spawn': 1,
        'max_driver_spawn': 1,
        'steps': 50,
        'display': False
    } for o in np.linspace(0, .2, 11)]

for p in parameters:
    model = CityModel(p).run()
    model.run()
#output1 = Experiment(CityModel, sample = parameters).run(n_jobs=-1)
#output2 = Experiment(CityModel, sample = parameters, record = True).run(n_jobs=-1)
