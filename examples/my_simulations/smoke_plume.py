from PIL import Image
import os
import sys
sys.path.insert(1,"../../local-src/")
import matplotlib.pyplot as plt
from phi.jax.flow import *
from tqdm.notebook import trange
import numpy as np
from datetime import datetime
from sim_utils import generate_sim_anim

domain = Box(x=100, y=100)
inflow = Sphere(x=50, y=9.5, radius=5)
inflow_rate = 0.2
# TODO add a functionality to load checkpoints in simulation

# define timestep update each iteration
@jit_compile
def step(v, s, p, dt):
    s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
    buoyancy = resample(s * (0, 0.1), to=v)
    v = advect.mac_cormack(v, v, dt) + buoyancy * dt
    v, p = fluid.make_incompressible(v, (), Solve('CG', 1e-3, x0=p))
    return v, s, p

v0 = StaggeredGrid(0, 0, domain, x=64, y=64)
smoke0 = CenteredGrid(0, ZERO_GRADIENT, domain, x=200, y=200)
v_trj, s_trj, p_trj = iterate(step, batch(time=300), v0, smoke0, None, dt=.5, range=trange, substeps=3)
print(type(v_trj))

# Generate animation and save it
output_dir = "../../simulation_output/smoke_plume/"
generate_sim_anim(output_dir,**{"density": s_trj, "velocity": v_trj, "pressure": p_trj})
# TODO add vorticity computation and visualization
# TODO add validation code