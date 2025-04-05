import os
from phi.jax.flow import *
from datetime import datetime

def save_plotly_anim(output_dir,**kwargs):
    if(kwargs is None):
        return None
    output_dir = output_dir + str(int(datetime.now().timestamp())) + "/"
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    for grid, anim in kwargs.items():
        anim.write_html(output_dir + grid + ".html")

def generate_sim_anim(output_dir, **kwargs):
    if(kwargs is None):
        print("No grid data passed!")
        return None
    grid_list = kwargs.keys()
    anim_data = {}
    # generate data
    for grid in grid_list:
        anim_data[grid] = plot(kwargs[grid], lib='plotly', animate='time', frame_time=100, repeat=True)
    # save data
    save_plotly_anim(output_dir,**anim_data)