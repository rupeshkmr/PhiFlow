from phi.flow import *
# Simulate PDEs using grids
values = math.random_normal(spatial(x=500, y=500))
bounds = Box['x,y',0:1,0:1]
# Scalar Grid
# grid = CenteredGrid(Noise(), extrapolation.ZERO, bounds,x=500,y=500)
# grid2 = CenteredGrid(grid,extrapolation.PERIODIC,Box['x,y',-1:2, -1:2],x=50,y=500)
# vis.plot(grid, grid2, show_color_bar=False)
# vis.show()
# Vector Grid
# grid = CenteredGrid(Noise(channel(vector=2), batch(batch=2)), extrapolation.PERIODIC, Box['x,y',0:1, 0:1], x=20, y=20)*0.01
# grid2 = CenteredGrid(grid, 0, Box['x,y',-1:2, -1:2], x=20, y=20)
# vis.plot(grid, grid2)
# vis.show()
grid = CenteredGrid(Noise(channel(vector=2), batch(batch=2)), extrapolation.PERIODIC, Box['x,y',0:1, 0:1], x=20, y=20)*0.01
v = grid.batch[0]
# vs = []
# for i in range(10):
#     # v = diffuse.explicit(v,0.1,dt=.1, substeps=100)
#     v = advect.semi_lagrangian(v,v,dt=1)
#     vs.append(v)
# vis.plot([x.vector['x'] for x in vs], show_color_bar=False)
# vis.show()
print(v.points)