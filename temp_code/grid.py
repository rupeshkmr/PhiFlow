from phi.jax.flow import *
import matplotlib.pyplot as plt
field = Field(UniformGrid(x=2, y=2), values=0, boundary=PERIODIC)
# data =
field_values = field.numpy()
field_values[0,0] = 1
field_values[0,1] = -1
# plt.imshow(field_values)
# plt.colorbar()
# plt.show()
# field.values
field.values.numpy('x,y')[0,0] = 1
field.values.numpy('x,y')[1,1] = -1
print(field.values.numpy('x,y'))
print(field.values.numpy('x,y')[0,1])
print(field.center.numpy('x,y'))