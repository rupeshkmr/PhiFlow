from typing import Dict

from phi import math

from ._geom import Geometry, _fill_spatial_with_singleton
from ..math import wrap, Tensor


class Sphere(Geometry):
    """
    N-dimensional sphere.
    Defined through center position and radius.
    """

    def __init__(self, center, radius):
        self._center = wrap(center)
        assert 'vector' in self._center.shape, f"Sphere.center must have a 'vector' dimension. Try ({center},) * rank."
        self._radius = wrap(radius)

    @property
    def shape(self):
        return _fill_spatial_with_singleton(self._center.shape & self._radius.shape).without('vector')

    @property
    def radius(self):
        return self._radius

    @property
    def center(self):
        return self._center

    @property
    def volume(self) -> math.Tensor:
        return 4 / 3 * math.PI * self._radius ** 3

    @property
    def shape_type(self) -> Tensor:
        return math.tensor('S')

    def lies_inside(self, location):
        distance_squared = math.sum((location - self.center) ** 2, dim='vector')
        return math.any(distance_squared <= self.radius ** 2, self.shape.instance)  # union for instance dimensions

    def approximate_signed_distance(self, location):
        """
        Computes the exact distance from location to the closest point on the sphere.
        Very close to the sphere center, the distance takes a constant value.

        Args:
          location: float tensor of shape (batch_size, ..., rank)

        Returns:
          float tensor of shape (*location.shape[:-1], 1).

        """
        distance_squared = math.vec_squared(location - self.center)
        distance_squared = math.maximum(distance_squared, self.radius * 1e-2)  # Prevent infinite spatial_gradient at sphere center
        distance = math.sqrt(distance_squared)
        return math.min(distance - self.radius, self.shape.instance)  # union for instance dimensions

    def sample_uniform(self, *shape: math.Shape):
        raise NotImplementedError('Not yet implemented')  # ToDo

    def bounding_radius(self):
        return self.radius

    def bounding_half_extent(self):
        return self.radius

    def shifted(self, delta):
        return Sphere(self._center + delta, self._radius)

    def rotated(self, angle):
        return self

    def scaled(self, factor: float or Tensor) -> 'Geometry':
        return Sphere(self.center, self.radius * factor)

    def __variable_attrs__(self):
        return '_radius', '_center'

    def unstack(self, dimension: str) -> tuple:
        center = self.center.dimension(dimension).unstack(self.shape.get_size(dimension))
        radius = self.radius.dimension(dimension).unstack(self.shape.get_size(dimension))
        return tuple([Sphere(c, r) for c, r in zip(center, radius)])

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        raise NotImplementedError()

    def __hash__(self):
        return hash(self._center) + hash(self._radius)
