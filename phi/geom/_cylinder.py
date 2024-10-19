from dataclasses import dataclass
from functools import cached_property
from typing import Union, Dict, Tuple, Optional, Sequence

from phiml import math
from phiml.math import Shape, dual, wrap, Tensor, expand, vec, where, ccat, clip, length, normalize, rotate_vector, minimum, vec_squared, rotation_matrix, channel, instance, stack, maximum
from phiml.math._magic_ops import all_attributes
from phiml.math.magic import slicing_dict
from ._geom import Geometry, _keep_vector
from ._sphere import Sphere


@dataclass(frozen=True)
class Cylinder(Geometry):
    """
    N-dimensional cylinder.
    Defined by center position, radius, depth, alignment axis, rotation.

    For cylinders whose bottom and top lie outside the domain or are otherwise not needed, you may use `infinite_cylinder` instead, which simplifies computations.
    """

    _center: Tensor
    radius: Tensor
    depth: Tensor
    rotation: Tensor  # rotation matrix
    axis: str

    variables: Tuple[str, ...] = ('_center', 'radius', 'depth', 'rotation')
    values: Tuple[str, ...] = ()

    @property
    def center(self) -> Tensor:
        return self._center

    @cached_property
    def shape(self) -> Shape:
        return self._center.shape & self.radius.shape & self.depth.shape

    @cached_property
    def radial_axes(self) -> Sequence[str]:
        return [d for d in self._center.vector.item_names if d != self.axis]

    @cached_property
    def volume(self) -> math.Tensor:
        return Sphere.volume_from_radius(self.radius, self.spatial_rank - 1) * self.depth

    @cached_property
    def up(self):
        return math.rotate_vector(vec(**{d: 1 if d == self.axis else 0 for d in self._center.vector.item_names}), self.rotation)

    def lies_inside(self, location):
        pos = rotate_vector(location - self._center, self.rotation, invert=True)
        r = pos.vector[self.radial_axes]
        h = pos.vector[self.axis]
        inside = (vec_squared(r) <= self.radius**2) & (h >= -.5*self.depth) & (h <= .5*self.depth)
        return math.any(inside, instance(self))  # union for instance dimensions

    def approximate_signed_distance(self, location: Union[Tensor, tuple]):
        location = math.rotate_vector(location - self._center, self.rotation, invert=True)
        r = location.vector[self.radial_axes]
        h = location.vector[self.axis]
        top_h = .5*self.depth
        bot_h = -.5*self.depth
        # --- Compute distances ---
        radial_outward = normalize(r, epsilon=1e-5)
        surf_r = radial_outward * self.radius
        radial_dist2 = vec_squared(r)
        inside_cyl = radial_dist2 <= self.radius**2
        clamped_r = where(inside_cyl, r, surf_r)
        # --- Closest point on bottom / top ---
        sgn_dist_side = abs(h) - top_h
        # --- Closest point on cylinder ---
        sgn_dist_cyl = length(r) - self.radius
        # inside (all <= 0) -> largest SDF, outside (any > 0) -> largest positive SDF
        sgn_dist = maximum(sgn_dist_cyl, sgn_dist_side)
        return math.min(sgn_dist, instance(self))

    def approximate_closest_surface(self, location: Tensor):
        location = math.rotate_vector(location - self._center, self.rotation, invert=True)
        r = location.vector[self.radial_axes]
        h = location.vector[self.axis]
        top_h = .5*self.depth
        bot_h = -.5*self.depth
        # --- Compute distances ---
        radial_outward = normalize(r, epsilon=1e-5)
        surf_r = radial_outward * self.radius
        radial_dist2 = vec_squared(r)
        inside_cyl = radial_dist2 <= self.radius**2
        clamped_r = where(inside_cyl, r, surf_r)
        # --- Closest point on bottom / top ---
        above = h >= 0
        flat_h = where(above, top_h, bot_h)
        on_flat = ccat([flat_h, clamped_r], self._center.shape['vector'])
        normal_flat = where(above, self.up, -self.up)
        # --- Closest point on cylinder ---
        clamped_h = clip(h, bot_h, top_h)
        on_cyl = ccat([surf_r, clamped_h], self._center.shape['vector'])
        normal_cyl = ccat([radial_outward, 0], self._center.shape['vector'], expand_values=True)
        # --- Choose closest ---
        d_flat = length(on_flat - location)
        d_cyl = length(on_cyl - location)
        flat_closer = d_flat <= d_cyl
        surf_point = where(flat_closer, on_flat, on_cyl)
        inside = inside_cyl & (h >= bot_h) & (h <= top_h)
        sgn_dist = minimum(d_flat, d_cyl) * where(inside, -1, 1)
        delta = surf_point - location
        normal = where(flat_closer, normal_flat, normal_cyl)
        delta = rotate_vector(delta, self.rotation)
        normal = rotate_vector(normal, self.rotation)
        if instance(self):
            sgn_dist, delta, normal = math.at_min((sgn_dist, delta, normal), key=sgn_dist, dim=instance(self))
        return sgn_dist, delta, normal, None, None

    def sample_uniform(self, *shape: math.Shape):
        raise NotImplementedError

    def bounding_radius(self):
        return math.length(vec(rad=self.radius, dep=.5*self.depth))

    def bounding_half_extent(self):
        if self.rotation is not None:
            return expand(self.bounding_radius(), self._center.shape.only('vector'))
        return ccat([.5*self.depth, expand(self.radius, channel(vector=self.radial_axes))], self._center.shape['vector'])

    def at(self, center: Tensor) -> 'Geometry':
        return Cylinder(center, self.radius, self.depth, self.rotation, self.axis, self.variables, self.values)

    def rotated(self, angle):
        if self.rotation is None:
            return Cylinder(self._center, self.radius, self.depth, angle, self.axis, self.variables, self.values)
        else:
            matrix = self.rotation @ (angle if dual(angle) else math.rotation_matrix(angle))
            return Cylinder(self._center, self.radius, self.depth, matrix, self.axis, self.variables, self.values)

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return Cylinder(self._center, self.radius * factor, self.depth * factor, self.rotation, self.axis, self.variables, self.values)

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        return Cylinder(self._center[_keep_vector(item)], self.radius[item], self.depth[item], math.slice(self.rotation, item), self.axis, self._variables, self.values)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(isinstance(v, Cylinder) for v in values) and all(v.axis == values[0].axis for v in values):
            var_attrs = set()
            var_attrs.update(*[set(v.variables) for v in values])
            val_attrs = set()
            val_attrs.update(*[set(v.values) for v in values])
            if any(v.rotation is not None for v in values):
                matrices = [v.rotation for v in values]
                if any(m is None for m in matrices):
                    any_angle = math.rotation_angles([m for m in matrices if m is not None][0])
                    unit_matrix = math.rotation_matrix(any_angle * 0)
                    matrices = [unit_matrix if m is None else m for m in matrices]
                rotation = stack(matrices, dim, **kwargs)
            else:
                rotation = None
            center = stack([v.center for v in values], dim, simplify=True, **kwargs)
            radius = stack([v.radius for v in values], dim, simplify=True, **kwargs)
            depth = stack([v.depth for v in values], dim, simplify=True, **kwargs)
            return Cylinder(center, radius, depth, rotation, values[0].axis, tuple(var_attrs), tuple(val_attrs))
        else:
            return Geometry.__stack__(values, dim, **kwargs)

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError(f"Cylinder.faces not implemented.")

    @property
    def face_centers(self) -> Tensor:
        raise NotImplementedError

    @property
    def face_areas(self) -> Tensor:
        raise NotImplementedError

    @property
    def face_normals(self) -> Tensor:
        raise NotImplementedError

    @property
    def boundary_elements(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return {}

    @property
    def face_shape(self) -> Shape:
        return self.shape.without('vector') & dual(shell='bottom,top,lateral')

    @property
    def corners(self) -> Tensor:
        return math.zeros(self.shape & dual(corners=0))

    def __eq__(self, other):
        return Geometry.__eq__(self, other)


def cylinder(center: Tensor = None,
             radius: Union[float, Tensor] = None,
             depth: Union[float, Tensor] = None,
             rotation: Optional[Tensor] = None,
             axis=-1,
             variables=('center', 'radius', 'depth', 'rotation'),
             **center_: Union[float, Tensor]):
    """
    Args:
        center: Cylinder center as `Tensor` with `vector` dimension.
            The spatial dimension order should be specified in the `vector` dimension via item names.
            Can be left empty to specify dimensions via kwargs.
        radius: Cylinder radius as `float` or `Tensor`.
        depth: Cylinder length as `float` or `Tensor`.
        rotation: Rotation angle(s) or rotation matrix.
        axis: The cylinder is aligned along this axis, perturbed by `rotation`.
        variables: Which properties of the cylinder are variable, i.e. traced and optimizable. All by default.
        **center_: Specifies center when the `center` argument is not given. Center position by dimension, e.g. `x=0.5, y=0.2`.
    """
    if center is not None:
        assert isinstance(center, Tensor), f"center must be a Tensor but got {type(center).__name__}"
        assert 'vector' in center.shape, f"Sphere center must have a 'vector' dimension."
        assert center.shape.get_item_names('vector') is not None, f"Vector dimension must list spatial dimensions as item names. Use the syntax Sphere(x=x, y=y) to assign names."
        center = center
    else:
        center = wrap(tuple(center_.values()), channel(vector=tuple(center_.keys())))
    radius = wrap(radius)
    depth = wrap(depth)
    rotation = rotation_matrix(rotation)
    axis = center.vector.item_names[axis] if isinstance(axis, int) else axis
    variables = [{'center': '_center'}.get(v, v) for v in variables]
    assert 'vector' not in radius.shape, f"Cylinder radius must not vary along vector but got {radius}"
    assert set(variables).issubset(set(all_attributes(Cylinder))), f"Invalid variables: {variables}"
    assert axis in center.vector.item_names, f"Cylinder axis {axis} not part of vector dim {center.vector}"
    return Cylinder(center, radius, depth, rotation, axis, tuple(variables), ())
