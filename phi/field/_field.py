import warnings
from numbers import Number
from typing import Callable, Union, Tuple

from phi import math
from phi.geom import Geometry, Box, Point, BaseBox, UniformGrid, Mesh, Sphere, Graph
from phi.geom._geom import slice_off_constant_faces
from phi.math import Shape, Tensor, channel, non_batch, expand, instance, spatial, wrap, dual, non_dual
from phi.math.extrapolation import Extrapolation
from phi.math.magic import BoundDim, slicing_dict
from phiml.math import batch, Solve, DimFilter


class FieldInitializer:

    def _sample(self, geometry: Geometry, at: str, boundaries: Extrapolation, **kwargs) -> math.Tensor:
        """ For internal use only. Use `sample()` instead. """
        raise NotImplementedError(self)


class Field:
    """
    Base class for all fields.
    
    Important implementations:
    
    * CenteredGrid
    * StaggeredGrid
    * PointCloud
    * Noise
    
    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self,
                 geometry: Union[Geometry, Tensor],
                 values: Union[Tensor, Number, bool, Callable, FieldInitializer, Geometry, 'Field'],
                 boundary: Union[Number, Extrapolation, 'Field', dict],
                 **sampling_kwargs):
        """
        Args:
          elements: Geometry object specifying the sample points and sizes
          values: values corresponding to elements
          extrapolation: values outside elements
        """
        assert isinstance(geometry, Geometry), f"geometry must be a Geometry object but got {type(geometry).__name__}"
        self._boundary: Extrapolation = as_boundary(boundary, geometry)
        self._geometry: Geometry = geometry
        if isinstance(values, (Tensor, Number, bool)):
            values = wrap(values)
        else:
            from ._resample import sample
            values = sample(values, geometry, 'center', self._boundary, **sampling_kwargs)
        if non_batch(geometry).non_channel not in values.shape:
            values = expand(wrap(values), non_batch(geometry).non_channel)
        self._values: Tensor = values
        math.merge_shapes(values, non_batch(self.sampled_elements).non_channel)  # shape check

    @property
    def geometry(self) -> Geometry:
        """
        Returns a geometrical representation of the discrete volume elements.
        The result is a tuple of Geometry objects, each of which can have additional spatial (but not batch) dimensions.
        
        For grids, the geometries are boxes while particle fields may be represented as spheres.
        
        If this Field has no discrete points, this method returns an empty geometry.
        """
        return self._geometry

    @property
    def mesh(self) -> Mesh:
        """Cast `self.geometry` to a `phi.geom.Mesh`."""
        assert isinstance(self._geometry, Mesh), f"Geometry is not a mesh but {type(self._geometry)}"
        return self._geometry

    @property
    def graph(self) -> Graph:
        """Cast `self.geometry` to a `phi.geom.Graph`."""
        assert isinstance(self._geometry, Graph), f"Geometry is not a mesh but {type(self._geometry)}"
        return self._geometry

    @property
    def faces(self):
        return get_faces(self._geometry, self._boundary)

    @property
    def face_centers(self):
        return self._geometry.face_centers
        # return slice_off_constant_faces(self._geometry.face_centers, self._geometry.boundary_faces, self._boundary)

    @property
    def face_normals(self):
        return self._geometry.face_normals
        # return slice_off_constant_faces(self._geometry.face_normals, self._geometry.boundary_faces, self._boundary)

    @property
    def face_areas(self):
        return self._geometry.face_areas
        # return slice_off_constant_faces(self._geometry.face_areas, self._geometry.boundary_faces, self._boundary)

    @property
    def sampled_elements(self) -> Geometry:
        """
        If the values represent are sampled at the element centers or represent the whole element, returns `self.geometry`.
        If the values are sampled at the faces, returns `self.faces`.
        """
        return get_faces(self._geometry, self._boundary) if is_staggered(self._values, self._geometry) else self._geometry

    @property
    def elements(self):
        # raise SyntaxError("Field.elements is deprecated. Use Field.geometry or Field.sampled_elements instead.")
        warnings.warn("Field.elements is deprecated. Use Field.geometry or Field.sampled_elements instead. Field.elements now defaults to Field.geometry.", DeprecationWarning, stacklevel=2)
        return self._geometry

    @property
    def is_centered(self):
        return not self.is_staggered

    @property
    def is_staggered(self):
        return is_staggered(self._values, self._geometry)

    @property
    def center(self) -> Tensor:
        """ Returns the center points of the `elements` of this `Field`. """
        # { ~vector, x }      for staggered grid
        # { ~cells }          for unstructured mesh
        # { particles }       for graph
        # { ~particles }      for graph edges
        if self.is_centered:
            return slice_off_constant_faces(self._geometry.center, self._geometry.boundary_elements, self.extrapolation)
        elif self.is_staggered:
            return slice_off_constant_faces(self._geometry.face_centers, self._geometry.boundary_faces, self.extrapolation)
        else:
            raise NotImplementedError

    @property
    def points(self):
        return self.center

    @property
    def values(self) -> Tensor:
        """ Returns the `values` of this `Field`. """
        return self._values

    data = values

    def uniform_values(self):
        """
        Returns a uniform tensor containing `values`.

        For periodic grids, which always have a uniform value tensor, `values' is returned directly.
        If `values` is not uniform, it is padded as in `StaggeredGrid.staggered_tensor()`.
        """
        if self.values.shape.is_uniform:
            return self.values
        else:
            return self.staggered_tensor()

    @property
    def boundary(self) -> Extrapolation:
        """
        Returns the boundary conditions set for this `Field`.

        Returns:
            Single `Extrapolation` instance that encodes the (varying) boundary conditions for all boundaries of this field's `elements`.
        """
        return self._boundary

    @property
    def extrapolation(self) -> Extrapolation:
        """ Returns the `Extrapolation` of this `Field`. """
        return self._boundary

    @property
    def shape(self) -> Shape:
        """
        Returns a shape with the following properties
        
        * The spatial dimension names match the dimensions of this Field
        * The batch dimensions match the batch dimensions of this Field
        * The channel dimensions match the channels of this Field
        """
        if self.is_staggered:
            return batch(self._geometry) & self.resolution & non_dual(self._values).without(self.resolution) & self._geometry.shape['vector']
        return self._geometry.shape.non_channel & self._values

    @property
    def resolution(self):
        return self._geometry.shape.non_channel.non_dual.non_batch

    @property
    def spatial_rank(self) -> int:
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        """
        return self._geometry.spatial_rank

    @property
    def bounds(self) -> BaseBox:
        """
        The bounds represent the area inside which the values of this `Field` are valid.
        The bounds will also be used as axis limits for plots.

        The bounds can be set manually in the constructor, otherwise default bounds will be generated.

        For fields that are valid without bounds, the lower and upper limit of `bounds` is set to `-inf` and `inf`, respectively.

        Fields whose spatial rank is determined only during sampling return an empty `Box`.
        """
        if isinstance(self._geometry.bounds, BaseBox):
            return self._geometry.bounds
        extent = self._geometry.bounding_half_extent().vector.as_dual('_extent')
        points = self._geometry.center + extent
        lower = math.min(points, dim=points.shape.non_batch.non_channel)
        upper = math.max(points, dim=points.shape.non_batch.non_channel)
        return Box(lower, upper)

    box = bounds

    @property
    def is_grid(self):
        return isinstance(self._geometry, UniformGrid)

    @property
    def is_mesh(self):
        return isinstance(self._geometry, Mesh)

    @property
    def is_point_cloud(self):
        if isinstance(self._geometry, (UniformGrid, Mesh)):
            return False
        if isinstance(self._geometry, (BaseBox, Sphere, Point)):
            return True
        return True

    @property
    def dx(self) -> Tensor:
        assert spatial(self._geometry), f"dx is only defined for elements with spatial dims but Field has elements {self._geometry.shape}"
        return self.bounds.size / self.resolution

    @property
    def cells(self):
        assert isinstance(self._geometry, (UniformGrid, Mesh))
        return self._geometry

    def at_centers(self, **kwargs) -> 'Field':
        """
        Interpolates the values to the cell centers.

        See Also:
            `Field.at_faces()`, `Field.at()`, `resample`.

        Args:
            **kwargs: Sampling arguments.

        Returns:
            `CenteredGrid` sampled at cell centers.
        """
        if self.is_centered:
            return self
        from ._resample import sample
        values = sample(self, self._geometry, at='center', boundary=self._boundary, **kwargs)
        return Field(self._geometry, values, self._boundary)

    def at_faces(self, boundary=None, **kwargs) -> 'Field':
        if self.is_staggered and not boundary:
            return self
        boundary = as_boundary(boundary, self._geometry) if boundary else self._boundary
        from ._resample import sample
        values = sample(self, self._geometry, at='face', boundary=boundary, **kwargs)
        return Field(self._geometry, values, boundary)

    @property
    def sampled_at(self):
        return 'face' if self.is_staggered else 'center'

    def at(self, representation: 'Field', keep_extrapolation=False, **kwargs) -> 'Field':
        """
        Short for `resample(self, representation)`

        See Also
            `resample()`.

        Returns:
            Field object of same type as `representation`
        """
        from ._resample import resample
        return resample(self, representation, keep_extrapolation, **kwargs)

    def closest_values(self, points: Tensor):
        """
        Sample the closest grid point values of this field at the world-space locations (in physical units) given by `points`.
        Points must have a single channel dimension named `vector`.
        It may additionally contain any number of batch and spatial dimensions, all treated as batch dimensions.

        Args:
            points: world-space locations

        Returns:
            Closest grid point values as a `Tensor`.
            For each dimension, the grid points immediately left and right of the sample points are evaluated.
            For each point in `points`, a *2^d* cube of points is determined where *d* is the number of spatial dimensions of this field.
            These values are stacked along the new dimensions `'closest_<dim>'` where `<dim>` refers to the name of a spatial dimension.
        """
        warnings.warn("Field.closest_values() is deprecated.", DeprecationWarning, stacklevel=2)
        if isinstance(points, Geometry):
            points = points.center
        # --- CenteredGrid ---
        local_points = self.box.global_to_local(points) * self.resolution - 0.5
        return math.closest_grid_values(self.values, local_points, self.extrapolation)
        # --- StaggeredGrid ---
        if 'staggered_direction' in points.shape:
            points_ = math.unstack(points, '~vector')
            channels = [component.closest_values(p) for p, component in zip(points_, self.vector.unstack())]
        else:
            channels = [component.closest_values(points) for component in self.vector.unstack()]
        return math.stack(channels, points.shape['~vector'])

    def with_values(self, values, **sampling_kwargs):
        """ Returns a copy of this field with `values` replaced. """
        if not isinstance(values, (Tensor, Number)):
            from ._resample import sample
            values = sample(values, self._geometry, self.sampled_at, self._boundary, dot_face_normal=self._geometry if 'vector' not in self._values.shape else None, **sampling_kwargs)
        return Field(self._geometry, values, self._boundary)

    def with_boundary(self, boundary):
        """ Returns a copy of this field with the `boundary` replaced. """
        boundary = as_boundary(boundary, self._geometry)
        boundary_elements = 'boundary_faces' if self.is_staggered else 'boundary_elements'
        old_determined_slices = {k: s for k, s in getattr(self._geometry, boundary_elements).items() if self._boundary.determines_boundary_values(k)}
        new_determined_slices = {k: s for k, s in getattr(self._geometry, boundary_elements).items() if boundary.determines_boundary_values(k)}
        if old_determined_slices.values() == new_determined_slices.values():
            return Field(self._geometry, self._values, boundary)  # ToDo unnecessary once the rest is implemented
        to_add = {k: sl for k, sl in old_determined_slices.items() if sl not in new_determined_slices.values()}
        to_remove = [sl for sl in new_determined_slices.values() if sl not in old_determined_slices.values()]
        values = math.slice_off(self._values, *to_remove)
        if to_add:
            if self.is_mesh:
                values = self.mesh.pad_boundary(values, to_add, self._boundary)
            else:
                values = math.pad(values, list(to_add.values()), self._boundary, bounds=self.bounds)
        return Field(self._geometry, values, boundary)

    with_extrapolation = with_boundary

    def with_bounds(self, bounds: Box):
        """ Returns a copy of this field with `bounds` replaced. """
        order = list(bounds.vector.item_names)
        geometry = self._geometry.vector[order]
        new_shape = self._values.shape.without(order) & self._values.shape.only(order, reorder=True)
        values = math.transpose(self._values, new_shape)
        return Field(geometry, values, self._boundary)

    def with_geometry(self, elements: Geometry):
        """ Returns a copy of this field with `elements` replaced. """
        assert non_batch(elements) == non_batch(self._geometry), f"Field.with_elements() only accepts elements with equal non-batch dimensions but got {elements.shape} for Field with shape {self._geometry.shape}"
        return Field(elements, self._values, self._boundary)

    with_elements = with_geometry

    def shifted(self, delta):
        return self.with_elements(self._geometry.shifted(delta))

    def pad(self, widths: Union[int, tuple, list, dict]) -> 'Field':
        """
        Alias for `phi.field.pad()`.

        Pads this `Field` using its extrapolation.

        Unlike padding the values, this function also affects the `geometry` of the field, changing its size and origin depending on `widths`.

        Args:
            widths: Either `int` or `(lower, upper)` to pad the same number of cells in all spatial dimensions
                or `dict` mapping dimension names to `(lower, upper)`.

        Returns:
            Padded `Field`
        """
        from ._field_math import pad
        return pad(self, widths)

    def gradient(self,
                 boundary: Extrapolation = None,
                 at: str = 'center',
                 dims: math.DimFilter = spatial,
                 stack_dim: Union[Shape, str] = channel('vector'),
                 order=2,
                 implicit: Solve = None,
                 scheme=None,
                 upwind: 'Field' = None,
                 gradient_extrapolation: Extrapolation = None):
        """Alias for `phi.field.spatial_gradient`"""
        from ._field_math import spatial_gradient
        return spatial_gradient(self, boundary=boundary, at=at, dims=dims, stack_dim=stack_dim, order=order, implicit=implicit, scheme=scheme, upwind=upwind, gradient_extrapolation=gradient_extrapolation)

    def divergence(self, order=2, implicit: Solve = None, upwind: 'Field' = None):
        """Alias for `phi.field.divergence`"""
        from ._field_math import divergence
        return divergence(self, order=order, implicit=implicit, upwind=upwind)

    def curl(self, at='corner'):
        """Alias for `phi.field.curl`"""
        from ._field_math import curl
        return curl(self, at=at)

    def laplace(self,
                axes: DimFilter = spatial,
                gradient: 'Field' = None,
                order=2,
                implicit: math.Solve = None,
                weights: Union[Tensor, 'Field'] = None,
                upwind: 'Field' = None,
                correct_skew=True):
        """Alias for `phi.field.laplace`"""
        from ._field_math import laplace
        return laplace(self, axes=axes, gradient=gradient, order=order, implicit=implicit, weights=weights, upwind=upwind, correct_skew=correct_skew)

    def downsample(self, factor: int):
        from ._field_math import downsample2x
        result = self
        while factor >= 2:
            result = downsample2x(result)
            factor /= 2
        if math.close(factor, 1.):
            return result
        from ._resample import resample
        raise NotImplementedError(f"downsample does not support fractional re-sampling. Only 2^n currently supported.")

    def staggered_tensor(self) -> Tensor:
        """
        Stacks all component grids into a single uniform `phi.math.Tensor`.
        The individual components are padded to a common (larger) shape before being stacked.
        The shape of the returned tensor is exactly one cell larger than the grid `resolution` in every spatial dimension.

        Returns:
            Uniform `phi.math.Tensor`.
        """
        assert self.resolution.names == self.shape.get_item_names('vector'), "Field.staggered_tensor() only defined for Fields whose vector components match the resolution"
        padded = []
        for dim, component in zip(self.resolution.names, self.vector):
            widths = {d: (0, 1) for d in self.resolution.names}
            lo_valid, up_valid = self.extrapolation.valid_outer_faces(dim)
            widths[dim] = (int(not lo_valid), int(not up_valid))
            padded.append(math.pad(component.values, widths, self.extrapolation[{'vector': dim}], bounds=self.bounds))
        result = math.stack(padded, channel(vector=self.resolution))
        assert result.shape.is_uniform
        return result

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Field':
        from ._field_math import stack
        return stack(values, dim, kwargs.get('bounds', None))

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Field':
        from ._field_math import concat
        return concat(values, dim)

    def __and__(self, other):
        assert isinstance(other, Field)
        assert instance(self).rank == instance(other).rank == 1, f"Can only use & on PointClouds that have a single instance dimension but got shapes {self.shape} & {other.shape}"
        from ._field_math import concat
        return concat([self, other], instance(self))

    def __matmul__(self, other: 'Field'):  # value @ representation
        # Deprecated. Use `resample(value, field)` instead.
        warnings.warn("value @ field is deprecated. Use resample(value, field) instead.", DeprecationWarning)
        from ._resample import resample
        return resample(self, to=other, keep_extrapolation=False)

    def __rmatmul__(self, other):  # values @ representation
        if isinstance(other, (Geometry, Number, tuple, list, FieldInitializer)):
            warnings.warn("value @ field is deprecated. Use resample(value, field) instead.", DeprecationWarning)
            from ._resample import resample
            return resample(other, to=self, keep_extrapolation=False)
        return NotImplemented

    def __rshift__(self, other):
        warnings.warn(">> operator for Fields is deprecated. Use field.at(), the constructor or obj @ field instead.", SyntaxWarning, stacklevel=2)
        return self.at(other, keep_extrapolation=False)

    def __rrshift__(self, other):
        warnings.warn(">> operator for Fields is deprecated. Use field.at(), the constructor or obj @ field instead.", SyntaxWarning, stacklevel=2)
        if not isinstance(self, Field):
            return NotImplemented
        if isinstance(other, (Geometry, float, int, complex, tuple, list, FieldInitializer)):
            from ._resample import resample
            return resample(other, to=self, keep_extrapolation=False)
        return NotImplemented

    def __getitem__(self, item) -> 'Field':
        """
        Access a slice of the Field.
        The returned `Field` may be of a different type than `self`.

        Args:
            item: `dict` mapping dimensions (`str`) to selections (`int` or `slice`) or other supported type, such as `int` or `str`.

        Returns:
            Sliced `Field`.
        """
        item = slicing_dict(self, item)
        if not item:
            return self
        boundary = self._boundary[item]
        item_without_vec = {dim: selection for dim, selection in item.items() if dim != 'vector'}
        if self.is_staggered and 'vector' in item and '~vector' in self.geometry.face_shape:
            assert isinstance(self._geometry, UniformGrid), f"Vector slicing is only supported for grids"
            item['~vector'] = item['vector']
            del item['vector']
            elements = self.sampled_elements[item]
        else:
            elements = self._geometry[item_without_vec]
        values = self._values[item]
        return Field(elements, values, boundary)

    def __getattr__(self, name: str) -> BoundDim:
        return BoundDim(self, name)

    def dimension(self, name: str):
        """
        Returns a reference to one of the dimensions of this field.

        The dimension reference can be used the same way as a `Tensor` dimension reference.
        Notable properties and methods of a dimension reference are:
        indexing using `[index]`, `unstack()`, `size`, `exists`, `is_batch`, `is_spatial`, `is_channel`.

        A shortcut to calling this function is the syntax `field.<dim_name>` which calls `field.dimension(<dim_name>)`.

        Args:
            name: dimension name

        Returns:
            dimension reference

        """
        return BoundDim(self, name)

    def __value_attrs__(self):
        return '_values', '_boundary'

    def __variable_attrs__(self):
        return '_values', '_geometry'

    def __expand__(self, dims: Shape, **kwargs) -> 'Field':
        return self.with_values(expand(self.values, dims, **kwargs))

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Field':
        elements = math.rename_dims(self._geometry, dims, new_dims)
        values = math.rename_dims(self._values, dims, new_dims)
        extrapolation = math.rename_dims(self._boundary, dims, new_dims, **kwargs)
        return Field(elements, values, extrapolation)

    def __eq__(self, other):
        if not isinstance(other, Field):
            return False
        # Check everything but __variable_attrs__ (values): elements type, extrapolation, add_overlapping
        if type(self._geometry) is not type(other._geometry):
            return False
        if self._boundary != other.boundary:
            return False
        if self._values is None:
            return other._values is None
        if other._values is None:
            return False
        if self._values.shape == other._values.shape:
            return False
        return math.always_close(self._values, other._values)

    def __mul__(self, other):
        return self._op2(other, lambda d1, d2: d1 * d2)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._op2(other, lambda d1, d2: d1 / d2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda d1, d2: d2 / d1)

    def __sub__(self, other):
        return self._op2(other, lambda d1, d2: d1 - d2)

    def __rsub__(self, other):
        return self._op2(other, lambda d1, d2: d2 - d1)

    def __add__(self, other):
        return self._op2(other, lambda d1, d2: d1 + d2)

    __radd__ = __add__

    def __pow__(self, power, modulo=None):
        return self._op2(power, lambda f, p: f ** p)

    def __neg__(self):
        return self._op1(lambda x: -x)

    def __gt__(self, other):
        return self._op2(other, lambda x, y: x > y)

    def __ge__(self, other):
        return self._op2(other, lambda x, y: x >= y)

    def __lt__(self, other):
        return self._op2(other, lambda x, y: x < y)

    def __le__(self, other):
        return self._op2(other, lambda x, y: x <= y)

    def __abs__(self):
        return self._op1(lambda x: abs(x))

    def _op1(self: 'Field', operator: Callable) -> 'Field':
        """
        Perform an operation on the data of this field.

        Args:
          operator: function that accepts tensors and extrapolations and returns objects of the same type and dimensions

        Returns:
          Field of same type
        """
        values = operator(self.values)
        extrapolation_ = operator(self._boundary)
        return self.with_values(values).with_extrapolation(extrapolation_)

    def _op2(self, other, operator) -> 'Field':
        if isinstance(other, Geometry):
            raise ValueError(f"Cannot combine {self.__class__.__name__} with a Geometry, got {type(other)}")
        if isinstance(other, Field):
            if self._geometry == other._geometry:
                values = operator(self._values, other.values)
                extrapolation_ = operator(self._boundary, other.extrapolation)
                return Field(self._geometry, values, extrapolation_)
            from ._resample import sample
            other_values = sample(other, self._geometry, self.sampled_at, self.boundary, dot_face_normal=self._geometry)
            values = operator(self._values, other_values)
            boundary = operator(self._boundary, other.extrapolation)
            return Field(self._geometry, values, boundary)
        else:
            if isinstance(other, (tuple, list)) and len(other) == self.spatial_rank:
                other = math.wrap(other, self._geometry.shape['vector'])
            else:
                other = math.wrap(other)
            # try:
            #     boundary = operator(self._boundary, as_boundary(other, self._geometry))
            # except TypeError:  # e.g. ZERO_GRADIENT + constant
            boundary = self._boundary  # constants don't affect the boundary conditions (legacy reasons)
            if 'vector' in self.shape and 'vector' not in self.values.shape and '~vector' in self.values.shape:
                other = other.vector.as_dual()
            values = operator(self._values, other)
            return Field(self._geometry, values, boundary)

    def __repr__(self):
        if self.is_grid:
            type_name = "Grid" if self.is_centered else "Grid faces"
        elif self.is_mesh:
            type_name = "Mesh" if self.is_centered else "Mesh faces"
        elif self.is_point_cloud:
            type_name = "Point cloud" if self.is_centered else "Point cloud faces"
        else:
            type_name = self.__class__.__name__
        if self._values is not None:
            return f"{type_name}[{self.values}, ext={self._boundary}]"
        else:
            return f"{type_name}[{self.resolution}, ext={self._boundary}]"

    def grid_scatter(self, *args, **kwargs):
        """Deprecated. Use `sample` with `scatter=True` instead."""
        warnings.warn("Field.grid_scatter() is deprecated. Use field.sample() with scatter=True instead.", DeprecationWarning, stacklevel=2)
        from ._resample import grid_scatter
        return grid_scatter(self, *args, **kwargs)


def as_boundary(obj: Union[Extrapolation, Tensor, float, Field, None], geometry: Union[type, Geometry]) -> Extrapolation:
    """
    Returns an `Extrapolation` representing `obj`.

    Args:
        obj: One of

            * `float` or `Tensor`: Extrapolate with a constant value
            * `Extrapolation`: Use as-is.
            * `Field`: Sample values from `obj`, embedding another field inside `obj`.

    Returns:
        `Extrapolation`
    """
    if isinstance(obj, Field):
        from ._embed import FieldEmbedding
        return FieldEmbedding(obj)
    else:
        return math.extrapolation.as_extrapolation(obj, two_sided=isinstance(geometry, UniformGrid) or geometry == UniformGrid)


def is_staggered(values: Tensor, geometry: Geometry):
    return bool(dual(values)) and geometry.face_shape.dual in dual(values)


def get_faces(geometry: Geometry, boundary: Extrapolation):
    return slice_off_constant_faces(geometry.faces, geometry.boundary_faces, boundary)


def get_sample_points(geometry: Geometry, at: str, boundary: Extrapolation):
    if at == 'center':
        return slice_off_constant_faces(geometry.center, geometry.boundary_elements, boundary)
    elif at == 'face':
        return slice_off_constant_faces(geometry.face_centers, geometry.boundary_faces, boundary)
    raise ValueError(at)
