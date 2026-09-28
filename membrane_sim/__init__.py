"""3D membrane / shell simulator with fluid chambers and rigid obstacles, solved statically with Newton-Raphson."""
from .contact import Obstacle, ObstacleField
from .environment import Environment
from .fluid import FluidVolume
from .mesh import (box_mesh, boundary_nodes, closed_mesh_volume, disk_mesh, extrude_polygon,
                   rectangle_mesh, triangulate_polygon)
from .render import Renderer
from .shell import Shell
from .solver import NewtonSolver, SolveResult
