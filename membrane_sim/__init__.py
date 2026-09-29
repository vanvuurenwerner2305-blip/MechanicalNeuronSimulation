"""3D membrane / shell simulator with fluid chambers and rigid obstacles, solved statically with Newton-Raphson."""
from .characterise import (BIGGEST_ERROR, LOWEST_TOTAL, activation_sensitivities, chamber_compliance,
                           fit_neuron_equation, input_paths, equation_align, equation_lines, evaluate_weight, input_weights,
                           neuron_equation_latex, polyfit_weight, polynomial_text, rebuild_activation_pressure,
                           solve_activation, weight_coefficients, weight_degree)
from .contact import Obstacle, ObstacleField
from .environment import Environment
from .fluid import FluidVolume
from .mesh import (box_mesh, boundary_nodes, closed_mesh_volume, disk_mesh, extrude_polygon,
                   rectangle_mesh, triangulate_polygon)
from .render import Renderer
from .shell import Shell
from .solver import NewtonSolver, SolveResult
