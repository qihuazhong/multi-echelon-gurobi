from itertools import product
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from plot_util import plot_gurobi_variables

from tqdm.notebook import tqdm
import os
import sys

sys.path.insert(0, os.path.abspath('./multi-echelon-drl/'))

from register_envs import register_envs
register_envs()


def define_cont_variables(model, J: int, K: int):
    key_facilities_period = list(product(range(J), range(-1, K)))
    key_facilities_period_with_init = list(product(range(J), range(-4, K)))

    orders = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name='orders')
    _inventory = model.addVars(key_facilities_period, vtype=GRB.CONTINUOUS, name='inventory')
    _backlogs = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name='backlogs')
    _shipments = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name='shipments')

    return orders, _inventory, _backlogs, _shipments


def define_bin_variables(model, J: int, K: int):
    key_facilities_period = list(product(range(J), range(-1, K)))
    key_facilities_period_with_init = list(product(range(J), range(-4, K)))

    _x = model.addVars(key_facilities_period, vtype=GRB.BINARY, name='x')
    _y = model.addVars(key_facilities_period_with_init, vtype=GRB.BINARY, name='y')

    return _x, _y


def solve_formulation(demands=None, init_orders=None, init_backlogs=None, init_inventory=None, init_shipments=None):
    J = 4  # num of facilities
    K = len(demands) # number of periods

    L_info = [2, 2, 2, 1]  # info lead times
    L_ship = [2, 2, 2, 2]  # shipment lead times

    c_h = [1.0, 0.75, 0.5, 0.25]
    c_b = [10.0, 0.00000001, 0.00000001, 0.00000001]  # a small positive value to prevent excessive ordering

    #     c_h = [0.5, 0.5, 0.5, 0.5]
    #     c_b = [1.0, 1.0, 1.0, 1.0]

    if demands is None:
        demands = np.round(np.maximum(10 + 2 * np.random.randn(K), 0)).astype(int)

    if init_orders is None:
        init_orders = [[4] * 4] * 4

    if init_inventory is None:
        init_inventory = [8] * 4

    if init_backlogs is None:
        init_backlogs = [0] * 4

    model = gp.Model('beer game')

    # define decisions variables
    _orders, _inventory, _backlogs, _shipments = define_cont_variables(model, J, K)

    # Fix initial conditions
    model.addConstrs(
        (_orders[j, k] == init_orders[j][k + 4] for j in range(J) for k in range(-4, 0)), name='init_orders')

    model.addConstrs(
        (_inventory[j, -1] == init_inventory[j] for j in range(J)), name='init_inv')

    model.addConstrs(
        (_backlogs[j, -1] == init_backlogs[j] for j in range(J)), name='init_backlogs')

    model.addConstrs(
        (_shipments[j, k - 2] == init_shipments[j - 1][k] for j in range(1, J) for k in range(0, 2)),
        name='init_shipments')

    # inventory and backlog
    # - Retailer
    model.addConstrs(
        (_inventory[0, k - 1] - _backlogs[0, k - 1] + _shipments[1, k - L_ship[0]] ==
         demands[k] + _inventory[0, k] - _backlogs[0, k]
         for k in range(K)), name='inv_transition_retailer')

    # - Inner nodes
    model.addConstrs(
        (_inventory[j, k - 1] + _shipments[j + 1, k - L_ship[j]] ==
         _shipments[j, k] + _inventory[j, k]
         for k in range(K) for j in range(1, J - 1)), name='inv_transition_inner_nodes')

    # - Upmost Supplier
    model.addConstrs(
        (_inventory[j, k - 1] + _orders[j, k - (L_ship[j] + L_info[j])] ==
         _shipments[j, k] + _inventory[j, k]
         for k in range(K) for j in [J - 1]), name='inv_transition_upmost_supplier')

    # backlog
    model.addConstrs(
        (gp.quicksum([_shipments[j, t] for t in range(k + 1)]) ==
         gp.quicksum([_orders[j - 1, t - L_info[j - 1]] for t in range(k + 1)]) - _backlogs[j, k] + _backlogs[j, -1]
         for j in range(1, J) for k in range(K))
    )

    _x, _y = define_bin_variables(model, J, K)

    model.addConstrs(
        (_inventory[j, k] <= _x[j, k] * 100000 for j in range(J) for k in range(K))
    )

    model.addConstrs(
        (_backlogs[j, k] <= _y[j, k] * 100000 for j in range(J) for k in range(K))
    )

    model.addConstrs(
        (_x[j, k] + _y[j, k] <= 1 for j in range(J) for k in range(K))
    )

    model.Params.LogToConsole = 0  # optimize in silence
    #     model.setParam("MIPGap", 1e-10)

    model.setObjective(
        gp.quicksum([_inventory[j, k] * c_h[j] for j in range(J) for k in range(K)]) +
        gp.quicksum([_backlogs[j, k] * c_b[j] for j in range(J) for k in range(K)]) +
        gp.quicksum(_orders[j, k] * 0.000001 for j in range(J) for k in range(K)),
        GRB.MINIMIZE
    )

    model.optimize()

    return model, _orders, _inventory, _backlogs, _shipments
