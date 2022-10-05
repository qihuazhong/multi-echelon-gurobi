from itertools import product
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from plot_util import plot_gurobi_variables


def solve_by_gurobi(demands=None, init_orders=None, init_backlogs=None, init_inventory=None):
    J = 4  # num of facilities
    K = 100  # num of periods

    L_info = [2, 2, 2, 1]  # info lead times
    L_ship = [2, 2, 2, 2]  # shipment lead times

    c_h = [1.0, 0.75, 0.5, 0.25]
    c_b = [10.0, 0.0, 0.0, 0.0]

    if demands is None:
        demands = np.round(np.maximum(10 + 2 * np.random.randn(K), 0)).astype(int)

    if init_orders is None:
        init_orders = [[4]*4]*4

    if init_inventory is None:
        init_inventory = [8] * 4

    if init_backlogs is None:
        init_backlogs = [0] * 4

    model = gp.Model('beer game')

    key_facilities_period = list(product(range(J), range(-1, K)))
    key_facilities_period_with_init = list(product(range(J), range(-4, K)))

    # define decisions variables
    _orders = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name='orders')
    _inventory = model.addVars(key_facilities_period, vtype=GRB.CONTINUOUS, name='inventory')
    _backlogs = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name='backlogs')
    _shipments = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name='shipments')

    # Constraints

    # initial conditions
    model.addConstrs(
        (_orders[j, k] == init_orders[j][k + 4] for j in range(J) for k in range(-4, 0)), name='init_orders')

    model.addConstrs(
        (_inventory[j, -1] == init_inventory[j] for j in range(J)), name='init_inv')

    model.addConstrs(
        (_backlogs[j, -1] == init_backlogs[j] for j in range(J)), name='init_backlogs')

    # inventory and backlog
    # - Retailer
    model.addConstrs(
        (_inventory[j, k - 1] + _backlogs[j, k] + _shipments[1, k - L_ship[j]] ==
         demands[k] + _inventory[j, k] + _backlogs[j, k - 1]
         for k in range(K) for j in [0]), name='inv_transition_retailer')

    # - Inner nodes
    model.addConstrs(
        (_inventory[j, k - 1] + _backlogs[j, k] + _shipments[j + 1, k - L_ship[j]] ==
         _orders[j - 1, k - L_info[j - 1]] + _inventory[j, k] + _backlogs[j, k - 1]
         for k in range(K) for j in range(1, J - 1)), name='inv_transition_suppliers')

    # - Upmost Supplier
    model.addConstrs(
        (_inventory[j, k - 1] + _backlogs[j, k] + _orders[j, k - (L_ship[j] + L_info[j])] ==
         _orders[j - 1, k - L_info[j - 1]] + _inventory[j, k] + _backlogs[j, k - 1]
         for k in range(K) for j in [J-1]), name='inv_transition_upmost_supplier')

    # shipments
    # - Upmost Supplier
    model.addConstrs(
        (_shipments[j, k] <= _inventory[j, k - 1] + _orders[j, k - (L_ship[j] + L_info[j])]
         for k in range(K) for j in [J-1]), name='shipments_constraints_upmost_supplier'
    )

    # - Inner nodes
    model.addConstrs(
        (_shipments[j, k] <= _inventory[j, k - 1] + _shipments[j + 1, k - L_ship[j]]
         for k in range(K) for j in range(1, J - 1)), name='shipments_inner_nodes'
    )

    # - Retailer
    model.addConstrs(
        (_shipments[j, k] <= _backlogs[j, k - 1] + _orders[j - 1, k - L_info[j]]
         for k in range(K) for j in range(1, J)), name='shipments_constraints_retailer'
    )

    # shipments <-> backlog
    model.addConstrs(
        (_shipments[j, k] - _orders[j - 1, k - L_info[j - 1]] == _backlogs[j, k - 1] - _backlogs[j, k]
         for j in range(1, J) for k in range(K))
    )

    model.addConstrs(
        (_shipments[j, k] - demands[k] == _backlogs[j, k - 1] - _backlogs[j, k]
         for j in range(0, 1) for k in range(K))
    )

    # shipments <-> inventory
    model.addConstrs(
        (_shipments[j + 1, k - L_ship[j]] - _shipments[j, k] == _inventory[j, k] - _inventory[j, k - 1]
         for j in range(J - 1) for k in range(K))
    )

    model.addConstrs(
        (_orders[j, k - (L_ship[j] + L_info[j])] - _shipments[j, k] == _inventory[j, k] - _inventory[j, k - 1]
         for j in range(J - 1, J) for k in range(K))
    )

    model.setObjective(
        gp.quicksum([_inventory[j, k] * c_h[j] for j in range(J) for k in range(K)]) +
        gp.quicksum([_backlogs[j, k] * c_b[j] for j in range(J) for k in range(K)]), GRB.MINIMIZE
    )

    model.optimize()

    return model, _orders, _inventory, _backlogs, _shipments


m, orders, inventory, backlogs, shipments = solve_by_gurobi()

plot_gurobi_variables(orders, name='orders')
plot_gurobi_variables(inventory, name='inventory')
plot_gurobi_variables(backlogs, name='backlogs')
plot_gurobi_variables(shipments, name='shipments')
