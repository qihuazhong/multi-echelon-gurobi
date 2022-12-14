from itertools import product
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from plot_util import plot_gurobi_variables


# class BeerGameGurobi:
#
#     def __init__(self, demands=None, init_orders=None, init_backlogs=None, init_inventory=None):
#         self.J = 4  # num of facilities
#         self.K = 100  # num of periods
#
#         self.L_info = [2, 2, 2, 1]  # info lead times
#         self.L_ship = [2, 2, 2, 2]  # shipment lead times
#
#         self.c_h = [1.0, 0.75, 0.5, 0.25]
#         self.c_b = [1.0, 0.0, 0.0, 10.0]
#
#         self.model = self.init_model_and_variables(demands, init_orders, init_backlogs, init_inventory)
#
#     def init_model_and_variables(self, demands, init_orders, init_backlogs, init_inventory):
#         if demands is None:
#             demands = np.round(np.maximum(10 + 2 * np.random.randn(self.K), 0)).astype(int)
#
#         if init_orders is None:
#             init_orders = [[4] * 4] * 4
#
#         if init_inventory is None:
#             init_inventory = [8] * 4
#
#         if init_backlogs is None:
#             init_backlogs = [0] * 4
#
#         model = gp.Model('beer game')
#
#         key_facilities_period = list(product(range(J), range(-1, K)))
#         key_facilities_period_with_init = list(product(range(J), range(-4, K)))
#
#         # define decisions variables
#         self._orders = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name='orders')
#         self._inventory = model.addVars(key_facilities_period, vtype=GRB.CONTINUOUS, name='inventory')
#         self._backlogs = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name='backlogs')
#         self._shipments = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name='shipments')
#
#         model.addConstrs(
#             (self._orders[j, k] == init_orders[j][k + 4] for j in range(J) for k in range(-4, 0)), name='init_orders')
#
#         model.addConstrs(
#             (self._inventory[j, -1] == init_inventory[j] for j in range(J)), name='init_inv')
#
#         model.addConstrs(
#             (self._backlogs[j, -1] == init_backlogs[j] for j in range(J)), name='init_backlogs')
#
#         return model


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
        (_inventory[j, k - 1] + _backlogs[j, k] + _shipments[j+1, k - L_ship[j]] ==
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

    # shipments: must be less or equal than inventory from last period plus newly received
    # - Upmost Supplier
    model.addConstrs(
        (_shipments[j, k] <= _inventory[j, k - 1] + _orders[j, k - (L_ship[j] + L_info[j])]
         for k in range(K) for j in [J-1]), name='shipments_constraints_upmost_supplier'
    )

    # - Inner nodes and the retailer
    model.addConstrs(
        (_shipments[j, k] <= _inventory[j, k - 1] + _shipments[j + 1, k - L_ship[j]]
         for k in range(K) for j in range(0, J - 1)), name='shipments_constraints'
    )

    # shipments: must be less than the accumulated backlogs and the new demand (Retailer Only)
    # - Retailer
    model.addConstrs(
        (_shipments[j, k] <= _backlogs[j, k - 1] + demands[k]
         for k in range(K) for j in [0]), name='shipments_constraints_retailer'
    )

    # - Other nodes
    # model.addConstrs(
    #     (_shipments[j, k] <= _backlogs[j, k - 1] + _orders[j - 1, k - L_info[j]]
    #      for k in range(K) for j in range(1, J)), name='shipments_constraints_retailer'
    # )

    # shipments <-> backlog
    # Other nodes
    model.addConstrs(
        (_shipments[j, k] - _orders[j - 1, k - L_info[j - 1]] == _backlogs[j, k - 1] - _backlogs[j, k]
         for j in range(1, J) for k in range(K))
    )

    # Retailer
    model.addConstrs(
        (_shipments[j, k] - demands[k] == _backlogs[j, k - 1] - _backlogs[j, k]
         for j in [0] for k in range(K))
    )

    # # shipments <-> inventory
    # Other nodes
    model.addConstrs(
        (_shipments[j + 1, k - L_ship[j]] - _shipments[j, k] == _inventory[j, k] - _inventory[j, k - 1]
         for j in range(J - 1) for k in range(K))
    )

    # Upmost supplier
    model.addConstrs(
        (_orders[j, k - (L_ship[j] + L_info[j])] - _shipments[j, k] == _inventory[j, k] - _inventory[j, k - 1]
         for j in [J - 1] for k in range(K))
    )

    model.setObjective(
        gp.quicksum([_inventory[j, k] * c_h[j] for j in range(J) for k in range(K)]) +
        gp.quicksum([_backlogs[j, k] * c_b[j] for j in range(J) for k in range(K)]), GRB.MINIMIZE
    )

    model.optimize()

    return model, _orders, _inventory, _backlogs, _shipments

demands = [8] * 100

m, orders, inventory, backlogs, shipments = solve_by_gurobi(demands=demands)
#
plot_gurobi_variables(orders, name='orders')
plot_gurobi_variables(inventory, name='inventory')
plot_gurobi_variables(backlogs, name='backlogs')
plot_gurobi_variables(shipments, name='shipments')
