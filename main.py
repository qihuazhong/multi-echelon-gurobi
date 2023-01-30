from itertools import product
import gym
import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from plot_util import plot_gurobi_variables

from tqdm.notebook import tqdm
import os
import sys

sys.path.insert(0, os.path.abspath("./multi-echelon-drl/"))

# noinspection PyUnresolvedReferences
from register_envs import register_envs

register_envs()


def define_cont_variables(model, J: int, K: int):
    key_facilities_period = list(product(range(J), range(-1, K)))
    key_facilities_period_with_init = list(product(range(J), range(-4, K)))

    orders = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name="orders")
    _inventory = model.addVars(key_facilities_period, vtype=GRB.CONTINUOUS, name="inventory")
    _backlogs = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name="backlogs")
    _shipments = model.addVars(key_facilities_period_with_init, vtype=GRB.CONTINUOUS, name="shipments")

    return orders, _inventory, _backlogs, _shipments


def define_bin_variables(model, J: int, K: int):
    key_facilities_period = list(product(range(J), range(-1, K)))
    key_facilities_period_with_init = list(product(range(J), range(-4, K)))

    _x = model.addVars(key_facilities_period, vtype=GRB.BINARY, name="x")
    _y = model.addVars(key_facilities_period_with_init, vtype=GRB.BINARY, name="y")

    return _x, _y


def solve_MIP(demands=None, init_orders=None, init_backlogs=None, init_inventory=None, init_shipments=None):
    J = 4  # num of facilities
    K = len(demands)  # number of periods

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

    model = gp.Model("beer game")

    # define decisions variables
    _orders, _inventory, _backlogs, _shipments = define_cont_variables(model, J, K)

    # Fix initial conditions
    model.addConstrs(
        (_orders[j, k] == init_orders[j][k + 4] for j in range(J) for k in range(-4, 0)), name="init_orders"
    )

    model.addConstrs((_inventory[j, -1] == init_inventory[j] for j in range(J)), name="init_inv")

    model.addConstrs((_backlogs[j, -1] == init_backlogs[j] for j in range(J)), name="init_backlogs")

    model.addConstrs(
        (_shipments[j, k - 2] == init_shipments[j - 1][k] for j in range(1, J) for k in range(0, 2)),
        name="init_shipments",
    )

    # inventory and backlog
    # - Retailer
    model.addConstrs(
        (
            _inventory[0, k - 1] - _backlogs[0, k - 1] + _shipments[1, k - L_ship[0]]
            == demands[k] + _inventory[0, k] - _backlogs[0, k]
            for k in range(K)
        ),
        name="inv_transition_retailer",
    )

    # - Inner nodes
    model.addConstrs(
        (
            _inventory[j, k - 1] + _shipments[j + 1, k - L_ship[j]] == _shipments[j, k] + _inventory[j, k]
            for k in range(K)
            for j in range(1, J - 1)
        ),
        name="inv_transition_inner_nodes",
    )

    # - Upmost Supplier
    model.addConstrs(
        (
            _inventory[j, k - 1] + _orders[j, k - (L_ship[j] + L_info[j])] == _shipments[j, k] + _inventory[j, k]
            for k in range(K)
            for j in [J - 1]
        ),
        name="inv_transition_upmost_supplier",
    )

    # backlog
    model.addConstrs(
        (
            gp.quicksum([_shipments[j, t] for t in range(k + 1)])
            == gp.quicksum([_orders[j - 1, t - L_info[j - 1]] for t in range(k + 1)])
            - _backlogs[j, k]
            + _backlogs[j, -1]
            for j in range(1, J)
            for k in range(K)
        )
    )

    _x, _y = define_bin_variables(model, J, K)

    model.addConstrs((_inventory[j, k] <= _x[j, k] * 100000 for j in range(J) for k in range(K)))

    model.addConstrs((_backlogs[j, k] <= _y[j, k] * 100000 for j in range(J) for k in range(K)))

    model.addConstrs((_x[j, k] + _y[j, k] <= 1 for j in range(J) for k in range(K)))

    model.Params.LogToConsole = 0  # optimize in silence
    #     model.setParam("MIPGap", 1e-10)

    model.setObjective(
        gp.quicksum([_inventory[j, k] * c_h[j] for j in range(J) for k in range(K)])
        + gp.quicksum([_backlogs[j, k] * c_b[j] for j in range(J) for k in range(K)])
        + gp.quicksum(_orders[j, k] * 0.000001 for j in range(J) for k in range(K)),
        GRB.MINIMIZE,
    )

    model.optimize()

    return model, _orders, _inventory, _backlogs, _shipments


def solve_with_perfect_info(num_instances: int = 100):
    env_name = "BeerGameNormalMultiFacility-v0"
    env = gym.make(env_name)

    total_rewards = []

    print(
        f"Verifying whether objective value is consistent when the LP solution is plugged into the beer game environment"
    )
    for seed in tqdm(range(num_instances)):

        np.random.seed(seed)
        env.reset()

        init_inv = [node.current_inventory for key, node in env.sn.nodes.items()][:-1]
        init_orders = [arc.previous_orders[::-1] for key, arc in env.sn.arcs.items()][::-1]
        init_backlogs = [node.unfilled_demand for key, node in env.sn.nodes.items()][:-1]
        init_shipments = [arc.shipments.shipment_quantity_by_time for key, arc in env.sn.arcs.items()][::-1]

        demands = env.sn.nodes["retailer"].demands._demands

        m, orders, inventory, backlogs, shipments = solve_MIP(
            demands=demands,
            init_inventory=init_inv,
            init_backlogs=init_backlogs,
            init_orders=init_orders,
            init_shipments=init_shipments,
        )

        np.random.seed(seed)
        env.reset()
        terminated = False
        k = 0
        total_reward = 0

        while not terminated:
            actions = [orders[(j, k)].x for j in range(4)]
            obs, reward, terminated, info = env.step(actions)
            k += 1

            total_reward += reward

        total_rewards.append(total_reward)

        print(
            "Actual total reward:",
            total_reward,
            ", MIP result",
            m.objVal,
            ", Passed:",
            abs(m.objVal + total_reward) < 0.01,
        )
        assert abs(m.objVal + total_reward) < 0.01

    pd.DataFrame(total_rewards, columns=["Cost"]).to_csv("results/cost_perfect_info.csv")


def solve_once_MIP_with_expected_demand(num_instances: int = 100):

    env_name = "BeerGameNormalMultiFacility-v0"
    env = gym.make(env_name)

    total_rewards = []
    for seed in tqdm(range(num_instances)):

        np.random.seed(seed)
        env.reset()

        demands = env.sn.nodes["retailer"].demands._demands

        init_inv = [node.current_inventory for key, node in env.sn.nodes.items()][:-1]
        init_backlogs = [node.unfilled_demand for key, node in env.sn.nodes.items()][:-1]
        init_orders = [arc.previous_orders[::-1] for key, arc in env.sn.arcs.items()][::-1]
        init_shipments = [arc.shipments.shipment_quantity_by_time for key, arc in env.sn.arcs.items()][::-1]

        m, orders, inventory, backlogs, shipments = solve_MIP(
            demands=[10] * len(demands),
            init_inventory=init_inv,
            init_backlogs=init_backlogs,
            init_shipments=init_shipments,
            init_orders=init_orders,
        )

        terminated = False
        k = 0
        total_reward = 0

        while not terminated:

            actions = [orders[(j, k)].x for j in range(4)]
            obs, reward, terminated, info = env.step(actions)
            k += 1
            total_reward += reward

        total_rewards.append(total_reward)
        print("Total reward:", total_reward)

    pd.DataFrame(total_rewards, columns=["Cost"]).to_csv("results/cost_once_MIP.csv")


def solve_iterative_MIP_with_expected_demand(num_instances: int = 100):

    env_name = "BeerGameNormalMultiFacility-v0"
    env = gym.make(env_name)

    total_rewards = []
    for seed in tqdm(range(num_instances)):

        np.random.seed(seed)
        env.reset()

        demands = env.sn.nodes["retailer"].demands._demands

        terminated = False
        k = 0

        total_reward = 0

        while not terminated:

            init_inv = [node.current_inventory for key, node in env.sn.nodes.items()][:-1]
            init_backlogs = [node.unfilled_demand for key, node in env.sn.nodes.items()][:-1]
            init_orders = [arc.previous_orders[::-1] for key, arc in env.sn.arcs.items()][::-1]
            init_shipments = [arc.shipments.shipment_quantity_by_time for key, arc in env.sn.arcs.items()][::-1]

            m, orders, inventory, backlogs, shipments = solve_MIP(
                demands=[demands[k]] + [10] * len(demands[k + 1 :]),
                init_inventory=init_inv,
                init_backlogs=init_backlogs,
                init_shipments=init_shipments,
                init_orders=init_orders,
            )

            actions = [orders[(j, 0)].x for j in range(4)]
            obs, reward, terminated, info = env.step(actions)
            k += 1
            total_reward += reward

        total_rewards.append(total_reward)
        print("Total reward:", total_reward)

    pd.DataFrame(total_rewards, columns=["Cost"]).to_csv("results/cost_iterative_MIP.csv")


def main():
    solve_with_perfect_info(10)
    solve_once_MIP_with_expected_demand(10)
    solve_iterative_MIP_with_expected_demand(10)


main()
