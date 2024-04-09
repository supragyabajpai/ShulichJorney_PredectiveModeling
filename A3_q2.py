import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from scipy.stats import binom

demand_dist_df = pd.read_csv(r'C:\Users\supra\Downloads\distributions.csv')

#----PARTB-------#

price_per_unit = 49.99
unit_cost = 24.44
num_stores = len(demand_dist_df)
trials = 50
scenarios_per_trial = 100

def simulate_demand_scenarios(demand_dist_df, scenarios_per_trial):
    demand_scenarios = {}
    for index, row in demand_dist_df.iterrows():
        n, p = row['n'], row['p']
        demand_scenarios[index] = binom.rvs(n, p, size=scenarios_per_trial)
    return demand_scenarios

model = gp.Model("OptimalOrderQuantity")

total_objectives = 0
avg_optimal_orders = np.zeros(num_stores)

for trial in range(trials):
    demand_scenarios = simulate_demand_scenarios(demand_dist_df, scenarios_per_trial)

    order_quantities = model.addVars(num_stores, lb=0, vtype=GRB.CONTINUOUS, name="OrderQuantity")
    above_deviation = model.addVars(num_stores, scenarios_per_trial, lb=0, vtype=GRB.CONTINUOUS, name="Above")
    below_deviation = model.addVars(num_stores, scenarios_per_trial, lb=0, vtype=GRB.CONTINUOUS, name="Below")

    underage_cost = sum(1.0/scenarios_per_trial * (price_per_unit - unit_cost) * below_deviation[store, scenario]
                        for store in range(num_stores)
                        for scenario in range(scenarios_per_trial))
    overage_cost = sum(1.0/scenarios_per_trial * (price_per_unit - unit_cost + unit_cost) * above_deviation[store, scenario]
                       for store in range(num_stores)
                       for scenario in range(scenarios_per_trial))
    revenue = sum(1.0/scenarios_per_trial * (price_per_unit - unit_cost) * order_quantities[store]
                  for store in range(num_stores)
                  for scenario in range(scenarios_per_trial))
    model.setObjective(revenue - underage_cost - overage_cost, GRB.MAXIMIZE)

    for store in range(num_stores):
        for scenario in range(scenarios_per_trial):
            model.addConstr(order_quantities[store] + below_deviation[store, scenario] == demand_scenarios[store][scenario] + above_deviation[store, scenario])

    model.optimize()

    total_objectives += model.objVal
    for store in range(num_stores):
        avg_optimal_orders[store] += order_quantities[store].x

    model.reset()

print("Objective: ", total_objectives/trials)

#--------PART C----------#
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import binom

distributions = pd.read_csv(r'C:\Users\supra\Downloads\distributions.csv')
cost_matrix = pd.read_csv(r'C:\Users\supra\Downloads\cost_matrix.csv', index_col=0)

selling_price = 49.99
cost_per_unit = 24.44
num_stores = len(distributions)
trials = 50
scenarios_per_trial = 100
scenario_costs = np.zeros(trials)

def simulate_demand(distributions, scenarios_per_trial):
    demand_scenarios = {}
    for index, row in distributions.iterrows():
        n, p = row['n'], row['p']
        demand_scenarios[index] = binom.rvs(n, p, size=scenarios_per_trial)
    return demand_scenarios

def solve_scenario(demand_scenario, cost_matrix):
    model = gp.Model("TransshipmentOptimization")

    order_quantities = model.addVars(num_stores, lb=0, vtype=GRB.CONTINUOUS, name="OrderQuantities")
    transshipments = model.addVars(num_stores, num_stores, lb=0, vtype=GRB.CONTINUOUS, name="Transshipments")

    profit = gp.quicksum(selling_price * demand_scenario[i] for i in range(num_stores)) - cost_per_unit * gp.quicksum(order_quantities[i] for i in range(num_stores))
    transshipment_cost = gp.quicksum(cost_matrix.iloc[i, j] * transshipments[i, j] for i in range(num_stores) for j in range(num_stores) if i < len(cost_matrix.index) and j < len(cost_matrix.columns))
    model.setObjective(profit - transshipment_cost, GRB.MAXIMIZE)

    for i in range(num_stores):
        model.addConstr(order_quantities[i] >= gp.quicksum(transshipments[j, i] for j in range(num_stores) if j != i))
        model.addConstr(gp.quicksum(transshipments[j, i] for j in range(num_stores) if j != i) <= demand_scenario[i])
        model.addConstr(gp.quicksum(transshipments[i, j] for j in range(num_stores) if j != i) == gp.quicksum(transshipments[j, i] for j in range(num_stores) if j != i))

    model.optimize()

    return model.objVal, model.getAttr('x', order_quantities)

for trial in range(trials):
    demand_scenarios = simulate_demand(distributions, scenarios_per_trial)

    for scenario in range(scenarios_per_trial):
        if scenario in demand_scenarios:
            obj_val, order_qty_solution = solve_scenario(demand_scenarios[scenario], cost_matrix)
            scenario_costs[trial] += obj_val / scenarios_per_trial

average_cost = np.mean(scenario_costs)
print("Average cost with transshipment: ", average_cost)