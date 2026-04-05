import logging
from typing import Any, Dict, Tuple
import gurobipy as gp
from gurobipy import GRB

from src.utilities.enums import Objectives
from src.utilities.tools import get_durations, get_costs


class OfflineSolver:
    """
    A class to solve the taxi routing problem using a MIP solver (Gurobi).

        Attributes:
        ------------
        objective: Objectives(Enum)
            The objective used to evaluate the effectiveness of the plan
        duration : dictionary
            travel time matrix between possible stop points
        costs: dictionary
            driving costs
        model: gp.Model
            The Gurobi model for optimization.
        X_var: Dict[Tuple[int, int], gp.Var]
            Decision variables for trip connections.
        Y_var: Dict[Tuple[int, int], gp.Var]
            Decision variables for assigning trips to vehicles.
        Z_var: Dict[int, gp.Var]
            Decision variables indicating if a trip is served.
        U_var: Dict[int, gp.Var]
            Decision variables for departure times from locations.
        """

    def __init__(self,
                 network: Any,
                 objective: Objectives,
                 weight: float = 0.5
                 ) -> None:
        """
        Initialize the OfflineSolver.

        Input:
        ------------
        network: Any
            The road network, including nodes representing stop points.
        objective: Objectives
            The objective used to evaluate the effectiveness of the plan.
        weight: float, optional
            Used only for MULTI_OBJECTIVE objective (ignored otherwise). Default 0.5.
        """
        self.objective = objective
        self.weight = weight

        self.objective_value = 0
        self.durations = get_durations(network)
        self.costs = get_costs(network)

        self.model = gp.Model("TaxiRoutingModel")
        # Set OutputFlag based on the logging level
        if logging.getLogger().getEffectiveLevel() > logging.INFO:
            self.model.setParam('OutputFlag', 0)  # Disable solver output
        else:
            self.model.setParam('OutputFlag', 1)  # Enable solver output
        self.X_var: Dict[Tuple[int, int], gp.Var] = {}  # Decision variables for trip connections between customers
        self.Y_var: Dict[Tuple[int, int], gp.Var] = {}  # Decision variables for assigning customers to vehicles
        self.Z_var: Dict[int, gp.Var] = {}  # Decision variables for whether a customer is served
        self.U_var: Dict[int, gp.Var] = {}  # Decision variables for departure times from customer locations

    def define_objective(self, K, P, vehicle_request_assign):
        """
        Define the objective function based on the selected objective.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.
        """
        if self.objective == Objectives.TOTAL_PROFIT:
            self.define_total_profit_objective(K, P, vehicle_request_assign)
        elif self.objective == Objectives.TOTAL_CUSTOMERS:
            self.define_total_customers_objective(P)
        elif self.objective == Objectives.WAIT_TIME:
            self.define_total_wait_time_objective(P)
        elif self.objective == Objectives.MULTI_OBJECTIVE:
            self.define_multi_objective(K, P, vehicle_request_assign)
        else:
            raise ValueError(f"Objective {self.objective} not recognized.")

    def define_total_customers_objective(self, P):
        """
        Define objective of maximizing the total number of served customers and add it to the model.

        Input:
        ------------
        P: List[Any]
            Set of customers to serve.
        """
        self.model.setObjective(
            sum(self.Z_var[f_i.id] for f_i in P),
            sense=GRB.MAXIMIZE
        )

    def define_total_profit_objective(self, K, P, vehicle_request_assign):
        """
        Define objective of maximizing the total profit and add it to the model.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.

        Hint:
        ------------
        - Access costs via self.costs[from_location][to_location]
        - Access vehicle departure stop via vehicle_request_assign[vehicle_id].departure_stop
        
        """

        # Revenue from served customers
        revenue = gp.quicksum(f_i.fare * self.Z_var[f_i.id] for f_i in P)

        # Cost: vehicle to first customer pickup
        cost_vehicle_to_first = gp.quicksum(
            self.costs[vehicle_request_assign[f_k.id].departure_stop][f_i.origin.label]
            * self.Y_var[f_k.id, f_i.id]
            for f_k in K for f_i in P
        )

        # Cost: serving each customer (origin to destination)
        cost_serving = gp.quicksum(
            self.costs[f_i.origin.label][f_i.destination.label] * self.Z_var[f_i.id]
            for f_i in P
        )

        # Cost: empty driving between consecutive customers
        cost_between = gp.quicksum(
            self.costs[f_i.destination.label][f_j.origin.label] * self.X_var[f_i.id, f_j.id]
            for f_i in P for f_j in P if f_i != f_j
        )

        profit = revenue - cost_vehicle_to_first - cost_serving - cost_between
        self.model.setObjective(profit, sense=GRB.MAXIMIZE)

    def define_total_wait_time_objective(self, P):
        """
        Define objective of minimizing the total wait time and add it to the model.

        Input:
        ------------
        P: List[Any]
            Set of customers to serve.

        Hint:
        ------------
        - Durations are given in seconds. Convert waiting time to minutes by dividing by 60.

        """

        # For served customers: wait = (u_i - ready_time_i) / 60
        # For unserved customers: wait = (latest_pickup_i - ready_time_i) / 60
        total_wait = gp.quicksum(
            (self.U_var[f_i.id] - f_i.ready_time) / 60.0
            + (1 - self.Z_var[f_i.id]) * (f_i.latest_pickup - f_i.ready_time) / 60.0
            for f_i in P
        )
        self.model.setObjective(total_wait, sense=GRB.MINIMIZE)

    def define_multi_objective(self, K, P, vehicle_request_assign):
        """
        Define weighted combined objective: maximize total profit and minimize wait time.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.

        Hint:
        ------------
        - Access costs via self.costs[from_location][to_location]
        - Access vehicle departure stop via vehicle_request_assign[vehicle_id].departure_stop
        - Uses single weight w: objective = w * (total profit) - (1 - w) * (total wait time).
        - Durations are given in seconds. Convert waiting time to minutes by dividing by 60.

        """

        w = self.weight

        # Profit component (same as define_total_profit_objective)
        revenue = gp.quicksum(f_i.fare * self.Z_var[f_i.id] for f_i in P)
        cost_vehicle_to_first = gp.quicksum(
            self.costs[vehicle_request_assign[f_k.id].departure_stop][f_i.origin.label]
            * self.Y_var[f_k.id, f_i.id]
            for f_k in K for f_i in P
        )
        cost_serving = gp.quicksum(
            self.costs[f_i.origin.label][f_i.destination.label] * self.Z_var[f_i.id]
            for f_i in P
        )
        cost_between = gp.quicksum(
            self.costs[f_i.destination.label][f_j.origin.label] * self.X_var[f_i.id, f_j.id]
            for f_i in P for f_j in P if f_i != f_j
        )
        profit = revenue - cost_vehicle_to_first - cost_serving - cost_between

        # Wait time component (same as define_total_wait_time_objective)
        total_wait = gp.quicksum(
            (self.U_var[f_i.id] - f_i.ready_time) / 60.0
            + (1 - self.Z_var[f_i.id]) * (f_i.latest_pickup - f_i.ready_time) / 60.0
            for f_i in P
        )

        # Z = maximize w * Profit - (1 - w) * Wait_time
        self.model.setObjective(w * profit - (1 - w) * total_wait, sense=GRB.MAXIMIZE)


    def create_model(self, K, P, vehicle_request_assign):
        """
        Create model to solve with Gurobi Solver.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.
        """

        for f_i in P:
            self.U_var[f_i.id] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f'U_{f_i.id}')
            self.Z_var[f_i.id] = self.model.addVar(vtype=GRB.BINARY, obj=0, name=f'Z_{f_i.id}')
            for f_j in P:
                if f_i != f_j:
                    self.X_var[f_i.id, f_j.id] = self.model.addVar(vtype=GRB.BINARY, name=f'X_{f_i.id}_{f_j.id}')

        for f_i in P:
            for f_k in K:
                self.Y_var[f_k.id, f_i.id] = self.model.addVar(vtype=GRB.BINARY, name=f'Y_{f_k.id}_{f_i.id}')

        # Update the model to include the new variables
        self.model.update()

        """Set up constraints"""
        # Constraints 1
        for f_i in P:
            self.model.addConstr(
                self.Z_var[f_i.id] == sum(self.Y_var[f_k.id, f_i.id] for f_k in K)
                + sum(self.X_var[f_j.id, f_i.id] for f_j in P if f_i != f_j),
                name=f"Constraint1_{f_i.id}"
            )

        # Constraints 2
        for f_i in P:
            self.model.addConstr(
                self.Z_var[f_i.id] >= sum(self.X_var[f_i.id, f_j.id] for f_j in P if f_i != f_j),
                name=f"Constraint2_{f_i.id}"
            )

        # Constraints 3
        for f_k in K:
            self.model.addConstr(
                sum(self.Y_var[f_k.id, f_i.id] for f_i in P) <= 1,
                name=f"Constraint3_{f_k.id}"
            )

        # Constraints 4
        for f_i in P:
            self.model.addConstr(self.U_var[f_i.id] >= f_i.ready_time, name=f"Constraint4a_{f_i.id}")
            self.model.addConstr(self.U_var[f_i.id] <= f_i.latest_pickup, name=f"Constraint4b_{f_i.id}")

        # Constraints 5
        for f_i in P:
            for f_j in P:
                if f_i != f_j:
                    T_ij = f_i.shortest_travel_time + self.durations[f_i.destination.label][f_j.origin.label]
                    delta = f_j.ready_time - f_i.latest_pickup
                    self.model.addConstr(
                        self.U_var[f_j.id] - self.U_var[f_i.id] >= delta + self.X_var[f_i.id, f_j.id] * (T_ij - delta),
                        name=f"Constraint5_{f_i.id}_{f_j.id}"
                    )

        # Constraints 6
        for f_i in P:
            for f_k in K:
                state = vehicle_request_assign[f_k.id]
                T_ki = self.durations[state.departure_stop][f_i.origin.label]
                delta = state.departure_time + T_ki - f_i.ready_time
                self.model.addConstr(
                    self.U_var[f_i.id] >= f_i.ready_time + delta * self.Y_var[f_k.id, f_i.id],
                    name=f"Constraint6_{f_i.id}_{f_k.id}"
                )
        self.model.update()

    def solve(self):
        """
        Optimize the model using Gurobi.

        Note:
        ------------
        This method has no input parameters. It optimizes the model stored in self.model
        and updates self.objective_value with the result.
        """
        self.model.optimize()

        # Check if the optimization was successful
        if self.model.status != GRB.OPTIMAL:
            print("Optimization did not converge to an optimal solution.")
            return
        self.objective_value = round(self.model.objVal, 3)

    def extract_solution(self, K, P, rejected_trips, vehicle_request_assign):
        """
        Extract the solution from the optimized model and convert it to vehicle_request_assign format.

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        rejected_trips: List[Any]
            List of trips that are rejected in the optimization process (will be populated by this method).
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects (will be populated with assignments by this method).

        Note:
        ------------
        - Adds trips to vehicle_request_assign[vehicle_id].assigned_requests
        - Adds unserved trips (Z_var < 0.5) to rejected_trips
        """

        # Extract the solution and populate the vehicle_request_assign and rejected_trips
        for f_k in K:
            for trip in P:
                if self.Y_var[f_k.id, trip.id].X > 0.5:
                    state = vehicle_request_assign[f_k.id]
                    state.assigned_requests.append(trip)
                    current_trip = trip
                    while True:
                        next_trip_found = False
                        for f_j in P:
                            if current_trip != f_j and self.X_var[current_trip.id, f_j.id].X > 0.5:
                                state.assigned_requests.append(f_j)
                                current_trip = f_j
                                next_trip_found = True
                                break
                        if not next_trip_found:
                            break

        for trip in P:
            if self.Z_var[trip.id].X < 0.5:
                rejected_trips.append(trip)

    def offline_solver(self, K, P, vehicle_request_assign, rejected_trips):
        """
        Solve the taxi routing problem using a MIP solver (Gurobi).

        Input:
        ------------
        K: List[Any]
            Set of vehicles.
        P: List[Any]
            Set of customers to serve.
        vehicle_request_assign: Dict[int, VehicleState]
            Dictionary mapping vehicle IDs to VehicleState objects containing vehicle-request assignments.
        rejected_trips: List[Any]
            List of trips that are rejected in the optimization process (will be populated by this method).
        """
        self.create_model(K, P, vehicle_request_assign)
        self.define_objective(K, P, vehicle_request_assign)
        self.solve()
        self.extract_solution(K, P, rejected_trips, vehicle_request_assign)
