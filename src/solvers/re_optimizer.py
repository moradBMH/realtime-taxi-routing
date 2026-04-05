from src.solvers.offline_solver import OfflineSolver
from src.utilities.enums import DestroyMethod
from src.solvers.solver import Solver
from src.utilities.config import SimulationConfig
from typing import Any, Dict, List



class ReOptimizer(Solver):
    """Re-optimize vehicle routing and trip-route assignment using destroy-and-repair (LNS) ideas.

    Includes three destroy methods:
        1. destroy_fix_arrival_times
        2. destroy_fix_variables
        3. destroy_bonus

    Attributes:
    ------------
        destroy_method : DestroyMethod(Enum)
            Method used for destruction in the LNS algorithm.
        initial_solution : dict
            Initial values for decision variables (X, Y, Z, U, assignment_dict).

        vehicle_request_assign : Dict[int, VehicleState]
            Mapping vehicle id to VehicleState (inherited from Solver). Each state holds: vehicle,
            assigned_requests, departure_stop, departure_time, last_stop, last_stop_time, assign_possible,
            random_number; used to save assignments and build route plans.

        durations : dictionary
            travel time matrix between possible stop points
            example: for duration between destination of trip_i and the origin of trip_j use:
                     self.durations[trip_i.destination.label][trip_i.origin.label]

        costs: dictionary
            driving costs (it works based on location ids like durations)
        algorithm: Algorithm(Enum)
            The optimization algorithm utilized for planning and assigning trips to vehicles.
        objective: Objectives(Enum)
            The objective used to evaluate the effectiveness of the plan (e.g., maximizing profit or minimizing wait time).
        objective_value: float
            The objective value from served requests.
        total_customers_served: int
            The count of customers successfully served.
    """

    def __init__(self,
                 network: Any,
                 vehicles: List[Any],
                 simulation_config: SimulationConfig) -> None:
        super().__init__(network, vehicles, simulation_config)
        self.initial_solution : Dict[str, Any] = {}
        self.destroy_method = simulation_config.algorithm_params["destroy_method"]


    def re_optimizer(self, K, P_not_served, rejected_trips):
        """Re-optimize the solution using destroy and repair (LNS).

        Input:
        ------------
            K : set of vehicles
            P_not_served : set of customers not yet served
            rejected_trips : list of trips rejected in the optimization process.

        Steps:
            1. Create the mathematical model (OfflineSolver).
            2. If initial_solution exists, destroy (fix) part of it according to destroy_method.
            3. Repair (re-optimize) with the MIP solver and extract solution.
            4. Save the final solution to initial_solution.
        """

        # Create and configure the offline model
        offline_model = OfflineSolver(self.network, self.objective)
        offline_model.create_model(K, P_not_served, self.vehicle_request_assign)

        if self.initial_solution:
            if self.destroy_method == DestroyMethod.FIX_ARRIVALS:
                # destroy by fixing a time window around the arrival times in the initial solution
                self.destroy_fix_arrival_times(P_not_served, offline_model)

            elif self.destroy_method == DestroyMethod.FIX_VARIABLES:
                # destroy by fixing some of the variables based on the initial solution
                self.destroy_fix_variables(K, P_not_served, offline_model)
            elif self.destroy_method == DestroyMethod.BONUS:
                # destroy the solution by your suggested function
                self.destroy_bonus(K, P_not_served, offline_model)

        # add objective
        offline_model.define_objective(K, P_not_served, self.vehicle_request_assign)

        # solve and get solution
        offline_model.solve()
        offline_model.extract_solution(K, P_not_served, rejected_trips, self.vehicle_request_assign)

        self.save_solution(offline_model)

    def save_solution(self, offline_model: OfflineSolver):
        """Save the solution from the offline model into initial_solution.

        Input:
        ------------
            offline_model : OfflineSolver instance (Gurobi MIP model).
        """
        # Extracting values of decision variables
        self.Y: Dict[Any, Dict[Any, float]] = {
            key[0]: {sub_key[1]: var.X for sub_key, var in offline_model.Y_var.items() if sub_key[0] == key[0]}
            for key in offline_model.Y_var.keys()
        }
        self.X: Dict[Any, Dict[Any, float]] = {
            key[0]: {sub_key[1]: var.X for sub_key, var in offline_model.X_var.items() if sub_key[0] == key[0]}
            for key in offline_model.X_var.keys()
        }
        self.Z: Dict[Any, float] = {key: var.X for key, var in offline_model.Z_var.items()}
        self.U: Dict[Any, float] = {key: var.X for key, var in offline_model.U_var.items()}

        assignment_dict: Dict[Any, Dict[str, List[Any]]] = {
            vehicle_id: {
                'assigned_requests': state.assigned_requests
            }
            for vehicle_id, state in self.vehicle_request_assign.items()
        }

        self.initial_solution = {
            'X': self.X,
            'Y': self.Y,
            'U': self.U,
            'Z': self.Z,
            'assignment_dict': assignment_dict
        }

    def destroy_fix_arrival_times(self, P, offline_model: OfflineSolver) -> None:
        """Fix pickup times to a time window around values in initial_solution (e.g. ±2 min).

        Input:
        ------------
            P : set of customers to serve
            offline_model : OfflineSolver instance (Gurobi MIP model).

        """
        # Tighten the pickup time window for requests that were served in the initial solution
        # by restricting U_var[i] to [U_initial[i] - 2min, U_initial[i] + 2min],
        # intersected with the original [ready_time, latest_pickup] bounds.
        initial_U = self.initial_solution['U']
        initial_Z = self.initial_solution['Z']
        time_delta = 2 * 60  # 2 minutes in seconds

        for f_i in P:
            if f_i.id in initial_Z and initial_Z[f_i.id] > 0.5:
                # Request was served in initial solution. Fix U around its initial pickup time
                u_init = initial_U[f_i.id]
                new_lb = max(f_i.ready_time, u_init - time_delta)
                new_ub = min(f_i.latest_pickup, u_init + time_delta)
                offline_model.U_var[f_i.id].lb = new_lb
                offline_model.U_var[f_i.id].ub = new_ub

    def destroy_fix_variables(self, K, P, offline_model: OfflineSolver):
        """ Fix some of Y_var, X_var variables based on the initial solution.

        Input:
        ------------
            K : set of vehicles
            P : set of customers to serve
            offline_model : OfflineSolver instance (Gurobi model).

        Hint:
            - Forbid the arcs that goes from one request to another one that were in different vehicle
            - Forbid the arcs that goes from departing node of a vehicle to other requests that were in different
                  vehicle
        """
        assignment_dict = self.initial_solution['assignment_dict']

        # Build a mapping. request_id -> vehicle_id for all requests assigned in the initial solution
        request_to_vehicle = {}
        for vehicle_id, data in assignment_dict.items():
            for request in data['assigned_requests']:
                request_to_vehicle[request.id] = vehicle_id

        # Forbid X_var[i,j] = 0 when requests i and j were assigned to different vehicles
        for f_i in P:
            for f_j in P:
                if f_i != f_j:
                    vi = request_to_vehicle.get(f_i.id)
                    vj = request_to_vehicle.get(f_j.id)
                    # If both were assigned but to different vehicles forbid the arc
                    if vi is not None and vj is not None and vi != vj:
                        offline_model.X_var[f_i.id, f_j.id].ub = 0

        # Forbid Y_var[k,i] = 0 when request i was assigned to a different vehicle in the initial solution
        for f_k in K:
            for f_i in P:
                vi = request_to_vehicle.get(f_i.id)
                # If request i was assigned to a vehicle different from k forbid this assignment
                if vi is not None and vi != f_k.id:
                    offline_model.Y_var[f_k.id, f_i.id].ub = 0


    def destroy_bonus(self, K, P, offline_model: OfflineSolver):
        """Custom destroy method: Profit-based selective destruction.

        Destroys the assignments of the least profitable vehicles (bottom 50%) while
        preserving the assignments of the most profitable ones. This targets the weakest
        parts of the solution for improvement and keeps high-quality assignments intact.

        Input:
        ------------
            K : set of vehicles
            P : set of customers to serve
            offline_model : OfflineSolver instance (Gurobi model).
        """
        assignment_dict = self.initial_solution['assignment_dict']

        # Compute profit per vehicle from the initial solution
        vehicle_profits = {}
        for vehicle_id, data in assignment_dict.items():
            profit = 0
            requests = data['assigned_requests']
            if requests:
                # Cost. Vehicle to first customer
                state = self.vehicle_request_assign[vehicle_id]
                profit -= self.costs[state.departure_stop][requests[0].origin.label]
                for idx, req in enumerate(requests):
                    profit += req.fare
                    profit -= self.costs[req.origin.label][req.destination.label]
                    if idx + 1 < len(requests):
                        profit -= self.costs[req.destination.label][requests[idx + 1].origin.label]
            vehicle_profits[vehicle_id] = profit

        # Sort vehicles by profit and identify bottom 50% to destroy
        sorted_vehicles = sorted(vehicle_profits.items(), key=lambda x: x[1])
        num_to_destroy = max(1, len(sorted_vehicles) // 2)
        vehicles_to_destroy = {v_id for v_id, _ in sorted_vehicles[:num_to_destroy]}

        # Build request->vehicle mapping for preserved vehicles only
        preserved_requests = {}
        for vehicle_id, data in assignment_dict.items():
            if vehicle_id not in vehicles_to_destroy:
                for request in data['assigned_requests']:
                    preserved_requests[request.id] = vehicle_id

        # Fix variables for preserved vehicles. Forbid cross-vehicle arcs
        for f_i in P:
            for f_j in P:
                if f_i != f_j:
                    vi = preserved_requests.get(f_i.id)
                    vj = preserved_requests.get(f_j.id)
                    if vi is not None and vj is not None and vi != vj:
                        offline_model.X_var[f_i.id, f_j.id].ub = 0

        for f_k in K:
            for f_i in P:
                vi = preserved_requests.get(f_i.id)
                if vi is not None and vi != f_k.id:
                    offline_model.Y_var[f_k.id, f_i.id].ub = 0

