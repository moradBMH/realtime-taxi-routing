import random
from typing import Any, List
import logging

from src.utilities.config import SimulationConfig
from src.utilities.enums import Algorithm, Objectives
from src.solvers.solver import Solver

logger = logging.getLogger(__name__)


class OnlineSolver(Solver):
    """Provide online solution to optimize the vehicle routing and the trip-route assignment. This
    method includes:
        1. greedy solver
        2. random solver
        3. ranking solver

    Attributes:
    ------------
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

        X: Dict[int, Dict[int, bool]]
            Binary variables indicating if customer i is picked immediately after j by a taxi.
        Y: Dict[int, Dict[int, bool]]
            Binary variables indicating if customer i is picked up by vehicle k as the first customer.
        U: Dict[int, float]
            Pickup times for customers.
        Z: Dict[int, bool]
            Binary variables indicating if customer i is selected to be served.
    """

    def __init__(self,
                 network: Any,
                 vehicles: List[Any],
                 simulation_config: SimulationConfig):
        super().__init__(network, vehicles, simulation_config)
        # Assign random numbers to each vehicle request assignment
        for veh_id, state in self.vehicle_request_assign.items():
            state.random_number = random.random()

    def determine_available_vehicles(self, trip):
        """Determine whether each vehicle can feasibly serve the trip (e.g. reach pickup by latest_pickup).

        Input:
        ------------
            trip : ride request to serve

        Hint:
            - Iterate over self.vehicle_request_assign and set assign_possible for each vehicle.
            - Use the parent class method calc_reach_time(veh_info, trip).
        """
        for veh_id, state in self.vehicle_request_assign.items():
            # A vehicle can serve the trip if its reach time to the origin is
            # before the trip's latest_pickup deadline.
            reach_time = self.calc_reach_time(state, trip)
            state.assign_possible = reach_time <= trip.latest_pickup


    def _select_greedy_vehicle(self, trip):
        """Return the feasible vehicle best matching the current objective, or None."""
        candidates = [s for s in self.vehicle_request_assign.values() if s.assign_possible]
        if not candidates:
            return None

        if self.objective == Objectives.WAIT_TIME:
            # Minimize pickup time (reduce wait)
            return min(candidates, key=lambda s: self.calc_reach_time(s, trip))
        if self.objective == Objectives.TOTAL_PROFIT:
            # Minimize empty-driving cost to pickup (maximizes marginal profit)
            return min(candidates, key=lambda s: self.costs[s.last_stop][trip.origin.label])
        # TOTAL_CUSTOMERS (and others): prefer earliest reach to keep capacity for others
        return min(candidates, key=lambda s: self.calc_reach_time(s, trip))

    def online_solver(self, K, P_not_assigned, rejected_trips):
        """Find a solution to assign ride requests to vehicles after arrival.

        Input:
        ------------
            K : set of vehicles
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Steps:
            1. Assign requests to vehicles/routes based on the chosen algorithm.
            2. Check the feasibility of the solution.
        """

        # Step 1: assign requests to the vehicles/ routes
        sorted_requests = sorted(P_not_assigned, key=lambda x: x.ready_time)

        if self.algorithm == Algorithm.GREEDY:
            assigned_requests = self.greedy_assign(sorted_requests, rejected_trips)
        elif self.algorithm == Algorithm.RANDOM:
            assigned_requests = self.random_assign(sorted_requests, rejected_trips)
        elif self.algorithm == Algorithm.RANKING:
            assigned_requests = self.ranking_assign(sorted_requests, rejected_trips)
        else:
            logger.error("Unsupported algorithm: %s", self.algorithm)
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # Step 2: check the feasibility of then solution
        self.create_online_solution()
        if self.verify_constraints(K, assigned_requests):
            self.calc_objective_value(K, sorted_requests)
            self.total_customers_served = sum(1 for f_i in P_not_assigned if self.Z[f_i.id])
            logger.info("Online solver solution is feasible. Served %d customers.", self.total_customers_served)

        else:
            logger.error("The solution is not feasible.")
            raise ValueError("The solution is not feasible")

    def greedy_assign(self, P_not_assigned: List[Any], rejected_trips: List[Any]) -> List[Any]:
        """Assign ride requests to vehicles using a greedy method (e.g. best objective per request).

        Input:
        ------------
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Output:
        ------------
            assigned_requests : list of assigned requests

        Hint:
            - for each trip in P_not_assigned you have to select a vehicle to assign or reject the request.
            - evaluating the feasibility of assigning a trip to a vehicle should be done inside "determine_available_vehicles" function.
            - If no vehicle is available, append the trip to rejected_trips.
            - if a vehicle is selected to assign a request:
                - Use the assign_trip_to_vehicle function to assign the task to the selected vehicle
                - add trip to the list of assigned_requests
        """
        # for each request find the best insertion position
        assigned_requests = []
        for trip in P_not_assigned:
            self.determine_available_vehicles(trip)
            selected = self._select_greedy_vehicle(trip)
            if selected is None:
                rejected_trips.append(trip)
            else:
                self.assign_trip_to_vehicle(selected, trip)
                assigned_requests.append(trip)
        return assigned_requests

    def random_assign(self, P_not_assigned: List[Any], rejected_trips: List[Any]) -> List[Any]:
        """Assign ride requests to vehicles based on random solution method

        Input:
        ------------
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Output:
        ------------
            assigned_requests : list of assigned requests

        Hint:
            - for each trip in P_not_assigned you have to select a vehicle to assign or reject the request.
            - evaluating the feasibility of assigning a trip to a vehicle should be done inside "determine_available_vehicles" function.
            - If no vehicle is available, append the trip to rejected_trips.
            - if a vehicle is selected to assign a request:
                - Use the assign_trip_to_vehicle function to assign the task to the selected vehicle
                - add trip the list of assigned_requests
        """
        # for each request find the best insertion position
        assigned_requests = []
        for trip in P_not_assigned:
            self.determine_available_vehicles(trip)
            candidates = [s for s in self.vehicle_request_assign.values() if s.assign_possible]
            if not candidates:
                rejected_trips.append(trip)
            else:
                selected = random.choice(candidates)
                self.assign_trip_to_vehicle(selected, trip)
                assigned_requests.append(trip)
        return assigned_requests

    def ranking_assign(self, P_not_assigned: List[Any], rejected_trips: List[Any]) -> List[Any]:
        """Assign ride requests to vehicles by ranking solution method.

        Input:
        ------------
            P_not_assigned : set of customers that are not assigned to be served
            rejected_trips : list of trips rejected in the optimization process.

        Output:
        ------------
            assigned_requests : list of assigned requests

        Hint:
            - for each trip in P_not_assigned you have to select a vehicle to assign or reject the request.
            - evaluating the feasibility of assigning a trip to a vehicle should be done inside "determine_available_vehicles" function.
            - If no vehicle is available, append the trip to rejected_trips.
            - if a vehicle is selected to assign a request:
                - Use the assign_trip_to_vehicle function to assign the task to the selected vehicle
                - add trip the list of assigned_requests

        """
        # for each request find the best insertion position
        assigned_requests = []
        for trip in P_not_assigned:
            self.determine_available_vehicles(trip)
            candidates = [s for s in self.vehicle_request_assign.values() if s.assign_possible]
            if not candidates:
                rejected_trips.append(trip)
            else:
                # Classic RANKING: pre-assigned random priority per vehicle;
                # pick the feasible vehicle with the smallest priority value.
                selected = min(candidates, key=lambda s: s.random_number)
                self.assign_trip_to_vehicle(selected, trip)
                assigned_requests.append(trip)
        return assigned_requests
