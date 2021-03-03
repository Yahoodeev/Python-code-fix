from mip import BINARY, CONTINUOUS, Model, maximize, Constr, Var, OptimizationStatus, LP_Method, SearchEmphasis
from mip import xsum as Σ 
from collections import defaultdict
from typing import Set, Dict, List, Tuple, Hashable, Union
import json
import itertools
import click
import datetime
import time
import pprint

def process(d, p):
    """Processing the data (d) and parameters (p)"""

    Rounds = list(range(1, p["Rounds"] + 1))
    Positions = set(p["Scoring Positions"]) | set(p["Substitute Positions"])
    return {
        "Players": set(d.keys()),
        "Positions": Positions,
        "Scoring Positions": set(p["Scoring Positions"]),
        "Substitute Positions": set(p["Substitute Positions"]),
        "Players Eligible For Position q In Round r": {
            (q_, r): set(Player for Player, x in d.items() if q in x["Position"][str(r)])
            for q in p["Scoring Positions"] for q_ in [q, "SUB " + q] for r in Rounds
        },
        "Positions Eligible For Player p In Round r": {
            (Player, r): set(x["Position"][str(r)]) | set(["SUB " + y for y in x["Position"][str(r)]])
            for Player, x in d.items() for r in Rounds
        },
        "Rounds": Rounds,
        "Season Trade Limit": p["Trades"]["Total"],
        "Trade Limit In Round r": defaultdict(lambda: p["Trades"]["Default"], {
            int(key): int(value) for key, value in p["Trades"]["Exceptions"].items()
        }),
        "Players Required In Position q": p["Capacities"],
        "Scoring Positions In Round r": defaultdict(lambda: p["Number Of Scoring Positions"]["Default"], {
            int(key): int(value) for key, value in p["Number Of Scoring Positions"]["Exceptions"].items()
        }),
        "Initial Salary Cap": p["Initial Salary Cap"],
        "Score Of Player p In Round r": {
            (Player, r): x["Score"][str(r)]
            for Player, x in d.items() for r in Rounds
        },
        "Price Of Player p In Round r": {
            (Player, r): x["Price"][str(r)]
            for Player, x in d.items() for r in Rounds
        }
    }

def binary(m):
    return m.add_var(var_type=BINARY)

def continuous(m):
    return m.add_var(var_type=CONTINUOUS)

def declare_constraints(model, constraints):
    for name, constraint_set in constraints.items():
        for nb, constraint in enumerate(constraint_set):
            model.add_constr(constraint)
    return constraints

def model(Data: Dict):
    m = Model()
    
    # Notation
    P: Set[str]                        = Data["Players"]
    Q: Set[str]                        = Data["Positions"]
    P_: Dict[str, int, Set[str]]       = Data["Players Eligible For Position q In Round r"]
    Q_: Dict[str, int, Set[str]]       = Data["Positions Eligible For Player p In Round r"]
    Q_score: Set[str]                  = Data["Scoring Positions"]
    Q_sub: Set[str]                    = Data["Substitute Positions"]
    R: List[int]                       = Data["Rounds"]
    T: int                             = Data["Season Trade Limit"]
    T_: Dict[int, int]                 = Data["Trade Limit In Round r"]
    C_: Dict[str, int]                 = Data["Players Required In Position q"]
    X_: Dict[int, int]                 = Data["Scoring Positions In Round r"]
    B: int                             = Data["Initial Salary Cap"]
    Ψ_: Dict[Tuple[str, int], int]     = Data["Score Of Player p In Round r"]
    v_: Dict[Tuple[str, int], int]     = Data["Price Of Player p In Round r"]

    # Variables
    m.variables: Dict[str, Dict[Tuple, Var]] = {
        # 1 if player p ∈ P is in position q ∈ Q_p in round r ∈ R.
        "In Team":    (x := {(p, q, r): m.add_var(var_type=BINARY) for p in P for r in R for q in Q_[p, r]}),
        # 1 if the points of player p ∈ P in round r ∈ R count to the score.
        "Scoring":    (x_bar := {(p, r): binary(m) for p in P for r in R}),
        # 1 if player p ∈ P is Captain in round r ∈ R.
        "Captain":    (c := {(p, r): binary(m) for p in P for r in R}),
        # 1 if player p ∈ P is traded into the team in round r ∈ R.
        "Trade In":   (t_in := {(p, r): binary(m) for p in P for r in R}),
        # 1 if player p ∈ P is traded out of the team in round r ∈ R.
        "Trade Out":  (t_out := {(p, r): binary(m) for p in P for r in R}),
        # Remaining Salary Cap available in round r ∈ R.
        "Salary Cap": (b := {r: continuous(m) for r in R})
    }

    # Objective
    m.objective = maximize(Σ(Ψ_[p, r] * (x_bar[p, r] + c[p, r]) for p in P for r in R))

    # Constraints 
    m.constraints: Dict[Hashable, List[Constr]] = declare_constraints(m, {
        # The number of trades across the season is less than or equal T.
        (1):   [Σ(t_in[p, r] for p in P for r in R[1:]) <= T],    
        # The number of trades per round is less than or equal to T_r.
        (2):   [Σ(t_in[p, r] for p in P) <= T_[r] for r in R],   
        # Links player trade variables with whether they are in the team 
        (3):   [Σ(x[p, q, r] - x[p, q, r-1] for q in Q_[p, r]) <= t_in[p, r] - t_out[p, r] for p in P for r in R[1:]],    
        # Exactly one captain per round.
        (4):   [Σ(c[p, r] for p in P) == 1 for r in R],    
        # Captain must be a member of the team.
        (5):   [c[p, r] <= Σ(x[p, q, r] for q in Q_[p, r]) for r in R for p in P], 
        # Each position must contain a exact number of players.
        (6):   [Σ(x[p, q, r] for p in P_[q, r]) == C_[q] for r in R for q in Q],
        # Each player can be in at most one position
        (7):   [Σ(x[p, q, r] for q in Q_[p, r]) <= 1 for r in R for p in P],
        # Player must be in scoring position to score
        (8):   [x_bar[p, r] <= Σ(x[p, q, r] for q in (Q_[p, r] & Q_score)) for r in R for p in P],
        # The number of scoring players are limited each round
        (9):   [Σ(x_bar[p, r] for p in P) <= X_[r] for r in R],
        # Remaining budget after selection of initial team
        (10):  [b[1] + Σ(v_[p, 1]*x[p, q, 1] for p in P for q in Q_[p, r]) == B for r in R],
        # Budget consistency constraints links budget and trades
        (11):  [b[r] == b[r-1] +  Σ(v_[p, r]*(t_out[p, r]-t_in[p, r]) for p in P) for r in R[1:]]
        }
    )
    return m

def saveasjson(m, Data, output_file):

    P = Data['Players']
    Q = Data["Positions"]
    ψ_ = Data["Score Of Player p In Round r"]
    v_ = Data["Price Of Player p In Round r"]
    R = Data["Rounds"]

    Results = {
        "Round": (round_results := {r: {
            "Team": (players_in_team := {
                q: [f"{p: <22}, ${v_[p, r]}, {ψ_[p, r]}" for p in Data["Players Eligible For Position q In Round r"][q, r]
                    if m.variables["In Team"][p, q, r].x > 0.5] for q in Q
            }),
            "Captain": (skip := [p for p in P if m.variables["Captain"][(p, r)].x > 0.5][0]) + f" [{ψ_[skip, r]}]",
            "Players Traded In": [f"{p: <22}, ${v_[p, r]}" for p in P if m.variables["Trade In"][p, r].x > 0.5],
            "Players Traded Out": [f"{p: <22}, ${v_[p, r]}" for p in P if m.variables["Trade Out"][p, r].x > 0.5],
            "Remaining Salary Cap": m.variables["Salary Cap"][r].x,
            "Team Value": sum(v_[p.split(",")[0].strip(), r] for temp in players_in_team.values() for p in temp),
            "Score": sum(ψ_[p, r] for p in P if m.variables["Scoring"][p, r].x > 0.5) + ψ_[skip, r]
        }
            for r in R
        }
                  ),
        "Season Summary": {
            "Total Season Points": sum(round_results[r]["Score"] for r in R),
            "Total Trades Used": sum(len(round_results[r]["Players Traded In"]) for r in R)
        }
    }

    pprint.pprint(Results)
    with open(output_file, 'w') as f_out:
        json.dump(Results, f_out, indent=2)

with open(r"2020 SC Parameters.json") as f:
    parameters = json.load(f)
with open(r"2020 SC Data.json") as f:
    playerdata = json.load(f)

Data = process(playerdata, parameters)
m = model(Data)

m.write(r"2020 SC.lp")
m.read(r"2020 SC.lp")

status = m.optimize(max_seconds=float("inf"))
if status == OptimizationStatus.OPTIMAL:
    print('Optimal Solution Found: {}'.format(m.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('Non-Optimal Solution Found: {}'.format(m.objective_value))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('No Solution Found')
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    print('Solution:')
    saveasjson(m, Data, r"2020 SC Solution.json")
else:
    raise ValueError("Incorrect Solve Type")