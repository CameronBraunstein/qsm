import gurobipy as gp
from gurobipy import GRB
from collections import OrderedDict
from utils.quantum.get_qubo_codec import get_qubo_codec

def get_gurobi_helper(args):
    gh = GUROBIHelper(args)
    return gh

class GUROBIHelper:
    def __init__(self,args):
        self.qubo_codec = get_qubo_codec(args)
        return 
    
    def solve_qubo(self,Q):
        var_names = self._get_ordered_list_of_vars(Q)
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model("QUBO", env=env) as model:
                x = model.addVars(len(var_names), vtype=GRB.BINARY, name="x")
                obj_expr = self._get_objective_exp(Q,x,var_names)
                model.setObjective(obj_expr, GRB.MINIMIZE)
                model.optimize()
                response = self._create_response(x,var_names)
        return response
    
    def _create_response(self,x,var_names):
        response = {}
        for i in range(len(x)):
            response[var_names[i]] = x[i].x
        return response
    
    def _get_objective_exp(self,Q,x,var_names):
        obj_expr = gp.QuadExpr()
        for i in range(len(x)):
            var_0 = var_names[i]
            for j in range(i, len(x)):
                var_1 = var_names[j]
                matrix_value = self.qubo_codec.get_matrix_entry(Q,var_0,var_1)
                if matrix_value != 0:
                    obj_expr += matrix_value * x[i] * x[j]
        return obj_expr

    def _get_ordered_list_of_vars(self,Q):
        ordered_set = OrderedDict.fromkeys([])
        for key in Q.keys():
            v1,v2 = key
            ordered_set[v1] = None
            ordered_set[v2] = None
        ordered_set = list(ordered_set.keys())
        return ordered_set