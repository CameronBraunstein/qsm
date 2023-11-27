from utils.quantum.get_qubo_codec import get_qubo_codec
from utils.quantum.gurobi_helper import get_gurobi_helper
from utils.quantum.graph_embedder import get_graph_embedder
from stereo_matchers.graph_cut_solver import get_graph_cut_solver
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite, LeapHybridSampler
import dimod


def get_qubo_solver_interface(args):
    ai = QUBOSolverInterface(args)
    return ai

class QUBOSolverInterface:
    def __init__(self,args):
        self.qubo_codec = get_qubo_codec(args)
        self.gurobi_helper = get_gurobi_helper(args)
        self.graph_embedder = get_graph_embedder(args)
        self.graph_cut_solver = get_graph_cut_solver(args)
        self.qubo_solver = args.qubo_solver
        self.annealing_runs = args.annealing_runs
        if (self.qubo_solver == 'dwave_qpu') or (self.qubo_solver == 'dwave_hybrid'):
            print('WARNING: Your QUBO solver will use DWave resources. This will drain your subscription')


    def initialize_model_at_step(self,step):
        self.qubo_codec.initialize_model_at_step(step)
        self.graph_embedder.initialize_model_at_step(step)
        
    def compute_displacements_for_bundle(self,bundle):
        if self.qubo_solver == 'gurobi':
            Q = self._create_QUBO_for_bundle(bundle)
            response = self.gurobi_helper.solve_qubo(Q)
            bundle_displacement = self.qubo_codec.create_displacements_from_response(response)
        elif self.qubo_solver == 'dwave_hybrid':
            Q = self._create_QUBO_for_bundle(bundle)
            sampler = LeapHybridSampler(solver={'category': 'hybrid'})
            bqm= dimod.binary.BinaryQuadraticModel.from_qubo(Q)
            sampleset = sampler.sample(bqm)
            response = sampleset.first.sample
            bundle_displacement = self.qubo_codec.create_displacements_from_response(response)
        elif self.qubo_solver == 'dwave_qpu':
            Q = self._create_QUBO_for_bundle(bundle)
            embedding = self.graph_embedder.get_embedding_for_bundle(Q,bundle)
            sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
            sampleset = sampler.sample_qubo(Q, num_reads=self.annealing_runs)
            response = sampleset.first.sample
            bundle_displacement = self.qubo_codec.create_displacements_from_response(response)
        elif self.qubo_solver == 'simulated_annealing':
            Q = self._create_QUBO_for_bundle(bundle)
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample_qubo(Q, num_reads=self.annealing_runs).first.sample
            bundle_displacement = self.qubo_codec.create_displacements_from_response(response)
        elif self.qubo_solver == 'graph_cut':
            bundle_displacement = self.graph_cut_solver.create_displacements(bundle)
        return bundle_displacement
    
    def _create_QUBO_for_bundle(self,bundle):
        height_range,width_range = bundle
        height_offset = height_range[0]
        width_offset = width_range[0]
        self.qubo_codec.set_bundle(bundle)

        Q = {}
        
        for base_y in height_range:
            for base_x in width_range:
                #Data Term

                self.qubo_codec.add_data_costs(Q,base_y,base_x)

                #Regularization Term
                if base_y > height_offset:
                    neighbor_y = base_y - 1
                    neighbor_x = base_x
                    self.qubo_codec.add_regularization_costs(Q,base_y,base_x,neighbor_y,neighbor_x)

                if base_x > width_offset:
                    neighbor_y = base_y
                    neighbor_x = base_x - 1
                    self.qubo_codec.add_regularization_costs(Q,base_y,base_x,neighbor_y,neighbor_x)
        return Q