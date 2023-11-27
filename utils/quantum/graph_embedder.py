import os
from dwave.system.samplers import DWaveSampler
from minorminer import find_embedding
from utils.file_io.datasaver import get_datasaver
from utils.file_io.dataloading.dataloader import get_dataloader
from utils.file_io.path_tracker import get_path_tracker
from stereo_matchers.matching_steps_handler import get_matching_steps_handler

def get_graph_embedder(args):
    ge = GraphEmbedder(args)
    return ge

def _get_total_qubits(embedded_graph):
    total = 0
    for _, chain in embedded_graph.items():
        total = total + len(chain)
    return total

def _get_max_chain_length(embedded_graph):
    max_chain_length = 0
    for _, chain in embedded_graph.items():
        if len(chain) > max_chain_length:
            max_chain_length = len(chain)
    return max_chain_length

class GraphEmbedder:
    def __init__(self,args):
        self.path_tracker = get_path_tracker(args)
        self.datasaver = get_datasaver(args)
        self.dataloader = get_dataloader(args)
        self.matching_steps_handler = get_matching_steps_handler(args)

    def initialize_model_at_step(self, step):
        self.num_candidates = self.matching_steps_handler.get_num_candidates_at_step(step)

    def get_embedding_for_bundle(self,Q,bundle):
        if self.embedding_exists(bundle):
            embedding = self.dataloader.get_embedding(bundle,self.num_candidates)
        else:
            print('Computing embedding')
            solver = DWaveSampler()
            _, target_edgelist, _ = solver.structure
            embedding = find_embedding(Q, target_edgelist, verbose=1)
            self.datasaver.save_embedding(embedding,bundle,self.num_candidates)
        print('Total qubits: ', _get_total_qubits(embedding))
        print('Max chain length: ', _get_max_chain_length(embedding))
        return embedding

    def embedding_exists(self,bundle):
        return os.path.exists(self.path_tracker.get_QUBO_embedding_path(bundle,self.num_candidates))    