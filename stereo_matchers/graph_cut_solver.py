from utils.file_io.dataloading.dataloader import get_dataloader
from energy_model.data_model import get_data_model
from utils.quantum.coordinate_canonicalizer import get_coordinate_canonicalizer
from utils.quantum.displacement_calculator import get_displacement_calculator
import numpy as np
import networkx as nx
from networkx.algorithms.flow import minimum_cut

def get_graph_cut_solver(args):
    GCS = GraphCutSolver(args)
    return GCS

class GraphCutSolver:
    def __init__(self,args):
        self.regularization_param = args.regularization_slopes[0]
        self.dataloader = get_dataloader(args)
        self.data_model = get_data_model(args)
        self.coordinate_canonicalizer = get_coordinate_canonicalizer(args)
        self.data_model.initialize_model_at_step(0)

        dc = get_displacement_calculator(args)
        self.min_displacement = 0
        self.max_displacement = dc._get_max_displacement(0)
        self.k = self.max_displacement - self.min_displacement + 1
        self.c = (4 * self.k * self.regularization_param) + 0.1

    def get_offsets(self):
        height, width = self.dataloader.get_frame_shape_at_step(0)
        offsets = np.zeros((height,width))
        for line in range(height):
            if line % 10 == 0:
                print('On line ', line)
            epipolar_line_offsets = self.create_displacements(line,width)
            offsets[line,:] = epipolar_line_offsets
        return offsets

    def create_displacements(self,bundle):
        self.coordinate_canonicalizer.set_bundle(bundle)
        st_graph = self.build_graph(bundle)
        cut = minimum_cut(st_graph, 'S', 'T')
        displacements = self.decode_cut(cut,bundle)
        return displacements
    
    def decode_cut(self,cut,bundle):
        if 'S' in cut[1][0]:
            source = cut[1][0]
        else:
            source = cut[1][1]

        height_range,width_range = bundle
        displacements = np.zeros((len(height_range),len(width_range)))
        for y in height_range:
            for x in width_range:
                canonical_y, canonical_x = self.coordinate_canonicalizer.canonicalize_coordinates(y,x)
                estimate = self.min_displacement
                for displacement in range(self.min_displacement, self.max_displacement+1):
                    v = _get_standard_vertex_name(canonical_y, canonical_x, displacement)
                    if v in source:
                        estimate = displacement+1
                    else:
                        break
                displacements[canonical_y, canonical_x] = estimate
        return displacements

    
    def build_graph(self,bundle):
        height_range,width_range = bundle
        G = nx.Graph()
        greatest_edge_weight = 0
        for y in height_range:
            for x in width_range:
                canonical_y, canonical_x = self.coordinate_canonicalizer.canonicalize_coordinates(y,x)
                for displacement in range(self.min_displacement, self.max_displacement+1):
                    v = _get_standard_vertex_name(canonical_y, canonical_x, displacement)
                    G.add_node(v)
                    if displacement > self.min_displacement: #t link edge
                        t_link_v = _get_standard_vertex_name(canonical_y, canonical_x, displacement-1)
                        edge = (v, t_link_v)

                        energy = self.data_model.get_data_energy(y,x,displacement)
                        edge_weight = energy + self.c

                        if edge_weight > greatest_edge_weight:
                            greatest_edge_weight = edge_weight
                        G.add_edge(*edge, capacity= edge_weight)
                    if canonical_x > 0: #n link edge
                        n_link_v = _get_standard_vertex_name(canonical_y, canonical_x-1, displacement)
                        edge = (v, n_link_v)

                        edge_weight = self.regularization_param
                        
                        if edge_weight > greatest_edge_weight:
                            greatest_edge_weight = edge_weight
                        G.add_edge(*edge, capacity=edge_weight )
                    if canonical_y > 0: #n link edge
                        n_link_v = _get_standard_vertex_name(canonical_y-1, canonical_x, displacement)
                        edge = (v, n_link_v)
                        edge_weight = self.regularization_param
                        
                        if edge_weight > greatest_edge_weight:
                            greatest_edge_weight = edge_weight
                        G.add_edge(*edge, capacity=edge_weight )

        greatest_edge_weight += 1
        s = 'S'
        t = 'T'
        G.add_node(s)
        G.add_node(t)
        for y in height_range:
            for x in width_range:
                canonical_y, canonical_x = self.coordinate_canonicalizer.canonicalize_coordinates(y,x)
                min_disp_v = _get_standard_vertex_name(canonical_y, canonical_x,self.min_displacement)
                edge = (s, min_disp_v)
                G.add_edge(*edge, weight=greatest_edge_weight)

                max_disp_v = _get_standard_vertex_name(canonical_y, canonical_x,self.max_displacement)
                edge = (t, max_disp_v)
                G.add_edge(*edge, weight=greatest_edge_weight)

        return G


def _get_standard_vertex_name(canonical_y,canonical_x, displacement):
    return 'p_{}_{}_{}'.format(canonical_y,canonical_x,displacement)