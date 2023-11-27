
from energy_model.edge_detector import get_edge_detector
from stereo_matchers.matching_steps_handler import get_matching_steps_handler

def get_regularization_model(args):
    dm = RegularizationModel(args)
    return dm

class RegularizationModel:
    def __init__(self,args):
        self.edge_detector = get_edge_detector(args)
        self.matching_steps_handler = get_matching_steps_handler(args)
        
    
    def initialize_model_at_step(self,step):
        self.edge_detector.initialize_model_at_step(step)
        self.regularization_slope = self.matching_steps_handler.get_regularization_slope_at_step(step)
        self.regularization_truncation_max = self.matching_steps_handler.get_regularization_truncation_max_at_step(step)
        self.regularization_edge_reduction = self.matching_steps_handler.get_regularization_edge_reduction_at_step(step)

    def get_regularization_energy(self,base_displacement,neighbor_displacement,base_y,base_x,neighbor_y,neighbor_x):
        displacement_diff = abs(base_displacement-neighbor_displacement)
        regularization_energy = self.regularization_slope*displacement_diff

        #TRUNCATE
        regularization_energy = min(self.regularization_truncation_max, regularization_energy)

        #EDGE DETECTION & ADJUSTMENT
        if self.edge_detector.is_edge(base_y,base_x,neighbor_y,neighbor_x):
            regularization_energy*=self.regularization_edge_reduction

        return regularization_energy
    

