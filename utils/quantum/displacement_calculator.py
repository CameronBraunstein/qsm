
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
from stereo_matchers.stereo_offsets_handler import get_stereo_offsets_handler
from utils.file_io.dataloading.dataloader import get_dataloader
import numpy as np

def get_displacement_calculator(args):
    if args.qubo_encoding_type == 'binary':
        raise Exception('NOT YET SUPPORTED')
    elif args.qubo_encoding_type == 'one_hot':
        dc = OneHotDisplacementCalculator(args)
    return dc

class DisplacementCalculator:
    def __init__(self,args):
        self.dataloader = get_dataloader(args)
        self.matching_steps_handler = get_matching_steps_handler(args)
        self.stereo_offsets_handler = get_stereo_offsets_handler(args)
        self.dataset = args.dataset
    
    def initialize_model_at_step(self,step):
        self.stereo_offsets  = self.stereo_offsets_handler.initialize_stereo_offsets_at_step(step)

        self.max_displacement_at_step = self._get_max_displacement(step)
        self.num_candidates = self.matching_steps_handler.get_num_candidates_at_step(step)
        if self.num_candidates > self.max_displacement_at_step:
            self.can_correct_overshoot = False
            print('WARNING: YOU ARE CONSIDERING MORE CANDIDATES THAN YOU HAVE TO AT THIS RESOLUTION')
            print('BEWARE THAT THIS CAN CAUSE UNNECESSARY INEFFICIENCIES')
        else:
            self.can_correct_overshoot = True
        
    def get_displacement_range(self,y,x):
        stereo_offset = self.stereo_offsets[y,x]
        start = stereo_offset - (self.num_candidates // 2)
        end = start + self.num_candidates

        under_shoot = min(0,start)
        start -= under_shoot
        end -= under_shoot

        if self.can_correct_overshoot:
            over_shoot = max(0,end - self.max_displacement_at_step)
            start -= over_shoot
            end -= over_shoot

        displacement_range = range(int(start),int(end))
        return displacement_range

    def get_displacement_and_canonical_range(self,y,x):
        displacement_range = self.get_displacement_range(y,x)
        canonical_range = range(self.num_candidates)
        return displacement_range , canonical_range
    
    def _get_max_displacement(self,step):
        ground_truth_displacements = self.dataloader.get_ground_truth_displacements(step=step)
        max_displacement = int(np.ceil(np.max(ground_truth_displacements)))
        return max_displacement

class OneHotDisplacementCalculator(DisplacementCalculator):
    def __init__(self,args):
        super().__init__(args)
    