from utils.file_io.dataloading.dataloader import get_dataloader
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
import numpy as np

def get_bundler(args):
    b = Bundler(args)
    return b

class Bundler:
    def __init__(self,args):
        self.dataloader = get_dataloader(args)
        self.matching_steps_handler = get_matching_steps_handler(args)
        self.load_to_height = args.load_to_bundle
    
    def get_bundles_at_step(self, step):
        bundle_height = self.matching_steps_handler.get_bundle_height_at_step(step)
        bundles = self._get_bundles_from_bundle_height(bundle_height,step)

        return bundles
    
    def _get_bundles_from_bundle_height(self,bundle_height,step):
        height, width = self.dataloader.get_frame_shape_at_step(step)

        height_divisions = np.arange(self.load_to_height,height,bundle_height)
        if height_divisions[-1] != height:
            height_divisions = np.append(height_divisions,[height])

        width_range = range(width)
        
        bundles = [(range(height_divisions[i],height_divisions[i+1]),width_range) for i in range(len(height_divisions)-1)]
        return bundles
        
