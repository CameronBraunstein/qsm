from utils.file_io.dataloading.dataloader import get_dataloader
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
import numpy as np

def get_edge_detector(args):
    ed = EdgeDetector(args)
    return ed

class EdgeDetector:
    def __init__(self,args):
        self.dataloader = get_dataloader(args)
        self.matching_steps_handler = get_matching_steps_handler(args)


    def initialize_model_at_step(self,step):
        self.edge_threshold = self.matching_steps_handler.get_edge_threshold_at_step(step)
        self._initialize_images_at_downsample_factor(step)

    #Here to estimate rough idea of where edges are detected, not the ground truth
    def get_detected_edges(self,step):
        self.initialize_model_at_step(step)
        height,width = self.dataloader.get_frame_shape_at_step(step)
        edges = np.zeros((height-1,width-1))
        for base_y in range(height-1):
            for base_x in range(width-1):
                neighbor_y = base_y
                neighbor_x = base_x + 1
                if self.is_edge(base_y,base_x,neighbor_y,neighbor_x):
                    edges[base_y,base_x] = 1
                neighbor_y = base_y + 1
                neighbor_x = base_x
                if self.is_edge(base_y,base_x,neighbor_y,neighbor_x):
                    edges[base_y,base_x] = 1
        return edges
    
    def _initialize_images_at_downsample_factor(self,step):
        self.frame_0 = self.dataloader.get_frame(0,step=step,convert_to_gray_scale=True)
        self.frame_1 = self.dataloader.get_frame(1,step=step,convert_to_gray_scale=True)

    def is_edge(self,base_y,base_x,neighbor_y,neighbor_x):
        base_pixel = self.frame_0[base_y,base_x]
        neighbor_pixel = self.frame_0[neighbor_y,neighbor_x]
        gray_value_diff = abs(base_pixel-neighbor_pixel)
        if gray_value_diff > self.edge_threshold:
            is_edge = True
        else:
            is_edge = False   
        return is_edge