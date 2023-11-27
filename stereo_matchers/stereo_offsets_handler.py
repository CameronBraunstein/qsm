
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
from utils.file_io.dataloading.dataloader import get_dataloader
from utils.file_io.datasaver import get_datasaver
import numpy as np
from skimage.transform import resize

def get_stereo_offsets_handler(args):
    soh = StereoOffsetsHandler(args)
    return soh

class StereoOffsetsHandler:
    def __init__(self,args):
        self.matching_steps_handler = get_matching_steps_handler(args)
        self.dataloader = get_dataloader(args)
        self.datasaver = get_datasaver(args)
        self.load_to_bundle = args.load_to_bundle

    def save_stereo_offsets(self,stereo_offsets,step):
        self.datasaver.save_stereo_offsets(stereo_offsets,step)

    def initialize_stereo_offsets_at_step(self,current_step):
        if self.load_to_bundle != 0:
            stereo_offsets = self.dataloader.get_estimated_displacements(current_step,can_post_process=False)
            return stereo_offsets

        if current_step == 0:
            stereo_offsets = self._initialize_stereo_offsets_empty()
        else:
            previous_step = current_step - 1
            stereo_offsets = self.dataloader.get_estimated_displacements(previous_step,can_post_process=True)
            stereo_offsets = self._resize_stereo_offsets(stereo_offsets,current_step)
            stereo_offsets = self._upscale_stereo_offsets(stereo_offsets,previous_step,current_step)
            
        return stereo_offsets
        

    def _initialize_stereo_offsets_empty(self):
        image_shape = self.dataloader.get_frame_shape_at_step(0)
        stereo_offsets = np.zeros(image_shape)
        return stereo_offsets


    def _resize_stereo_offsets(self,stereo_offsets,current_step):
        new_stereo_shape = self.dataloader.get_frame_shape_at_step(current_step)
        resized_stereo_offsets = resize(stereo_offsets,new_stereo_shape,order=0)
        return resized_stereo_offsets


    def _upscale_stereo_offsets(self,stereo_offsets,previous_step,current_step):
        current_downsample_factor = self.matching_steps_handler.get_downsample_factor_at_step(previous_step)
        next_downsample_factor = self.matching_steps_handler.get_downsample_factor_at_step(current_step)
        upscale_factor = current_downsample_factor // next_downsample_factor
        upscaled_stereo_offsets = stereo_offsets * upscale_factor
        return upscaled_stereo_offsets
    

    def add_bundle_offsets_to_stereo_offsets(self,stereo_offsets,bundle_offsets,bundle):
        height_range,width_range = bundle
        min_height = height_range[0]
        max_height = height_range[-1]
        min_width = width_range[0]
        max_width = width_range[-1]
        stereo_offsets[min_height:max_height+1,min_width:max_width+1] = bundle_offsets
