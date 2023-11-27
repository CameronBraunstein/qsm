

from utils.file_io.path_tracker import get_path_tracker
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
from utils.file_io.dataloading.dataloader_helper import (_load_pickle, _load_json, _convert_numpy_rgb_to_gray_scale, _resize_numpy_displacements,
    _median_filter, _bilateral_filter, _cast_to_uint8, _downsample_numpy_image, _calculate_frame_shape_from_downsample_factor, _downscale_numpy_array_by_factor)

class AbstractDataLoader:
    def __init__(self,args):
        self.matching_steps_handler = get_matching_steps_handler(args)
        self.scene_name = args.scene_name

    def get_frame(self,frame_number,step,convert_to_gray_scale=False):
        frame = self._load_frame(frame_number)
        frame = self._downsample_frame_at_step(frame,step)
        if convert_to_gray_scale:
            frame = _convert_numpy_rgb_to_gray_scale(frame)
        return frame
    
    def get_ground_truth_displacements(self, step,for_display=False,for_evaluation=False):
        ground_truth_displacements = self._load_ground_truth_displacements()
        if for_evaluation:
            ground_truth_displacements = self._crop_for_evaluation(ground_truth_displacements)
        downsample_factor = self.matching_steps_handler.get_downsample_factor_at_step(step)
        ground_truth_displacements = _resize_numpy_displacements(ground_truth_displacements,downsample_factor)
        if not for_display:
            ground_truth_displacements = self._downscale_displacements_for_step(ground_truth_displacements,step)
        else:
            
            ground_truth_displacements = _cast_to_uint8(ground_truth_displacements)
        return ground_truth_displacements 
    
    def get_estimated_displacements(self,step,can_post_process=True,for_display=False,for_evaluation=False):
        estimated_displacements = self._load_estimated_displacements(step)
        if for_evaluation:
            estimated_displacements = self._crop_for_evaluation(estimated_displacements)
        if not for_display:
            estimated_displacements = self._downscale_displacements_for_step(estimated_displacements,step)
        estimated_displacements = _cast_to_uint8(estimated_displacements)
        if can_post_process:
            estimated_displacements = self._apply_post_processing(estimated_displacements,step)
        return estimated_displacements
    
    def _crop_for_evaluation(self,displacements):
        crop_offset = self._get_crop_offset()
        if crop_offset > 0:
            displacements = displacements[crop_offset:-crop_offset, crop_offset:-crop_offset]
        return displacements
    
    def _downscale_displacements_for_step(self,displacements,step):
        downsample_factor = self.matching_steps_handler.get_downsample_factor_at_step(step)
        upscale_factor = self._get_upscale_factor()
        total_downscaling_factor = downsample_factor*upscale_factor
        displacements = _downscale_numpy_array_by_factor(displacements,total_downscaling_factor )
        return displacements

    def _load_ground_truth_displacements(self):
        ground_truth_displacement_path = self.path_tracker.get_ground_truth_displacement_path()
        ground_truth_displacements = self._load_ground_truth_displacements_from_path(ground_truth_displacement_path)
        return ground_truth_displacements 
    
    def _load_estimated_displacements(self,step):
        estimated_displacements_path = self.path_tracker.get_stereo_estimate_save_path(step)
        estimated_displacements = self._load_estimated_displacements_from_path(estimated_displacements_path)
        return estimated_displacements

    def _apply_post_processing(self,stereo_offsets,step):
        if self.matching_steps_handler.can_median_filter_after_step(step):
            median_filter_size = self.matching_steps_handler.get_median_filter_size(step)
            stereo_offsets = _median_filter(stereo_offsets, median_filter_size)
        if self.matching_steps_handler.can_bilateral_filter_after_step(step):
          stereo_offsets = _bilateral_filter(stereo_offsets)
        return stereo_offsets
        

    def get_frame_shape_at_step(self,step):
        downsample_factor = self.matching_steps_handler.get_downsample_factor_at_step(step)
        ground_truth_displacements = self._load_ground_truth_displacements()
        frame_shape = _calculate_frame_shape_from_downsample_factor(ground_truth_displacements,downsample_factor)
        return frame_shape
    
    def get_embedding(self,bundle, num_candidates):
        embedding_path = self.path_tracker.get_QUBO_embedding_path(bundle, num_candidates)
        embedding = _load_pickle(embedding_path)
        return embedding
    
    def _load_frame(self,frame_number):
        frame_path = self.path_tracker.get_frame_path(frame_number)
        frame = self._load_frame_from_path(frame_path)
        return frame
    
    def _downsample_frame_at_step(self,frame,step):
        downsample_shape= self.get_frame_shape_at_step(step)
        frame = _downsample_numpy_image(frame,downsample_shape)
        return frame
    
    def _get_upscale_factor(self):
        return 1

    def _get_crop_offset(self):
        return 0
    
class DataLoaderNoConfig(AbstractDataLoader):
    def __init__(self,args):
        super().__init__(args)
        self.path_tracker = get_path_tracker(args,ignore_config_settings=True)

    def load_config_dict(self):
        config_file_path = self.path_tracker.get_config_file_path()
        config_dict = _load_json(config_file_path)
        return config_dict


class DataLoaderConfig(AbstractDataLoader):
    def __init__(self,args):
        super().__init__(args)
        self.path_tracker = get_path_tracker(args,ignore_config_settings=False)