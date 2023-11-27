from utils.file_io.dataloading.abstract_dataloaders import DataLoaderConfig
from utils.file_io.dataloading.dataloader_helper import _load_ppm, _adjust_displacement_pgm, _load_pgm

class MiddleburyDataloader(DataLoaderConfig):
    def _load_frame_from_path(self,frame_path):
        return _load_ppm(frame_path)
    
    def _load_ground_truth_displacements_from_path(self,displacement_path):
        ground_truth_displacements = self._load_displacements(displacement_path)
        return ground_truth_displacements

    def _load_estimated_displacements_from_path(self,displacement_path):
        return self._load_displacements(displacement_path)

    def _load_displacements(self,displacement_path):
        pgm_as_numpy = _load_pgm(displacement_path)
        return pgm_as_numpy

    def _get_upscale_factor(self):
        if self.scene_name == 'tsukuba':
            upscale_factor = 16
        else:
            upscale_factor = 8
        return upscale_factor
    
    def _get_crop_offset(self):
        if self.scene_name == 'tsukuba':
            crop_offset = 18
        else:
            crop_offset = 0
        return crop_offset