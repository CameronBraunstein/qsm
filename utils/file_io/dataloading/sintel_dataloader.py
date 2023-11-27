from utils.file_io.dataloading.abstract_dataloaders import DataLoaderConfig
from utils.file_io.dataloading.dataloader_helper import _load_png, _load_disp_from_png, _load_pgm

class SintelDataloader(DataLoaderConfig):
    def _load_frame_from_path(self,frame_path):
        return _load_png(frame_path)
    
    def _load_ground_truth_displacements_from_path(self,ground_truth_displacement_path):
        return _load_disp_from_png(ground_truth_displacement_path)
    
    def _load_estimated_displacements_from_path(self,displacement_path):
        return _load_pgm(displacement_path)

