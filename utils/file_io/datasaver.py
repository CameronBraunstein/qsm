from stereo_matchers.matching_steps_handler import get_matching_steps_handler
from utils.file_io.path_tracker import get_path_tracker
from utils.file_io.datasaver_helper import _make_missing_directories, _save_pgm, _upscale_numpy_array_by_factor, _save_pickle
import numpy as np


def get_datasaver(args):
    if args.dataset == 'Middlebury':
        dl = MiddleburyDatasaver(args)
    elif args.dataset == 'Sintel':
        dl = SintelDatasaver(args)
    return dl

class Datasaver:
    def __init__(self,args):
        self.path_tracker = get_path_tracker(args)
        self.matching_steps_handler = get_matching_steps_handler(args)

    def save_stereo_offsets(self,stereo_offsets,step):
        stereo_offsets_copy = np.copy(stereo_offsets)
        stereo_offsets_copy = self._rescale_displacements_for_saving(stereo_offsets_copy,step)
        save_path = self.path_tracker.get_stereo_estimate_save_path(step)
        _save_pgm(stereo_offsets_copy,save_path)

    def _rescale_displacements_for_saving(self,displacements,step):
        downsample_factor = self.matching_steps_handler.get_downsample_factor_at_step(step)
        upscale_factor = self._get_upscale_factor()
        total_upscale_factor = downsample_factor * upscale_factor
        displacements = _upscale_numpy_array_by_factor(displacements,total_upscale_factor)
        return displacements


    def save_embedding(self,embedding,bundle,num_candidates):
        embedding_path = self.path_tracker.get_QUBO_embedding_path(bundle,num_candidates)
        _make_missing_directories(embedding_path)
        _save_pickle(embedding,embedding_path)


class MiddleburyDatasaver(Datasaver):
    def __init__(self,args):
        super().__init__(args)
        self.scene_name = args.scene_name

    def _get_upscale_factor(self):
        if self.scene_name == 'tsukuba':
            upscale_factor = 16
        else:
            upscale_factor = 8
        return upscale_factor
    

class SintelDatasaver(Datasaver):
    def _get_upscale_factor(self):
        return 1