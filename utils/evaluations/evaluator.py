from utils.file_io.dataloading.dataloader import get_dataloader
from utils.evaluations.report_log import get_report_log
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
import numpy as np
import math

def get_evaluator(args):
    e = Evaluator(args)
    return e

def _calculate_differences(array_1, array_2):
    int_array_1 = array_1.astype(float)
    int_array_2 = array_2.astype(float)
    differences = int_array_1 - int_array_2
    return differences

def _calculate_root_mean_squared(ground_truth_displacements,estimated_displacements):
    differences = _calculate_differences(ground_truth_displacements,estimated_displacements)
    squared_differences = np.square(differences)
    mean_squared_differences = np.mean(squared_differences)
    root_mean_squared_differences = math.sqrt(mean_squared_differences)
    return root_mean_squared_differences

def _calculate_bad_pixels(ground_truth_displacements,estimated_displacements,disparity_error_tolerance):
    differences = _calculate_differences(ground_truth_displacements,estimated_displacements)
    absolute_value_differences = np.abs(differences)
    bad_pixel_count = np.sum(absolute_value_differences > disparity_error_tolerance)
    b = bad_pixel_count / np.size(differences) * 100
    return b

class Evaluator:
    def __init__(self,args):
        self.disparity_error_tolerance = args.disparity_error_tolerance
        self.dataloader = get_dataloader(args)
        self.report_log = get_report_log(args)
        self.matching_steps_handler = get_matching_steps_handler(args)
        self.scene_name =args.scene_name
    
    def evaluate(self):
        evaluation_steps = self.matching_steps_handler.get_evaluation_steps()
        for step in evaluation_steps:
            ground_truth_displacements = self.dataloader.get_ground_truth_displacements(step=step,for_evaluation=True)
            estimated_displacements = self.dataloader.get_estimated_displacements(step,can_post_process=True, for_evaluation=True)
            rms = _calculate_root_mean_squared(ground_truth_displacements, estimated_displacements)
            bad_pixels = _calculate_bad_pixels(ground_truth_displacements, estimated_displacements,self.disparity_error_tolerance)

            self.report_log.add_root_mean_squared(rms,step)
            self.report_log.add_bad_pixels(bad_pixels,step)
        self.report_log.print_report()