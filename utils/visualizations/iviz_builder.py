from utils.file_io.dataloading.dataloader import get_dataloader
from utils.file_io.datasaver import get_datasaver
from utils.file_io.path_tracker import get_path_tracker
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
from itypes import Dataset
from energy_model.edge_detector import get_edge_detector
import numpy as np


def get_iviz_builder(args):
    ib = IvizBuilder(args)
    return ib

class IvizBuilder:
    def __init__(self,args):
        self.dataloader = get_dataloader(args)
        self.datasaver = get_datasaver(args)
        self.path_tracker = get_path_tracker(args)
        self.matching_steps_handler = get_matching_steps_handler(args)
        self.edge_detector = get_edge_detector(args)
        self.displayed_data = args.displayed_data
        self.display_friendly_displacements = not args.display_true_displacements
        self.image_display_step = args.image_display_step
        self.displayed_steps = args.displayed_steps
        self.display_displacements_with_post_processing = args.display_displacements_with_post_processing
        self.dataset = args.dataset
        return
    
    def build_iviz(self):
        ds = Dataset(file=self.path_tracker.get_iviz_save_file_path(), auto_write=True)
        with ds.viz.new_row() as row:
            self.initialize_row(row)

        group_name = self._get_iviz_group_string()
        with ds.seq.group(group_name, label=group_name) as group:
            self.populate_row(group)

        print(self.path_tracker.get_iviz_save_file_path())

    def initialize_row(self,row):
        if 'frame_0' in self.displayed_data:
            row.add_cell("image",  var = self._get_frame_name(0))
        if 'frame_1' in self.displayed_data:
            row.add_cell("image",  var = self._get_frame_name(1))
        if 'detected_edges' in self.displayed_data:
            row.add_cell("image", var = self._get_edge_name())
        if 'estimated_displacements' in self.displayed_data:
            for step in self.displayed_steps:
                row.add_cell("image",  var = self._get_estimated_displacement_name(step))

        if 'gt_displacements' in self.displayed_data:
            row.add_cell("image",  var = self._get_ground_truth_displacement_name())

    def populate_row(self,group):
        item_name = self._get_iviz_item_string()
        with group.item(item_name, label=item_name) as item:
            if 'frame_0' in self.displayed_data:
                frame_0 = self.dataloader.get_frame(0,step=self.image_display_step)
                item[self._get_frame_name(0)].set_data(frame_0,dims="hwc")
            if 'frame_1' in self.displayed_data:
                frame_1 = self.dataloader.get_frame(1, step=self.image_display_step)
                item[self._get_frame_name(1)].set_data(frame_1,dims="hwc")
            if 'detected_edges' in self.displayed_data:
                edges = self.edge_detector.get_detected_edges(step=self.image_display_step)
                item[self._get_edge_name()].set_data(edges,dims="hwc")
            if 'estimated_displacements' in self.displayed_data:
                for step in self.displayed_steps:
                    estimated_displacements = self.dataloader.get_estimated_displacements(step,can_post_process=self.display_displacements_with_post_processing,for_display=True)
                    item[self._get_estimated_displacement_name(step)].set_data(estimated_displacements,dims="hwc")

            if 'gt_displacements' in self.displayed_data:
                gt_displacements = self.dataloader.get_ground_truth_displacements(step=self.image_display_step,for_display=True)
                item[self._get_ground_truth_displacement_name()].set_data(gt_displacements,dims="hwc")

    def _get_frame_name(self,frame_number):
        return 'frame_{}'.format(frame_number)
    
    def _get_edge_name(self):
        return 'detected_edges'
    
    def _get_ground_truth_displacement_name(self):
        return 'ground truth displacement'
    
    def _get_estimated_displacement_name(self,step):
        downsample_factor = self.matching_steps_handler.get_downsample_factor_at_step(step)
        num_candidates = self.matching_steps_handler.get_num_candidates_at_step(step)
        estimated_displacement_string = 'estimated_displacement_{}_{}_{}'
        estimated_displacement_string = estimated_displacement_string.format(step,downsample_factor,num_candidates)
        return estimated_displacement_string
    
    def _get_iviz_group_string(self):
        return 'MIDDLEBURY'
    
    def _get_iviz_item_string(self):
        return 'ITEM NAME'