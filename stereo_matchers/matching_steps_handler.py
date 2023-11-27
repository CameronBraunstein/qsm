
def get_matching_steps_handler(args):
    msh = MatchingStepsHandler(args)
    return msh

class MatchingStepsHandler:
    def __init__(self,args):
        self.jump_to_step = args.jump_to_step
        self.downsample_factors = args.downsample_factors
        self.candidate_counts = args.candidate_counts
        
        self.median_filter_flags = args.median_filter_flags
        self.median_filter_ranges = args.median_filter_ranges
        self.bilateral_filter_flags = args.bilateral_filter_flags

        self.edge_thresholds = args.edge_thresholds

        self.regularization_slopes = args.regularization_slopes
        self.regularization_truncation_max = args.regularization_truncation_max
        self.regularization_edge_reduction = args.regularization_edge_reduction

        self.evaluate_all_steps = args.evaluate_all_steps
        self.bundle_heights = args.bundle_heights

    def get_median_filter_size(self,step):
        median_filter_size = self.median_filter_ranges[step]
        return median_filter_size

    def can_median_filter_after_step(self,step):
        can_median_filter_after_step = True if self.median_filter_flags[step] else False
        return can_median_filter_after_step
    
    def can_bilateral_filter_after_step(self,step):
        can_bilateral_filter_after_step = True if self.bilateral_filter_flags[step] else False
        return can_bilateral_filter_after_step
    
    def has_no_next_step(self,current_step):
        matching_steps = self.get_range_of_matching_steps()
        next_step = current_step + 1
        if next_step in matching_steps:
            has_no_next_step = False
        else:
            has_no_next_step = True
        return has_no_next_step

    def get_downsample_factor_at_step(self,step):
        return self.downsample_factors[step]
    
    def get_num_candidates_at_step(self,step):
        return self.candidate_counts[step]
    
    def get_edge_threshold_at_step(self,step):
        return self.edge_thresholds[step]
    
    def get_regularization_slope_at_step(self, step):
        return self.regularization_slopes[step]
    
    def get_regularization_truncation_max_at_step(self,step):
        return self.regularization_truncation_max[step]
    
    def get_regularization_edge_reduction_at_step(self,step):
        return self.regularization_edge_reduction[step]
     
    def get_bundle_height_at_step(self,step):
        return self.bundle_heights[step]
    

    def get_downsample_levels_and_candidate_counts(self):
        return self.downsample_factors, self.candidate_counts
    
    def get_evaluation_steps(self):
        all_matching_steps = self.get_range_of_matching_steps()
        if self.evaluate_all_steps:
            evaluation_steps = all_matching_steps
        else:
            evaluation_steps = [all_matching_steps[-1]]
        return evaluation_steps
    
    def get_range_of_matching_steps(self,ignore_jump_to_step=True):
        total_number_of_steps = self.get_total_number_of_steps()
        if ignore_jump_to_step:
            matching_steps = range(total_number_of_steps)
        else:
            matching_steps = range(self.jump_to_step,total_number_of_steps)
        return matching_steps
    
    def get_total_number_of_steps(self):
        total_number_of_steps = len(self.downsample_factors)
        return total_number_of_steps
    
    def get_bundle_height_max_at_step(self,step):
        if self.custom_bundle_height_maxes is None:
            bundle_height_max = float('inf')
        else:
            bundle_height_max = self.custom_bundle_height_maxes[step]
        return bundle_height_max
