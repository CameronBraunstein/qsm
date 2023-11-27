from stereo_matchers.matching_steps_handler import get_matching_steps_handler

def get_report_log(args):
    rl = ReportLog(args)
    return rl

class ReportLog:
    def __init__(self,args):
        self.matching_steps_handler = get_matching_steps_handler(args)
        self.root_mean_squared_log = {}
        self.bad_pixels_log = {}
        self.scene_name = args.scene_name
    
    def add_root_mean_squared(self,rms,step):
        self.root_mean_squared_log[step] = rms

    def add_bad_pixels(self,bad_pixels,step):
        self.bad_pixels_log[step] = bad_pixels

    def print_report(self):
        evaluation_steps = self.matching_steps_handler.get_evaluation_steps()
        for step in evaluation_steps:
            downsample_factor = self.matching_steps_handler.get_downsample_factor_at_step(step)
            num_candidates = self.matching_steps_handler.get_num_candidates_at_step(step)

            print('')
            print('_____')
            print(self.scene_name)
            print('RMS: ', self.root_mean_squared_log[step])
            print('Bad Pixel Metric: ', self.bad_pixels_log[step])
            print('_____')
