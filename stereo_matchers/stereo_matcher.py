
from utils.quantum.qubo_solver_interface import get_qubo_solver_interface
from stereo_matchers.bundler import get_bundler
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
from stereo_matchers.stereo_offsets_handler import get_stereo_offsets_handler


def get_stereo_matcher(args):
    sm = StereoMatcher(args)
    return sm

class StereoMatcher:
    def __init__(self,args):
        self.bundler = get_bundler(args)
        self.qubo_solver_interface = get_qubo_solver_interface(args)
        self.matching_steps_handler = get_matching_steps_handler(args)
        self.stereo_offsets_handler  = get_stereo_offsets_handler(args)
        self.load_to_bundle = args.load_to_bundle
    
    def make_stereo_match(self):
        matching_steps = self.matching_steps_handler.get_range_of_matching_steps(ignore_jump_to_step=False)
        for step in matching_steps:
            print('On step', step)
            stereo_offsets = self.stereo_offsets_handler.initialize_stereo_offsets_at_step(step)
            self.qubo_solver_interface.initialize_model_at_step(step)
            bundles = self.bundler.get_bundles_at_step(step)
            for bundle in bundles:
                print('On bundle', bundle)
                bundle_offsets = self.qubo_solver_interface.compute_displacements_for_bundle(bundle)
                self.stereo_offsets_handler.add_bundle_offsets_to_stereo_offsets(stereo_offsets,bundle_offsets,bundle)
                self.stereo_offsets_handler.save_stereo_offsets(stereo_offsets,step)

            self.stereo_offsets_handler.save_stereo_offsets(stereo_offsets,step)

            if self.load_to_bundle != 0:
                self.load_to_bundle = 0