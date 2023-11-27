import argparse
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
from utils.file_io.dataloading.dataloader import get_dataloader


def get_args():
    parser = argparse.ArgumentParser(description='Utility for training and evaluating DCFlow',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--actions', nargs='+',choices=['stereo_match','evaluate','build_display','visualize_paper', 'visualize_matrix'],
                        default=['evaluate'],
                        help='what actions should this utility take?')
    parser.add_argument('--dataset', choices=['Middlebury', 'Sintel'], default='Middlebury',
                        help='Dataset to Select')
    parser.add_argument('--scene_name',type=str,
                        help='Scene To Select')
    parser.add_argument('--scene_frame_number',type=int,
                        help = "Frame number for a Sintel scene")
    

    parser.add_argument('--config_name', help='Name of json file to store configuration for all matching steps. This allows for easier tracking of config experiments')


    parser.add_argument('--qubo_solver', choices=['gurobi', 'simulated_annealing', 'dwave_qpu', 'dwave_hybrid','graph_cut'],
                        help='which QUBO solving technique do we use?')

    #CONFIG SETTINGS
    parser.add_argument('--downsample_factors',nargs='+',help='Downsample factors for each step')
    parser.add_argument('--candidate_counts',nargs='+',help='Candidate counts for each step')



    parser.add_argument('--median_filter_flags', nargs='+',
                help="when should we apply a median filter to the stereo offsets as post processing?")
    parser.add_argument('--median_filter_ranges', nargs='+',
                help="how large are the median filters")
    parser.add_argument('--bilateral_filter_flags', nargs='+',
                help="when should we apply a bilateral filter to the stereo offsets as post processing?")
    parser.add_argument('--edge_thresholds', nargs='+',
                help="How large of a grayscale difference must there be in order to detect an edge?")

    parser.add_argument('--regularization_slopes' , nargs='+')
    parser.add_argument('--regularization_truncation_max' , nargs='+')
    parser.add_argument('--regularization_edge_reduction' , nargs='+')
    parser.add_argument('--bundle_heights' , nargs='+',
                help="Specify the maximum bundle height for a specfic step.")


    #DEV SETTINGS
    parser.add_argument('--jump_to_step',type=int,default=0, help="Skip ahead and use previously calculated off sets")
    parser.add_argument('--load_to_bundle',type=int,default=0, help="Should we only calculate epipolar lines starting at a certain level? (Useful for interruptions)")
    

    #ENERGY MODEL SETTINGS
    parser.add_argument("--use_non_granular_constraints", action='store_true',
                    help="do not use granular rectifiers for the one hot encoding")


    #QUBO SETTINGS
    parser.add_argument('--annealing_runs', type=int, default=500,
            help="how many candidates do we consider per QUBO?")
    parser.add_argument('--qubo_encoding_type',choices=['binary','one_hot'], default='one_hot',
                help='How should the QUBO be encoded?')
    parser.add_argument('--constraint_weight', default=1.0, type=float,
                help= 'how strongly should we consider constraint weights in the one hot encoding qubo case?')
    
    #EVALUATION SETTINGS
    parser.add_argument('--disparity_error_tolerance', type=float, default=1.0, 
                help="What disparity difference from ground truth counts as bad?")
    parser.add_argument('--evaluate_all_steps', action='store_true',
            help="do we look at errors in coarser steps as well?")
    
    #DISPLAY SETTINGS
    parser.add_argument('--displayed_data', nargs='+',choices=['frame_0','frame_1','gt_displacements','estimated_displacements'],
                        default=['frame_0','frame_1','gt_displacements', 'estimated_displacements','detected_edges'],
                        help='what data do we display in iviz')
    parser.add_argument('--display_true_displacements', action='store_true',
                help="when displaying displacements, do we not upscale them by factor 8?")
    parser.add_argument('--displayed_steps',type=int, nargs='+', default=[-1],
                help="Which steps from matching do we display with iviz?")
    
    parser.add_argument('--image_display_step', type=int, default=0, 
                help="From which step do we display the frames and edges?")
    
    parser.add_argument('--display_displacements_with_post_processing', action='store_true',
                help="when displaying displacements, do we show how they look with filter post processing?")
    
    args = parser.parse_args()

    _load_config_file(args)
    _update_display_info(args)
    return args

def _load_config_file(args):
    dl = get_dataloader(args,ignore_config_settings=True)
    config_dict = dl.load_config_dict()
    for key, value in config_dict.items():
        if getattr(args, key) is not None:
            print('WARNING: You are overriding config file settings for {}. This is not recommended'.format(key))
        else:
            setattr(args,key,value)

def _update_display_info(args):
    if -1 in args.displayed_steps:
        msh = get_matching_steps_handler(args)
        all_steps = msh.get_range_of_matching_steps()
        args.displayed_steps = all_steps