from stereo_matchers.matching_steps_handler import get_matching_steps_handler
import os

def get_path_tracker(args,ignore_config_settings=False):
    if ignore_config_settings:
        pt = PathTrackerNoConfig(args)
    else:
        if args.dataset == 'Middlebury':
            pt = PathTrackerMiddlebury(args)
        elif args.dataset == 'Sintel':
            pt = PathTrackerSintel(args)

    return pt

class PathTrackerNoConfig:
    def __init__(self,args):
        self.matching_steps_handler = get_matching_steps_handler(args)
        self.scene_name = args.scene_name
        self.config_name = args.config_name
        self.qubo_solver = args.qubo_solver

    def get_config_file_path(self):
        config_file_directory = self._get_config_file_directory()
        config_file_name = '{}.json'
        config_file_name = config_file_name.format(self.config_name)
        config_file_path = os.path.join(config_file_directory,config_file_name)
        return config_file_path

    def _get_config_file_directory(self):
        config_file_directory = 'matching_step_configs'
        return config_file_directory

class PathTrackerBase(PathTrackerNoConfig):
    def __init__(self,args):
        super().__init__(args)
        self._parse_qubo_solver_specific_args(args)

    def _parse_qubo_solver_specific_args(self,args):
        if self.qubo_solver == 'graph_cut':
            self.graph_cut_regularizer = args.regularization_slopes[0]
        if self.qubo_solver == 'gurobi':
            self._parse_mrf_related_args(args)
        if self.qubo_solver == 'dwave_qpu':
            self._parse_annealing_related_args(args)
        if self.qubo_solver == 'dwave_hybrid':
            self._parse_annealing_related_args(args)
        if self.qubo_solver == 'simulated_annealing':
            self._parse_annealing_related_args(args)

    def _parse_annealing_related_args(self,args):
        self.annealing_runs = args.annealing_runs
        self._parse_mrf_related_args(args)

    def _parse_mrf_related_args(self,args):
        self.qubo_encoding_type = args.qubo_encoding_type
        if self.qubo_encoding_type == 'one_hot':
            self.constraint_weight = args.constraint_weight

    def get_frame_path(self,frame_number):
        scene_directory = self._get_scene_directory()
        frame_filename = self._get_frame_name(frame_number)
        frame_path = os.path.join(scene_directory,frame_filename)
        return frame_path
    
    def get_ground_truth_displacement_path(self):
        scene_directory = self._get_scene_directory()
        displacement_filename = self._get_displacement_file_name()
        displacement_path = os.path.join(scene_directory,displacement_filename)
        return displacement_path
    
    def get_iviz_save_file_path(self):
        output_data_path = self._get_output_data_path()
        regimine_string = self._get_matching_steps_regimine_string()
        save_file_name = 'iviz_{}_{}'
        save_file_name = save_file_name.format(self.scene_name,regimine_string)
        iviz_save_file_path = os.path.join(output_data_path,'iviz_data',save_file_name)
        return iviz_save_file_path
    
    def get_stereo_estimate_save_path(self,step):
        output_data_path = self._get_output_data_path()
        regimine_directory = self._get_matching_steps_regimine_string()
        save_file_name = self._get_stereo_estimate_save_file_name(step)
        stereo_estimate_save_file_path = os.path.join(output_data_path,'stereo_estimates',regimine_directory,save_file_name)
        return stereo_estimate_save_file_path
    
    def get_QUBO_embedding_path(self,bundle,num_candidates):
        output_data_path = self._get_output_data_path()
        height = len(bundle[0])
        width = len(bundle[1])
        qubo_embedding_file_name = '{}_{}_{}_{}.pickle'.format(height,width,self.qubo_encoding_type,num_candidates)
        qubo_embedding_path = os.path.join(output_data_path,'QUBO_embeddings',qubo_embedding_file_name)
        return qubo_embedding_path
        
    def _get_matching_steps_regimine_string(self):
        regimine_string = self.config_name
        return regimine_string
    
    def _get_annealer_description(self):
        if self.qubo_solver == 'graph_cut':
            annealer_description = 'Graph_Cut_{}'.format(int(self.graph_cut_regularizer*100000))
        elif self.qubo_solver == 'gurobi':
            if self.constraint_weight != 1:
                annealer_description = 'GUROBI_{}'.format(int(self.constraint_weight*100))
            else:
                annealer_description = 'GUROBI'
        elif self.qubo_solver == 'dwave_qpu':
            annealer_description = 'Pure_DWAVE_{}'.format(self.annealing_runs)
        elif self.qubo_solver == 'dwave_hybrid':
            annealer_description = 'Hybrid_DWAVE_{}'.format(self.annealing_runs)
        elif self.qubo_solver == 'simulated_annealing':
            if self.constraint_weight != 1:
                annealer_description = 'SA_{}_{}'.format(self.annealing_runs,int(self.constraint_weight*100))
            else:
                annealer_description = 'SA_{}'.format(self.annealing_runs)
        return annealer_description
    
    def _get_stereo_estimate_save_file_name(self,step):
        annealer_description = self._get_annealer_description()
        save_file_name = 'stereo_estimate_{}_{}_{}.pgm'
        save_file_name = save_file_name.format(self.scene_name,step,annealer_description)
        return save_file_name
    
    def _get_output_data_path(self):
        return os.path.join('output_data')

    def _get_scene_directory(self):
        dataset_directory = self._get_dataset_directory()
        scene_directory = os.path.join(dataset_directory,self.scene_name)
        return scene_directory
    
    def get_paper_image_path(self,old_image_name,folder_name):
        return os.path.join('paper_visualizations',folder_name,old_image_name)


class PathTrackerSintel(PathTrackerBase):
    def __init__(self,args):
        super().__init__(args)
        self.scene_frame_number = args.scene_frame_number
    
    def get_frame_path(self,frame_number):
        if frame_number == 0:
            left_or_right = 'clean_right'
        elif frame_number == 1:
            left_or_right = 'clean_left'

        frame_filename = 'frame_{:04}.png'.format(self.scene_frame_number)
        frame_path = os.path.join('datasets','training',left_or_right,self.scene_name,frame_filename)
        return frame_path
    
    def _get_stereo_estimate_save_file_name(self,step):
        annealer_description = self._get_annealer_description()
        save_file_name = 'stereo_estimate_{}_{}_{}_{}.pgm'
        save_file_name = save_file_name.format(self.scene_name,self.scene_frame_number,step,annealer_description)
        return save_file_name
    
    def get_ground_truth_displacement_path(self):
        frame_filename = 'frame_{:04}.png'.format(self.scene_frame_number)
        displacement_path = os.path.join('datasets','training','disparities',self.scene_name,frame_filename)
        return displacement_path

    # def _get_frame_name(self,frame_number):
    #     frame_filename = '{}.png'
    #     if frame_number == 0:
    #         frame_filename = frame_filename.format('right')
    #     elif frame_number == 1:
    #         frame_filename = frame_filename.format('left')
    #     return frame_filename
    
    # def _get_displacement_file_name(self):
    #     displacement_filename = 'disparities.png'
    #     return displacement_filename

    # def _get_dataset_directory(self):
    #     return os.path.join('datasets','Sintel')

class PathTrackerCustomScene(PathTrackerBase):
    def __init__(self,args):
        super().__init__(args)
        self.custom_scene_frame_number = args.custom_scene_frame_number #First frame number

    def get_frame_path(self,frame_number):
        if frame_number == 0:
            left_or_right = 'clean_right'
        elif frame_number == 1:
            left_or_right = 'clean_left'

        frame_filename = 'frame_{:04}.png'.format(self.custom_scene_frame_number)
        frame_path = os.path.join('datasets','custom',left_or_right,self.scene_name,frame_filename)
        return frame_path
    
    def _get_stereo_estimate_save_file_name(self,step):
        annealer_description = self._get_annealer_description()
        save_file_name = 'stereo_estimate_{}_{}_{}_{}.pgm'
        save_file_name = save_file_name.format(self.scene_name,self.custom_scene_frame_number,step,annealer_description)
        return save_file_name
    
    def get_ground_truth_displacement_path(self):
        frame_filename = 'frame_{:04}.png'.format(self.custom_scene_frame_number)
        displacement_path = os.path.join('datasets','custom','disparities',self.scene_name,frame_filename)
        return displacement_path

class PathTrackerMiddlebury(PathTrackerBase):
    def _get_frame_name(self,frame_number):
        frame_filename = 'im{}.ppm'
        if frame_number == 0:
            frame_filename = frame_filename.format('6')
        elif frame_number == 1:
            frame_filename = frame_filename.format('2')
        return frame_filename 
    
    def _get_displacement_file_name(self):
        displacement_filename = 'disp2.pgm'
        return displacement_filename
    
    def _get_dataset_directory(self):
        return os.path.join('datasets','Middlebury')
