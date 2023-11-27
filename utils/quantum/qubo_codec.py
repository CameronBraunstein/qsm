from energy_model.data_model import get_data_model
from energy_model.regularization_model import get_regularization_model
from utils.quantum.displacement_calculator import get_displacement_calculator
from utils.quantum.coordinate_canonicalizer import get_coordinate_canonicalizer
import numpy as np

class QUBOCodec:
    def __init__(self,args):
        self.data_model = get_data_model(args)
        self.regularization_model = get_regularization_model(args)
        self.displacement_calculator = get_displacement_calculator(args)
        self.coordinate_canonicalizer = get_coordinate_canonicalizer(args)

    def initialize_model_at_step(self,step):
        self.data_model.initialize_model_at_step(step)
        self.regularization_model.initialize_model_at_step(step)
        self.displacement_calculator.initialize_model_at_step(step)

    def set_bundle(self,bundle):
        self.height_range, self.width_range = bundle
        self.coordinate_canonicalizer.set_bundle(bundle)


    def create_displacements_from_response(self,response):
        displacements = np.zeros((len(self.height_range),len(self.width_range)))
        for base_y in self.height_range:
            for base_x in self.width_range:
                displacement = self._get_displacement_at_coordinates(base_y,base_x,response)

                canonical_y,canonical_x = self.coordinate_canonicalizer.canonicalize_coordinates(base_y,base_x)

                displacements[canonical_y,canonical_x] = displacement
        return displacements


    def _is_point_in_bundle(self,y,x):
        is_point_in_tile = True
        if not (y in self.height_range):
            is_point_in_tile = False
        if not (x in self.width_range):
            is_point_in_tile = False
        return is_point_in_tile

    def get_matrix_entry(self,Q,var_0,var_1):
        key = self._get_key(var_0,var_1)
        if key in Q.keys():
            value = Q[key]
        else:
            value = 0
        return value

    def _add_to_QUBO_matrix(self,Q, encoded_string_0,encoded_string_1,cost):
        key = self._get_key(encoded_string_0,encoded_string_1)
        if key in Q.keys():
            Q[key] = Q[key]+cost
        else:
            Q[key] = cost

    #Lexographic ordering
    def _get_key(self,encoded_string_0, encoded_string_1):
        if encoded_string_0 < encoded_string_1:
            key = (encoded_string_0 , encoded_string_1)
        else:
            key = (encoded_string_1 , encoded_string_0)
        return key







    
    

