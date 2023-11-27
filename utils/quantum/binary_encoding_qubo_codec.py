from utils.quantum.qubo_codec import QUBOCodec
from stereo_matchers.matching_steps_handler import get_matching_steps_handler
import itertools
import math

class BinaryEncodingQUBOCodec(QUBOCodec):
    def __init__(self,args):
        super().__init__(args)
        self.matching_steps_handler = get_matching_steps_handler(args)

    def initialize_model_at_step(self,step):
        super().initialize_model_at_step(step)
        num_candidates = self.matching_steps_handler.get_num_candidates_at_step(step)
        self.encoding_length = int(math.log2(num_candidates))

    def _get_displacement_at_coordinates(self,y,x,response):
        displacement_range = self.displacement_calculator.get_displacement_range(y,x)
        canonical_displacement = 0
        canonical_y,canonical_x = self.coordinate_canonicalizer.canonicalize_coordinates(y,x)
        for i in range(self.encoding_length):
            key = _encode_displacement_qubit_label(canonical_y,canonical_x,i) 
            encoded_response = response[key]
            canonical_displacement += (2**i)*encoded_response

        displacement = displacement_range[canonical_displacement]
        return displacement
    
    #DATA COST FUNCTIONS
    def add_data_costs(self,Q,base_y,base_x):
        data_energies = self._get_binary_encoded_data_energies(base_y,base_x)
        polynomial_coefficients = _get_coefficients_from_energies(data_energies,self.encoding_length)
        self._encode_data_polynomial_coefficients_into_QUBO(Q,polynomial_coefficients,base_y,base_x)

    def _get_binary_encoded_data_energies(self,base_y,base_x):
        energies = {}
        displacement_range = self.displacement_calculator.get_displacement_range(base_y,base_x)
        for combination in itertools.product([0, 1], repeat=self.encoding_length):
            binary_string = ''.join(map(str, combination))
            canonical_displacement = int(binary_string, 2)
            displacement = displacement_range[canonical_displacement]
            energy = self.data_model.get_data_energy(base_y,base_x,displacement)
            energies[binary_string] = energy
        return energies

    def _encode_data_polynomial_coefficients_into_QUBO(self,Q,polynomial_coefficients,base_y,base_x):
        def __get_encoding_qubit(indices,i,encoding_length,canonical_base_y,canonical_base_x):
            binary_digit = encoding_length - indices[i] -1
            encoding_qubit = _encode_displacement_qubit_label(canonical_base_y,canonical_base_x,binary_digit)
            return encoding_qubit
        
        canonical_base_y, canonical_base_x = self.coordinate_canonicalizer.canonicalize_coordinates(base_y,base_x)

        for combination in itertools.product([0, 1], repeat=self.encoding_length):
            binary_string = ''.join(map(str, combination))
            coefficient = polynomial_coefficients[binary_string]
            indices = _get_indices_of_ones(combination)
            if len(indices) == 0:
                continue
            elif len(indices) == 1:
                p = __get_encoding_qubit(indices,0,self.encoding_length,canonical_base_y,canonical_base_x)
                q = __get_encoding_qubit(indices,0,self.encoding_length,canonical_base_y,canonical_base_x)
                self._add_to_QUBO_matrix(Q,p,q,coefficient)
            elif len(indices) == 2:
                p = __get_encoding_qubit(indices,0,self.encoding_length,canonical_base_y,canonical_base_x)
                q = __get_encoding_qubit(indices,1,self.encoding_length,canonical_base_y,canonical_base_x)
                self._add_to_QUBO_matrix(Q,p,q,coefficient)
            else:
                raise Exception('Not handling larger encoding schemes currently')
    
    #REGULARIZATION COST FUNCTIONS
    def add_regularization_costs(self, Q, base_y, base_x, neighbor_y, neighbor_x):
        regularization_energies = self._get_binary_encoded_regularization_energies(base_y,base_x,neighbor_y, neighbor_x)
        polynomial_coefficients = _get_coefficients_from_energies(regularization_energies,2*self.encoding_length)
        self._encode_regularization_polynomial_coefficients_into_QUBO(Q,polynomial_coefficients,base_y,base_x,neighbor_y, neighbor_x)

    def _get_binary_encoded_regularization_energies(self,base_y,base_x,neighbor_y, neighbor_x):
        energies = {}
        base_displacement_range = self.displacement_calculator.get_displacement_range(base_y,base_x)
        neighbor_displacement_range = self.displacement_calculator.get_displacement_range(neighbor_y,neighbor_x)
        for base_combination in itertools.product([0, 1], repeat=self.encoding_length):
            base_binary_string = ''.join(map(str, base_combination))
            canonical_base_displacement = int(base_binary_string, 2)
            base_displacement = base_displacement_range[canonical_base_displacement]
            for neighbor_combination in itertools.product([0, 1], repeat=self.encoding_length):
                neighbor_binary_string = ''.join(map(str, neighbor_combination))
                canonical_neighbor_displacement = int(neighbor_binary_string, 2)
                neighbor_displacement = neighbor_displacement_range[canonical_neighbor_displacement]
                energy = self.regularization_model.get_regularization_energy(base_displacement,neighbor_displacement,base_y,base_x,neighbor_y,neighbor_x)
                energies[base_binary_string+neighbor_binary_string] = energy
        return energies
    
    def _encode_regularization_polynomial_coefficients_into_QUBO(self,Q,polynomial_coefficients,base_y,base_x,neighbor_y, neighbor_x):
        def get_encoding_qubit(indices,i,encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x):
            index = indices[i]
            if index>=encoding_length:
                binary_digit = 2*encoding_length - index -1
                encoding_qubit = _encode_displacement_qubit_label(canonical_neighbor_y,canonical_neighbor_x,binary_digit)
            else:
                binary_digit = encoding_length - index -1
                encoding_qubit = _encode_displacement_qubit_label(canonical_base_y,canonical_base_x,binary_digit)
            return encoding_qubit
        canonical_base_y, canonical_base_x = self.coordinate_canonicalizer.canonicalize_coordinates(base_y,base_x)
        canonical_neighbor_y, canonical_neighbor_x = self.coordinate_canonicalizer.canonicalize_coordinates(neighbor_y,neighbor_x)

        for combination in itertools.product([0, 1], repeat=self.encoding_length*2):
            binary_string = ''.join(map(str, combination))
            coefficient = polynomial_coefficients[binary_string]
            indices = _get_indices_of_ones(combination)
            if len(indices) == 0:
                continue
            elif len(indices) == 1:
                p = get_encoding_qubit(indices,0,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                q = get_encoding_qubit(indices,0,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                self._add_to_QUBO_matrix(Q,p,q,coefficient)
            elif len(indices) == 2:
                p = get_encoding_qubit(indices,0,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                q = get_encoding_qubit(indices,1,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                self._add_to_QUBO_matrix(Q,p,q,coefficient)
            elif len(indices) == 3:
                p = get_encoding_qubit(indices,0,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                q = get_encoding_qubit(indices,1,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                r = get_encoding_qubit(indices,2,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                aux_qubit = _encode_aux_qubit_label(canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x,binary_string)

                if coefficient < 0: #Eq 5) of Ishikawa et al:
                    self._add_to_QUBO_matrix(Q,p,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,q,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,r,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,aux_qubit,aux_qubit,-2*coefficient)
                elif coefficient > 0:  #Eq 6) of Ishikawa et al:
                    self._add_to_QUBO_matrix(Q,p,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,q,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,r,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,aux_qubit,aux_qubit,-1*coefficient)

                    self._add_to_QUBO_matrix(Q,p,q,coefficient)
                    self._add_to_QUBO_matrix(Q,q,r,coefficient)
                    self._add_to_QUBO_matrix(Q,r,p,coefficient)

                    self._add_to_QUBO_matrix(Q,p,p,-1*coefficient)
                    self._add_to_QUBO_matrix(Q,q,q,-1*coefficient)
                    self._add_to_QUBO_matrix(Q,r,r,-1*coefficient)
            elif len(indices) == 4:
                p = get_encoding_qubit(indices,0,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                q = get_encoding_qubit(indices,1,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                r = get_encoding_qubit(indices,2,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                s = get_encoding_qubit(indices,3,self.encoding_length,canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x)
                aux_qubit = _encode_aux_qubit_label(canonical_base_y,canonical_base_x,canonical_neighbor_y,canonical_neighbor_x,binary_string)
                if coefficient < 0: #pg. 1237 of Ishikawa et al 
                    self._add_to_QUBO_matrix(Q,p,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,q,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,r,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,s,aux_qubit,coefficient)
                    self._add_to_QUBO_matrix(Q,aux_qubit,aux_qubit,-3*coefficient)
                elif coefficient > 0:
                    self._add_to_QUBO_matrix(Q,p,aux_qubit,-2*coefficient)
                    self._add_to_QUBO_matrix(Q,q,aux_qubit,-2*coefficient)
                    self._add_to_QUBO_matrix(Q,r,aux_qubit,-2*coefficient)
                    self._add_to_QUBO_matrix(Q,s,aux_qubit,-2*coefficient)
                    self._add_to_QUBO_matrix(Q,aux_qubit,aux_qubit,3*coefficient)

                    self._add_to_QUBO_matrix(Q,p,q,coefficient)
                    self._add_to_QUBO_matrix(Q,p,r,coefficient)
                    self._add_to_QUBO_matrix(Q,p,s,coefficient)
                    self._add_to_QUBO_matrix(Q,q,r,coefficient)
                    self._add_to_QUBO_matrix(Q,q,s,coefficient)
                    self._add_to_QUBO_matrix(Q,r,s,coefficient)
            else:
                raise Exception('Not handling larger encoding schemes currently')

#UTILIES
def _encode_displacement_qubit_label(y,x,binary_digit):
    return 'p({},{})_{}'.format(x,y,binary_digit)

def _encode_aux_qubit_label(base_y,base_x,neighbor_y,neighbor_x,binary_string):
    return 'aux(({},{})({},{}))_{}'.format(base_y,base_x,neighbor_y,neighbor_x,binary_string)

def _get_indices_of_ones(indexable_tuple):
    indices = []
    index = -1  # Start index before the first occurrence
    for _ in range(indexable_tuple.count(1)):
        index = indexable_tuple.index(1, index + 1)
        indices.append(index)
    return indices

def _get_sub_ordered_strings(binary_string,encoding_length):
    sub_ordered_strings = []
    for combination in itertools.product([0, 1], repeat=encoding_length):
        potential_string = ''.join(map(str, combination))
        is_sub_string = True
        for i in range(len(potential_string)):
            if potential_string[i] > binary_string[i]:
                is_sub_string = False
                break
        if is_sub_string:
            sub_ordered_strings.append(potential_string)
    return sub_ordered_strings

def _get_coefficients_from_energies(energies,encoding_length):
    a = {}
    for combination in itertools.product([0, 1], repeat=encoding_length):
        binary_string = ''.join(map(str, combination))
        binary_string_ones_count = binary_string.count('1')
        coefficient = 0
        sub_ordered_strings = _get_sub_ordered_strings(binary_string,encoding_length)
        for sub_binary_string in sub_ordered_strings:
            sub_binary_string_ones_count = sub_binary_string.count('1')
            coefficient += (energies[sub_binary_string]*(-1)**(binary_string_ones_count-sub_binary_string_ones_count))
        a[binary_string] = coefficient
    return a