from utils.quantum.qubo_codec import QUBOCodec

class OneHotQUBOCodec(QUBOCodec):
    def __init__(self,args):
        super().__init__(args)
        self.constraint_weight = args.constraint_weight
        self.use_non_granular_constraints = args.use_non_granular_constraints
        self.epsilon = 0.01

    def _get_displacement_at_coordinates(self,y,x,response):
        displacement_range, canonical_displacement_range = self.displacement_calculator.get_displacement_and_canonical_range(y,x)
        canonical_displacement = 0

        canonical_y,canonical_x = self.coordinate_canonicalizer.canonicalize_coordinates(y,x)
        for i in canonical_displacement_range:
            key = _encode_flow_qubit_label(canonical_y,canonical_x,i) 
            encoded_response = response[key]
            if encoded_response == 1:
                canonical_displacement = i

        displacement = displacement_range[canonical_displacement]
        return displacement


    def add_data_costs(self, Q, base_y, base_x):
        canonical_base_y, canonical_base_x = self.coordinate_canonicalizer.canonicalize_coordinates(base_y,base_x)
        chi = self._get_chi(base_y,base_x)
        max_negative_theta = self._get_max_negative_theta(base_y,base_x)
        displacement_range, canonical_displacement_range = self.displacement_calculator.get_displacement_and_canonical_range(base_y, base_x)
        for displacement_i, canonical_displacement_i in zip(displacement_range, canonical_displacement_range):
            
            data_energy = self.data_model.get_data_energy(base_y,base_x,displacement_i)
            
            p_i = _encode_flow_qubit_label(canonical_base_y,canonical_base_x,canonical_displacement_i)
            #add the data energy
            self._add_to_QUBO_matrix(Q, p_i,p_i,data_energy)

            #add the constraint to guarantee at least one candidate is selected
            self._add_to_QUBO_matrix(Q, p_i,p_i,-chi*self.constraint_weight)

            #add the constraints to guarantee at most one candidate is selected
            for displacement_j, canonical_displacement_j in zip(displacement_range, canonical_displacement_range):
                if canonical_displacement_j <= canonical_displacement_i:
                    continue

                p_j = _encode_flow_qubit_label(canonical_base_y,canonical_base_x,canonical_displacement_j)
                
                if self.use_non_granular_constraints:
                    self._add_to_QUBO_matrix(Q, p_i,p_j,max(chi, max_negative_theta)*self.constraint_weight)
                else:
                    theta = self._get_theta(base_y,base_x,displacement_i,displacement_j)
                    self._add_to_QUBO_matrix(Q, p_i,p_j,(chi-theta)*self.constraint_weight)

    def _get_max_negative_theta(self,base_y,base_x):
        max_negative_theta = float('-inf')
        displacement_range, canonical_displacement_range = self.displacement_calculator.get_displacement_and_canonical_range(base_y, base_x)
        for displacement_i, canonical_displacement_i in zip(displacement_range, canonical_displacement_range):
            for displacement_j, canonical_displacement_j in zip(displacement_range, canonical_displacement_range):
                if canonical_displacement_j <= canonical_displacement_i:
                    continue
                theta = self._get_theta(base_y,base_x,displacement_i,displacement_j)
                max_negative_theta = max(max_negative_theta, -theta)
        return max_negative_theta

    def _get_gamma(self,base_y,base_x,base_displacement, neighbor_y,neighbor_x):
        max_val = float('-inf')
        for neighbor_displacement in self.displacement_calculator.get_displacement_range(neighbor_y,neighbor_x):
            regularization_energy = self.regularization_model.get_regularization_energy(base_displacement,neighbor_displacement,base_y,base_x,neighbor_y,neighbor_x)
            max_val = regularization_energy if regularization_energy > max_val else max_val
        return max_val

    def _get_chi(self,base_y,base_x):
        min_val = float('inf')
        offsets = [[0,1],[0,-1],[1,0],[-1,0]]

        for base_displacement in self.displacement_calculator.get_displacement_range(base_y,base_x):
            worst_penalty_total = 0
            worst_penalty_total += self.data_model.get_data_energy(base_y,base_x,base_displacement)
            for offset in offsets:
                neighbor_y = base_y + offset[0]
                neighbor_x = base_x + offset[1]
                if self._is_point_in_bundle(neighbor_y,neighbor_x):
                    worst_penalty_total += max(0,self._get_gamma(base_y,base_x,base_displacement,neighbor_y,neighbor_x))
            min_val = worst_penalty_total if worst_penalty_total < min_val else min_val
        chi = max(0,min_val + self.epsilon)
        return chi
    
    def _get_zeta(self,base_y,base_x,base_displacement):
        zeta = 0
        offsets = [[0,1],[0,-1],[1,0],[-1,0]]
        for offset in offsets:
            neighbor_y = base_y + offset[0]
            neighbor_x = base_x + offset[1]
            if self._is_point_in_bundle(neighbor_y,neighbor_x):
                for neighbor_displacement in self.displacement_calculator.get_displacement_range(neighbor_y,neighbor_x):
                    regularization_energy  = self.regularization_model.get_regularization_energy(base_displacement,neighbor_displacement,base_y,base_x,neighbor_y,neighbor_x)
                    zeta += min(0,regularization_energy)
        return zeta

    def _get_theta(self,base_y,base_x,base_displacement_0,base_displacement_1):
        def __get_energy_decrease(base_y,base_x,base_displacement):
            data_energy = self.data_model.get_data_energy(base_y,base_x,base_displacement)
            zeta = self._get_zeta(base_y,base_x,base_displacement)
            energy_decrease = data_energy + zeta
            return energy_decrease
        energy_decrease_0 = __get_energy_decrease(base_y,base_x,base_displacement_0)
        energy_decrease_1 = __get_energy_decrease(base_y,base_x,base_displacement_1)
        theta = min(0, energy_decrease_0 - self.epsilon, energy_decrease_1 - self.epsilon)
        return theta
    
    def add_regularization_costs(self,Q, base_y, base_x, neighbor_y, neighbor_x):
        canonical_base_y, canonical_base_x = self.coordinate_canonicalizer.canonicalize_coordinates(base_y,base_x)
        canonical_neighbor_y, canonical_neighbor_x = self.coordinate_canonicalizer.canonicalize_coordinates(neighbor_y,neighbor_x)

        base_displacement_range, canonical_base_displacement_range = self.displacement_calculator.get_displacement_and_canonical_range(base_y,base_x)
        for base_displacement,canonical_base_displacement in zip(base_displacement_range, canonical_base_displacement_range) :
            p = _encode_flow_qubit_label(canonical_base_y,canonical_base_x,canonical_base_displacement)

            neighbor_displacement_range , canonical_neighbor_displacement_range = self.displacement_calculator.get_displacement_and_canonical_range(neighbor_y, neighbor_x)
            for neighbor_displacement, canonical_neighbor_displacement in zip(neighbor_displacement_range , canonical_neighbor_displacement_range) :
                q = _encode_flow_qubit_label(canonical_neighbor_y,canonical_neighbor_x,canonical_neighbor_displacement)

                regularization_energy = self.regularization_model.get_regularization_energy(base_displacement,neighbor_displacement,base_y,base_x,neighbor_y,neighbor_x)
                self._add_to_QUBO_matrix(Q, p,q,regularization_energy)
    

def _encode_flow_qubit_label(y,x,ranking):
    return 'p({},{})_{}'.format(x,y,ranking)