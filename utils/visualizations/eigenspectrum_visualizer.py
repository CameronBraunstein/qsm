import numpy as np

def _get_pauli_z_matrix():
    z = np.zeros((2,2))
    z[0,0] = 1
    z[1,1] = -1
    return z

def _get_pauli_x_matrix():
    x = np.zeros((2,2))
    x[0,1] = 1
    x[1,0] = 1
    return x

def _get_matrix_dimension(size):
    return  2**(max(0,size))

def _compute_identity(size):
    identity_size = _get_matrix_dimension(size)
    i = np.identity(identity_size)
    return i

def _compute_tensor_product_at_indices(indices,max_index,pauli_matrix):
    indices = sorted(set(indices))
    t = _compute_identity(0)
    previous_index = -1
    for i in indices:
        identity = _compute_identity(i-previous_index-1)
        t = np.kron(t,identity)
        t = np.kron(t, pauli_matrix)
        previous_index = i
    identity = _compute_identity(max_index-indices[-1]-1)
    t = np.kron(t,identity)
    return t

def _initialize_hamitonian(qubit_max):
    dim = _get_matrix_dimension(qubit_max)
    H = np.zeros((dim,dim))
    return H



def _compute_hamiltonian_from_ising_matrix(J,qubit_max):
    H =  _initialize_hamitonian(qubit_max)
    z = _get_pauli_z_matrix()
    for i in range(qubit_max):
        for j in range(qubit_max):
            ising_value = J[i,j]
            H_component = _compute_tensor_product_at_indices([i,j],qubit_max,z)
            H += (H_component*ising_value)
    return H


def _compute_initial_hamiltonian(qubit_max):
    H =  _initialize_hamitonian(qubit_max)
    x = _get_pauli_x_matrix()
    for i in range(qubit_max):
        H_component =  _compute_tensor_product_at_indices([i],qubit_max,x)
        H += H_component
    return H

def _compute_eigenspectra_over_times(J,qubit_max,times):
    H_initial = _compute_initial_hamiltonian(qubit_max)
    H_final = _compute_hamiltonian_from_ising_matrix(J,qubit_max)
    eigenspectra = np.zeros((len(times),H_initial.shape[0]))
    for i in range(len(times)):
        t = times[i]
        H_t = H_initial*(1-t) + H_final*t
        eigenvalues = np.linalg.eigvals(H_t)
        eigenspectra[i,:] = np.sort(eigenvalues)
    return eigenspectra
