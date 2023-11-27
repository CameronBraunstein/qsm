
from utils.quantum.qubo_solver_interface import get_qubo_solver_interface
from stereo_matchers.bundler import get_bundler
import matplotlib.pyplot as plt
import numpy as np
from utils.visualizations.eigenspectrum_visualizer import _compute_eigenspectra_over_times

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def get_qubo_matrix_visualizer(args):
    qmv = QuboMatrixVisualizer(args)
    return qmv


def _make_dict_key_tuples_into_ordered_list(dictionary):
    s = set()
    for key in dictionary.keys():
        e_0 = key[0]
        e_1 = key[1]
        s.add(e_0)
        s.add(e_1)
    sorted_list = sorted(s)
    return sorted_list

def _build_matrix_from_entry_list(dictionary,entry_list):
    list_length = len(entry_list)
    m = np.zeros((list_length,list_length))
    for i in range(list_length):
        for j in range(i,list_length):
            key = (entry_list[i],entry_list[j])
            if key in dictionary:
                value = dictionary[key] /2
                m[i,j] += value 
                m[j,i] += value 
    return m

def _qubo_matrix_to_ising_matrix(Q):
    row, col = Q.shape
    J = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            Q_value = Q[i,j]
            if i == j:
                J[i,j] += Q_value/2
            else:
                J[i,j] += Q_value/4
                J[i,i] += Q_value/4
                J[j,j] += Q_value/4
    return J

def _qubo_dictionary_into_matrix(Q_dictonary):
    qubit_list = _make_dict_key_tuples_into_ordered_list(Q_dictonary)
    Q_matrix = _build_matrix_from_entry_list(Q_dictonary,qubit_list)
    return Q_matrix

def _get_label(i):
    if i == 0:
        label = 'Ground State'
    elif i == 1:
        label = 'Next Lowest State'
    else:
        label = 'Eigenvalue {}'.format(i)
    return label

def _rescale_matrix_in_bounds(J):
    b = _compute_bias_scale_factor(J)
    c = _compute_coupler_scale_factor(J)
    J *= min(b,c)
    return J

def _compute_bias_scale_factor(J):
    bias_range_max = 2
    max_diagonal = np.max(np.abs(J[np.where(np.eye(J.shape[0], dtype=bool))]))
    bias_scale_factor = bias_range_max/max_diagonal
    return bias_scale_factor


def _compute_coupler_scale_factor(J):
    coupler_range_max = 1
    max_off_diagonal = np.max(np.abs(J[np.where(~np.eye(J.shape[0], dtype=bool))]))
    coupler_scale_factor = coupler_range_max/max_off_diagonal
    return coupler_scale_factor



def _get_submatrix(Q):
    base_y = 333
    base_x = 333 #Subfig 1

    # base_y = 26
    # base_x = 127

    base_y = 48
    base_x = 414

    
    offset = 76
    Q = Q[base_y:base_y + offset, base_x:base_x + offset]
    return Q

def _crop_matrix(J,qubit_max):
    offset = 15

    J = J[offset:qubit_max + offset,offset:qubit_max + offset]
    return J


def _get_nth_color(n):
    default_color_cycle = plt.rcParams['axes.prop_cycle']
    nth_default_color = default_color_cycle.by_key()['color'][n]
    return nth_default_color


def _add_eigenspectrum_to_axis(axis,times,eigenspectrum,spectral_lines_to_show):
    for i in range(spectral_lines_to_show):
        spectral_line = eigenspectrum[:,i]
        label=_get_label(i)
        color = _get_nth_color(i)
        axis.plot(times,spectral_line,label=label,color=color)


class QuboMatrixVisualizer:
    def __init__(self,args):
        self.qubo_solver_interface = get_qubo_solver_interface(args)
        self.qubo_solver_interface.initialize_model_at_step(0)
        self.bundler = get_bundler(args)

    def display_qubo_matrix(self):
        J = self._get_ising_matrix()
        #J = _get_submatrix(J)
        self._plot_matrix(J)

    def _plot_matrix(self,J):
        print(np.max(J))
        plt.axis('off')
        plt.imshow(J, cmap='jet', vmin=0, vmax=2, interpolation='none')
        color_bar = plt.colorbar()
        plt.show()

    def _get_ising_matrix(self,crop_matrix=False,qubit_max=-1):
        bundles = self.bundler.get_bundles_at_step(0)
        bundle = bundles[0]
        Q_dictionary = self.qubo_solver_interface._create_QUBO_for_bundle(bundle)
        Q = _qubo_dictionary_into_matrix(Q_dictionary)
        J = _qubo_matrix_to_ising_matrix(Q)
        if crop_matrix:
            J = _crop_matrix(J,qubit_max)
        J = _rescale_matrix_in_bounds(J)
        print(J)
        return J

    def display_eigenspectrum(self):
        qubit_max = 10 #10 #12 is about the limit
        J = self._get_ising_matrix(crop_matrix=True,qubit_max=qubit_max)
        
        times = np.linspace(0, 1, num=20)
        es = _compute_eigenspectra_over_times(J,qubit_max,times)
        self._plot_eigenspectrum(es,times)
        print(np.min(es[:,1]-es[:,0]))

    def _plot_eigenspectrum(self,eigenspectrum,times):
        spectral_lines_to_show = 2
        fig = plt.figure()
        ax = plt.axes()
        _add_eigenspectrum_to_axis(ax,times,eigenspectrum,spectral_lines_to_show)

        ax.set_xlabel('s')
        ax.set_ylabel('Eigenvalues')
        ax.legend()

        axins = zoomed_inset_axes(ax, zoom=1.5, loc=8, borderpad=2) # zoom = 2
        _add_eigenspectrum_to_axis(axins,times,eigenspectrum,spectral_lines_to_show)
        
        axins.set_xlim(0.9, 1.0)
        axins.set_ylim(-12.5, -12)

        mark_inset(ax, axins, loc1=2, loc2=4)
        plt.show()

