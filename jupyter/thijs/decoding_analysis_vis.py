## Analysis & visualisation code for decoding analysis
## Thijs van der Plas

from json import decoder
import os, sys, pickle, copy
import loadpaths

user_paths_dict = loadpaths.loadpaths()
pkl_folder = user_paths_dict['pkl_folder']
vape_path = user_paths_dict['vape_path']
s2p_path = user_paths_dict['s2p_path']

## Old data paths, from 'rob_setup_notebook.ipynb'
# qnap_data_path = '/home/rlees/mnt/qnap/Data' # for Ubuntu
# qnap_path = qnap_data_path[:-5]
# pkl_folder = os.path.join(qnap_path, 'pkl_files')
# master_path = os.path.join(qnap_path, 'master_pkl', 'master_obj.pkl')
# fig_save_path = os.path.join(qnap_path, 'Analysis', 'Figures')
# stam_save_path = os.path.join(qnap_path, 'Analysis', 'STA_movies')
# s2_borders_path = os.path.join(qnap_path, 'Analysis', 'S2_borders')

from select import select
from urllib import response
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_palette('colorblind')
import scipy.optimize

import pandas as pd
sys.path.append(vape_path)
# from utils.utils_funcs import correct_s2p_combined 
import xarray as xr
import scipy
from profilehooks import profile, timecall
import sklearn.discriminant_analysis, sklearn.model_selection, sklearn.decomposition
from tqdm import tqdm
from statsmodels.stats import multitest
sys.path.append('/home/tplas/repos/reproducible_figures/scripts/')
import rep_fig_vis as rfv 

## From Vape:
# import utils.ia_funcs as ia 
# import utils.utils_funcs as uf

sess_type_dict = {'sens': 'sensory_2sec_test',
                  'proj': 'projection_2sec_test'}

sys.path.append(os.path.join(vape_path, 'my_suite2p')) # to import ops from settings.py in that folder
if s2p_path is not None:
    sys.path.append(s2p_path)
# import suite2p

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colour_tt_dict = {'sensory': colors[1],
                  'random': colors[0],
                  'projecting': colors[4],
                  'non_projecting': colors[5],
                  'whisker': colors[6],
                  'sham': colors[2]}
label_tt_dict = {'sensory': 'Sensory',
                  'random': 'Random',
                  'projecting': 'Projecting',
                  'non_projecting': 'Non-projecting',
                  'whisker': 'Whisker',
                  'sham': 'Sham'}                 

for tmp_dict in [colour_tt_dict, label_tt_dict]:
    tmp_dict['sham_sens'] = tmp_dict['sham']
    tmp_dict['sham_proj'] = tmp_dict['sham']
for k, v in label_tt_dict.items():
    colour_tt_dict[v] = colour_tt_dict[k]


def get_session_names(pkl_folder=pkl_folder,
                      sess_type='sens'):
    pkl_folder_path = os.path.join(pkl_folder, sess_type_dict[sess_type])
    list_session_names = os.listdir(pkl_folder_path)
    exclude_list = ['2020-09-09_RL100.pkl', '2020-09-15_RL102.pkl']  # should be excluded (Rob said)
    list_session_names = [x for x in list_session_names if x not in exclude_list]
    assert len(list_session_names) == 6, 'less or more than 6 sessions found'
    return list_session_names

def load_session(pkl_folder=pkl_folder, 
                 sess_type='sens',
                 session_name=None,
                 session_id=0, verbose=1):
    if session_name is None and type(session_id) == int:
        list_session_names = get_session_names(pkl_folder=pkl_folder,
                    sess_type=sess_type)
        assert session_id < len(list_session_names)
        session_name = list_session_names[session_id]
        if verbose > 0:
            print('session name :', session_name)

    pkl_path = os.path.join(pkl_folder, sess_type_dict[sess_type], session_name)
    with open(pkl_path, 'rb') as f:
        ses_obj = pickle.load(f)

    if verbose > 1:
        # Show all attributes in ses_obj or exp_objs
        print('Session object attributes')
        for key, value in vars(ses_obj).items():
            print(key, type(value))

        print('\nExperimental object attributes')
        for key, value in vars(ses_obj.spont).items():
            print(key, type(value))

    return ses_obj, session_name 

class SimpleSession():
    """Class that stores session object in format that's easier to access for decoding analysis"""
    def __init__(self, sess_type='sens', session_id=0, verbose=1,
                 shuffle_trial_labels=False, shuffle_timepoints=False, 
                 shuffle_all_data=False, prestim_baseline=True,
                 bool_filter_neurons=True):
        self.sess_type = sess_type
        self.session_id = session_id
        self.verbose = verbose 
        self.shuffle_timepoints = shuffle_timepoints
        self.shuffle_trial_labels = shuffle_trial_labels
        self.shuffle_all_data = shuffle_all_data
        self.bool_filter_neurons = bool_filter_neurons
        self.prestim_baseline = prestim_baseline

        self.SesObj, self.session_name = load_session(sess_type=self.sess_type,
                                session_id=self.session_id, verbose=self.verbose)
        self.session_name_readable = self.session_name.rstrip('.pkl')
        
        if sess_type == 'sens':
            self._list_tt_original = ['photostim_s', 'photostim_r', 'spont', 'whisker_stim']
        elif sess_type == 'proj':
            self._list_tt_original = ['photostim_s', 'photostim_r', 'spont']

        self.sorted_inds_neurons = None  # default 

        self.filter_neurons(filter_max_abs_dff=self.bool_filter_neurons)
        self.create_data_objects()
        self.create_nonbaselinsed_alltrials_object()
        self.create_full_dataset(prestim_baseline=self.prestim_baseline)
        # self.sort_neurons(sorting_method='normal')

    def filter_neurons(self, filter_max_abs_dff=True):
        """Filter neurons that should not be used in any analysis. These are:

        - Neurons with max(abs(DF/F)) > 10 for any trial type
        """
        ## Find which neurons to keep:
        for i_tt, tt in enumerate(self._list_tt_original):
            data_obj = getattr(self.SesObj, tt)
            if i_tt == 0:  # make mat that stores max values 
                n_neurons = data_obj.all_trials[0].shape[0]
                mv_neurons_mat = np.zeros((n_neurons, len(self._list_tt_original)))
            else:
                assert n_neurons == data_obj.all_trials[0].shape[0]

            mv_neurons_mat[:, i_tt] = np.max(np.abs(data_obj.all_trials[0]), (1, 2)) # get max values

        if self.verbose > 1:
            print(mv_neurons_mat)
        mv_neurons_mat = np.max(mv_neurons_mat, 1)  # max across tt
        if filter_max_abs_dff:
            filter_neurons_arr = mv_neurons_mat > 10 
        else:
            filter_neurons_arr = np.zeros(n_neurons, dtype=bool)
        mask_neurons = np.logical_not(filter_neurons_arr)  # flip (True = keep)
        if self.verbose > 0:
            print(f'Excluded {np.sum(filter_neurons_arr)} out of {len(filter_neurons_arr)} neurons')

        self._n_neurons_original = len(filter_neurons_arr)
        self._n_neurons_removed = np.sum(filter_neurons_arr)
        self._mask_neurons_keep = mask_neurons  
        # assert type(self._mask_neurons_keep[0]) == np.bool, type(self._mask_neurons_keep)
        self.n_neurons = np.sum(mask_neurons)
        
    def create_data_objects(self):
        """Put original data objects of self.SesObj in more convenient format that incorporates neuron filtering"""
        self.all_trials, self.targeted_cells = {}, {}
        self.n_targeted_cells, self._n_targeted_cells_original = {}, {} 
        self.cell_s1, self.cell_s2, self.cell_id= {}, {}, {}
        self.stim_dur, self.target_coords = {}, {}

        ## concatenating all trials
        """trials are done in block format; 100 trials per type. 
        Do in 4 blocks (no interleaving). Whisker, sham, PS1 PS2. (swap last two every other experiment). 
        ITI 15 seconds, (but data is defined as 12 second trials)"""
        if self.sess_type == 'sens':
            self._key_dict = {'photostim_s': 'sensory', 
                              'photostim_r': 'random',
                              'whisker_stim': 'whisker',
                              'spont': 'sham'}
        elif self.sess_type == 'proj':
            self._key_dict = {'photostim_s': 'projecting', 
                              'photostim_r': 'non_projecting',
                              'spont': 'sham'}
        self.list_tt = [self._key_dict[tt] for tt in self._list_tt_original]
        for i_tt, tt in enumerate(self._list_tt_original):
            # print(tt)
            assert len(getattr(self.SesObj, tt).all_trials) == 1, 'more than 1 imaging plane detected ??'
            self.all_trials[self._key_dict[tt]] = getattr(self.SesObj, tt).all_trials[0][self._mask_neurons_keep, :, :]
            ## You could do the same for:
                ## all_amplitudes
                ## stats
                ## prob
                
            if i_tt == 0:
                self.time_array = getattr(self.SesObj, tt).time
                self.n_timepoints = len(self.time_array)
                self.frame_array = np.arange(self.n_timepoints)
                if self.sess_type == 'sens':
                    assumed_artefact_tps = np.array([31, 32, 33, 34, 35, 36])  # I'm just hard-coding this check to be sure
                elif self.sess_type == 'proj':
                    assumed_artefact_tps = np.array([31, 32, 33, 34])
                self.artefact_bool = np.zeros(self.n_timepoints, dtype='bool')
                self.artefact_bool[assumed_artefact_tps] = True  # this will be double checked with DF/F data in create_full_dataset()
                assert self.n_timepoints == 182, 'length time axis not as expected..?'
                assert self.n_timepoints == self.all_trials[self._key_dict[tt]].shape[1]
            else:
                assert np.isclose(self.time_array, getattr(self.SesObj, tt).time, atol=1e-6).all(), 'time arrays not equal across tts?'
            
            if tt != 'whisker_stim':  # targets not defined for whisker stim tt
                self.targeted_cells[self._key_dict[tt]] = getattr(self.SesObj, tt).targeted_cells[self._mask_neurons_keep]
                self.n_targeted_cells[self._key_dict[tt]] = np.sum(self.targeted_cells[self._key_dict[tt]])
                self._n_targeted_cells_original[self._key_dict[tt]] = np.sum(getattr(self.SesObj, tt).targeted_cells)
                self.target_coords[self._key_dict[tt]] = [x for i_x, x in enumerate(getattr(self.SesObj, tt).target_coords) if self._mask_neurons_keep[i_x]]  # note that sham == random, but of course none were stimulated on sham trials
            
            self.cell_s1[self._key_dict[tt]] = np.array(getattr(self.SesObj, tt).cell_s1[0])[self._mask_neurons_keep]
            self.cell_s2[self._key_dict[tt]] = np.array(getattr(self.SesObj, tt).cell_s2[0])[self._mask_neurons_keep]
            self.cell_id[self._key_dict[tt]] = np.array(getattr(self.SesObj, tt).cell_id[0])[self._mask_neurons_keep]

            self.stim_dur[self._key_dict[tt]] = getattr(self.SesObj, tt).stim_dur

    def create_full_dataset(self, zscore=False, prestim_baseline=True):
        if prestim_baseline:
            full_data = np.concatenate([self.all_trials[tt] for tt in self.list_tt], axis=2)  # concat across trials 
        else:
            full_data = np.concatenate([self.all_trials_nonbaselined[tt] for tt in self.list_tt], axis=2)  # concat across trials 
        assert full_data.shape[0] == self.n_neurons and full_data.shape[1] == self.n_timepoints 
        tt_arr = []
        for tt in self.list_tt:
            tt_arr += [tt] * self.all_trials[tt].shape[2]  # same shape as all_trials_nonbaselined (assert in creation function)
        tt_arr = np.array(tt_arr)
        assert len(tt_arr) == full_data.shape[2]

        if zscore:
            full_data = (full_data - full_data.mean((0, 1))) / full_data.std((0, 1))
            print('WARNING: z-scoring messes up the baselining (that average pre-stim activity =0)')
        else:
            if prestim_baseline:
                ## assert baselining is as expected
                baseline_frames = self.SesObj.spont.pre_frames  # same for any trial type
                assert np.abs(full_data[:, :baseline_frames, :].mean()) < 1e-8  # mean across neurons and pre-stim time points and trials
                assert np.max(np.abs(full_data[:, :baseline_frames, :].mean((0, 1)))) < 1e-8  # mean across neurons and pre-stim time points

        if self.shuffle_trial_labels:
            # random_inds = np.random.permutation(full_data.shape[2])
            trial_inds = np.arange(full_data.shape[2])
            # full_data = full_data[:, :, random_inds]
            # for ineuron in range(full_data.shape[0]):
            for itp in range(full_data.shape[1]):
                random_trial_inds = np.random.permutation(full_data.shape[2])
                full_data[:, itp, :] = full_data[:, itp, :][:, random_trial_inds]
            if self.verbose > 0:
                print('WARNING: trials labels are shuffled!')
        else:
            trial_inds = np.arange(full_data.shape[2])

        if self.shuffle_timepoints:
            n_trials = full_data.shape[2]
            for it in range(n_trials):
                random_tp_inds = np.random.permutation(full_data.shape[1])
                full_data[:, :, it] = full_data[:, random_tp_inds, it]
            # for it in range(n_trials):
            #     random_neuron_inds = np.random.permutation(full_data.shape[0])
            #     full_data[:, :, it] = full_data[random_neuron_inds, :, it]
                # random_tp_inds = np.random.permutation(full_data.shape[1])
                # full_data[:, :, it] = full_data[:, random_tp_inds, it]
            if self.verbose > 0:
                print('WARNING: time points are shuffled per trial')

        if self.shuffle_all_data:
            full_data_shape = full_data.shape 
            full_data = full_data.ravel()
            np.random.shuffle(full_data)
            full_data = full_data.reshape(full_data_shape)
            # full_data = np.random.randn(full_data.shape[0], full_data.shape[1], full_data.shape[2])  # totally white noise 

            print('WARNING: all data points shuffled!')

        data_arr = xr.DataArray(full_data, dims=('neuron', 'time', 'trial'),
                                coords={'neuron': np.arange(full_data.shape[0]),  #could also do cell id but this is easier i think
                                        'time': self.time_array,
                                        'trial': trial_inds})
        data_arr.time.attrs['units'] = 's'
        data_arr.neuron.attrs['units'] = '#'
        data_arr.trial.attrs['units'] = '#'
        data_arr.attrs['units'] = 'DF/F'

        target_dict = {f'targets_{tt}': ('neuron', self.targeted_cells[tt]) for tt in self.list_tt if tt not in ['sham', 'whisker']}

        ## Double check position of artefact:
        artefact_data = data_arr.data
        assert artefact_data.shape[1] == len(data_arr.time)
        artefact_data = artefact_data[:, self.artefact_bool, :]
        artefact_data = np.unique(artefact_data)
        assert len(artefact_data) == 1 and artefact_data[0] == 0, (artefact_data, data_arr)  # check that all points labeled as artefact were set to 0
        artefact_data = data_arr.data
        artefact_data = artefact_data[:, int(self.artefact_bool[0] - 1), :]
        artefact_data = np.unique(artefact_data)
        assert len(artefact_data) > 1 and artefact_data[0].mean() != 0  # check that first artefact tp was labelled correctly
        artefact_data = data_arr.data
        artefact_data = artefact_data[:, int(self.artefact_bool[-1] + 1), :]
        artefact_data = np.unique(artefact_data)
        assert len(artefact_data) > 1 and artefact_data[0].mean() != 0  # check that last artefact tp was labelled correctly
        
        data_set = xr.Dataset({**{'activity': data_arr, 
                                  'cell_s1': ('neuron', self.cell_s1['sham']),  # same for all tt anyway
                                  'cell_id': ('neuron', self.cell_id['sham']),
                                  'trial_type': ('trial', tt_arr),
                                  'frame_array': ('time', self.frame_array),
                                  'artefact_bool': ('time', self.artefact_bool)},
                               **target_dict})

        self.full_ds = data_set
        all_vars = list(dict(self.full_ds.variables).keys())                    
        self.coord_dict = {var_name: self.full_ds[var_name].dims for var_name in all_vars}  # original (squeezed) coordinates
        self.datatype_dict = {var_name: self.full_ds[var_name].dtype for var_name in all_vars}
        self.time_aggr_ds = None

    def squeeze_coords(self, tmp_dataset):
        '''Squeeze coordinates based on original coords (self.coord_dict)'''
        all_vars = list(dict(tmp_dataset.variables).keys())
            
        for var_name in all_vars:
            if var_name not in ['time', 'neuron', 'trial', 'activity']:  # leave main vars out of this
                original_var_coords = self.coord_dict[var_name]  # original (squeezed) coordinates
                if len(original_var_coords) == 1:  ## assuming this is the true squeezed # of dims
                    
                    if original_var_coords[0] == 'trial':  ## Trial type only array: (hand-coded differently because of isel, could probably soft-code?)
                        if 'time' in tmp_dataset[var_name].dims:
                            tmp_dataset = tmp_dataset.assign({var_name: tmp_dataset[var_name].isel(time=0).drop('time')}) 
                        if 'neuron' in tmp_dataset[var_name].dims:
                            tmp_dataset = tmp_dataset.assign({var_name: tmp_dataset[var_name].isel(neuron=0).drop('neuron').astype(self.datatype_dict[var_name])}) 

                    if original_var_coords[0] == 'neuron':  ## neuron-only dim 
                        if 'time' in tmp_dataset[var_name].dims:
                            tmp_dataset = tmp_dataset.assign({var_name: tmp_dataset[var_name].isel(time=0).drop('time')})  ## use kwargs to call var_name, and use tmp_dataset[var_name] because that is updated from time to trial
                        if 'trial' in tmp_dataset[var_name].dims:
                            tmp_dataset = tmp_dataset.assign({var_name: tmp_dataset[var_name].isel(trial=0).drop('trial').astype(self.datatype_dict[var_name])}) 

                    if original_var_coords[0] == 'time':  ## Time only array:
                        if 'trial' in tmp_dataset[var_name].dims:
                            tmp_dataset = tmp_dataset.assign({var_name: tmp_dataset[var_name].isel(trial=0).drop('trial')}) 
                        if 'neuron' in tmp_dataset[var_name].dims:
                            tmp_dataset = tmp_dataset.assign({var_name: tmp_dataset[var_name].isel(neuron=0).drop('neuron').astype(self.datatype_dict[var_name])}) 

                else:
                    assert len(original_var_coords) == 3, 'Also implement squeeze for 2D!'
        return tmp_dataset

    def dataset_selector(self, region=None, min_t=None, max_t=None, trial_type_list=None,
                         exclude_targets_s1=False, frame_id=None,
                         sort_neurons=False, reset_sort=False,
                         deepcopy=True):
        """## xarray indexing cheat sheet:
        self.full_ds.activity.data  # retrieve numpy array type 
        self.full_ds.activity[0, :, :]  # use np indexing 

        self.full_ds.activity.isel(time=6)  # label-based index-indexing (ie the 6th time point)
        self.full_ds.activity.sel(trial=[6, 7, 8], neuron=55)  # label-based value-indexing (NB: can ONLY use DIMENSIONS, not other variables in dataset)
        self.full_ds.activity.sel(time=5, method='nearest')  # label-based value-indexing, finding the nearest match (good with floating errors etc)
        self.full_ds.sel(time=5, method='nearest')  # isel and sel can also be used on entire ds
        
        self.full_ds.activity.where(tmp.full_ds.bool_s1, drop=True)  # index by other data array; use drop to get rid of excluded data poitns 
        self.full_ds.where(tmp.full_ds.bool_s1, drop=True)  # or on entire ds (note that it works because bool_s1 is specified to be along 'neuron' dimension)
        self.full_ds.where(tmp.full_ds.time > 3, drop=True)  # works with any bool array
        self.full_ds.where(tmp.full_ds.neuron == 50, drop=True)  # dito 
        self.full_ds.where(tmp_full_ds.trial_type.isin(['sensory', 'random']), drop=True)  # use da.isin for multipel value checking
        
        ### Somehow the order of where calls matters A LOT for the runtime:
        ## time, region, tt => 4.3s
        ## time, tt, region => 1.5s
        ## tt, region, time => long
        ## time, region => 1.7
        ## region, time => 27s

        """
        if deepcopy:
            tmp_data = self.full_ds.copy(deep=True)
        else:
            tmp_data = self.full_ds
        
        if frame_id is not None:
            '''Only implemented == operation and not limits because this is much faster. ssh '''
            assert min_t == None and max_t == None, 'you cannot select both frame # and time points'
            tmp_data = tmp_data.where(tmp_data.frame_array == frame_id, drop= True)
            tmp_data = self.squeeze_coords(tmp_dataset=tmp_data)  # xr.where() broadcasts data vars into additional dimensions, which 1) uses more RAM and 2) makes the next indexing slow. So squeeze after each where() call
        else:
            if min_t is not None:
                tmp_data = tmp_data.where(tmp_data.time >= min_t, drop=True)
                tmp_data = self.squeeze_coords(tmp_dataset=tmp_data)
            if max_t is not None:
                tmp_data = tmp_data.where(tmp_data.time <= max_t, drop=True)
                tmp_data = self.squeeze_coords(tmp_dataset=tmp_data)

        if trial_type_list is not None:
            assert type(trial_type_list) == list
            assert np.array([tt in self.list_tt for tt in trial_type_list]).all(), f'{trial_type_list} not in {self.list_tt}'
            tmp_data = tmp_data.where(tmp_data.trial_type.isin(trial_type_list), drop=True)
            tmp_data = self.squeeze_coords(tmp_dataset=tmp_data)

        if region is not None:
            assert region in ['s1', 's2']
            if region == 's1':
                tmp_data = tmp_data.where(tmp_data.cell_s1, drop=True)
                tmp_data = self.squeeze_coords(tmp_dataset=tmp_data)
            elif region == 's2':
                tmp_data = tmp_data.where(np.logical_not(tmp_data.cell_s1), drop=True)
                tmp_data = self.squeeze_coords(tmp_dataset=tmp_data)

        if exclude_targets_s1 and region == 's1':
            target_names = [xx for xx in list(dict(tmp_data.variables).keys()) if xx[:7] == 'targets']
            for tn in target_names:
                tmp_data = tmp_data.where(np.logical_not(tmp_data[tn]), drop=True)
                tmp_data = self.squeeze_coords(tmp_dataset=tmp_data)
            
        # apply sorting (to neurons, and randomizatin of trials ? )
        if sort_neurons:
            if self.sorted_inds_neurons is None or reset_sort:
                if self.verbose > 0:
                    print('sorting neurons')
                assert tmp_data.activity.data.mean(2).shape[0] == len(tmp_data.neuron)
                sorting, _ = self.sort_neurons(data=tmp_data.activity.data.mean(2), sorting_method='correlation')
                assert len(sorting) == len(tmp_data.neuron)
                assert len(sorting) == len(self.sorted_inds_neurons)
                assert len(sorting) == len(self.sorted_inds_neurons_inverse)
            tmp_data = tmp_data.assign(sorting_neuron_indices=('neuron', self.sorted_inds_neurons_inverse))  # add sorted indices on neuron dim
            tmp_data = tmp_data.sortby(tmp_data.sorting_neuron_indices)

        return tmp_data

    def sort_neurons(self, data=None, sorting_method='sum', save_sorting=True):
        """from pop off"""
        if data is None:
            print('WARNING; using pre-specified data for sorting')
            data = self.all_trials['sensory'].mean(2)  # trial averaged
        else:
            assert data.ndim == 2
            # should be neurons x times 
            # TODO; do properly with self.full_data 
        if sorting_method == 'correlation':
            sorting = opt_leaf(data, link_metric='correlation')[0]
        elif sorting_method == 'euclidean':
            sorting = opt_leaf(data, link_metric='euclidean')[0]
        elif sorting_method == 'max_pos':
            arg_max_pos = np.argmax(data, 1)
            assert len(arg_max_pos) == data.shape[0]
            sorting = np.argsort(arg_max_pos)
        elif sorting_method == 'abs_max_pos':
            arg_max_pos = np.argmax(np.abs(data), 1)
            assert len(arg_max_pos) == data.shape[0]
            sorting = np.argsort(arg_max_pos)
        elif sorting_method == 'normal':
            return np.arange(data.shape[0])
        elif sorting_method == 'amplitude':
            max_val_arr = np.max(data, 1)
            sorting = np.argsort(max_val_arr)[::-1]
        elif sorting_method == 'sum':
            sum_data = np.sum(data, 1)
            sorting = np.argsort(sum_data)[::-1]
        if self.verbose > 0:
            print(f'Neurons sorted by {sorting_method}')
        tmp_rev_sort = np.zeros_like(sorting)
        for old_ind, new_ind in enumerate(sorting):
            tmp_rev_sort[new_ind] = old_ind
        if save_sorting:
            self.sorted_inds_neurons = sorting
            self.sorted_inds_neurons_inverse = tmp_rev_sort
        return sorting, tmp_rev_sort

    def create_time_averaged_response(self, t_min=0.4, t_max=2, 
                        region=None, aggregation_method='average',
                        sort_neurons=False, subtract_pop_av=False,
                        subtract_pcs=False,
                        trial_type_list=None):
        """region: 's1', 's2', None [for both]"""
        selected_ds = self.dataset_selector(region=region, min_t=t_min, max_t=t_max,
                                    sort_neurons=False, 
                                    trial_type_list=trial_type_list)  # all trial types
        
        if subtract_pcs:
            n_timepoints_per_trial = len(selected_ds.time)
            n_trials = len(selected_ds.trial)
            selected_ds_2d = xr.concat([selected_ds.sel(trial=i_trial) for i_trial in selected_ds.trial], dim='time')  # concat trials on time axis
            # lfa = sklearn.decomposition.FactorAnalysis(n_components=2)
            lfa = sklearn.decomposition.PCA(n_components=3)
            activity_fit = selected_ds_2d.activity.data  # neurons x times
            activity_fit = activity_fit.transpose()
            pc_activity = lfa.fit_transform(X=activity_fit)
            print(lfa.explained_variance_ratio_)
            activity_neurons_proj_pcs = np.dot(pc_activity, lfa.components_)  # dot prod of PCA activity x loading
            activity_neurons_proj_pcs = activity_neurons_proj_pcs.transpose()
            activity_neurons_proj_pcs_3d = np.stack([activity_neurons_proj_pcs[:, (i_trial * n_timepoints_per_trial):((i_trial + 1) * n_timepoints_per_trial)] for i_trial in range(n_trials)], 
                                                    axis=2)
            # print(f'Subtracted LFA.')
            # selected_ds = selected_ds.assign(activity=selected_ds.activity - activity_neurons_proj_pcs_3d)
            print(f'Showing top 3 PCs')
            selected_ds = selected_ds.assign(activity=(('neuron', 'time', 'trial'), activity_neurons_proj_pcs_3d))
        if aggregation_method == 'average':
            tt_arr = selected_ds.trial_type.data  # extract becauses it's an arr of str, and those cannot be meaned (so will be dropped)
            selected_ds = selected_ds.mean('time')
            selected_ds = selected_ds.assign(trial_type=('trial', tt_arr))  # put back
        elif aggregation_method == 'max':
            tt_arr = selected_ds.trial_type.data  # extract becauses it's an arr of str, and those cannot be meaned (so will be dropped)
            selected_ds = selected_ds.max('time')
            selected_ds = selected_ds.assign(trial_type=('trial', tt_arr))  # put back
        else:
            print(f'WARNING: {aggregation_method} method not implemented!')
        
        if subtract_pop_av:
            selected_ds = selected_ds.assign(activity=selected_ds.activity - selected_ds.activity.mean('neuron'))

        if sort_neurons:
            self.sort_neurons(data=selected_ds.activity, 
                            sorting_method='correlation')
            selected_ds = selected_ds.assign(sorting_neuron_indices=('neuron', self.sorted_inds_neurons_inverse))  # add sorted indices on neuron dim
            selected_ds = selected_ds.sortby(selected_ds.sorting_neuron_indices)
        
        self.time_aggr_ds = selected_ds
        self.time_aggr_ds_pop_av_subtracted = subtract_pop_av
        return selected_ds
        
    def find_discr_index_neurons(self, tt_1='sensory', tt_2='sham'):

        assert self.time_aggr_ds is not None 
        mean_1 = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type==tt_1, drop=True).mean('trial')
        mean_2 = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type==tt_2, drop=True).mean('trial')
        
        var_1 = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type==tt_1, drop=True).var('trial')
        var_2 = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type==tt_2, drop=True).var('trial')

        dprime = np.abs(mean_1 - mean_2) / np.sqrt(var_1 + var_2)
        name = f'dprime_{tt_1}_{tt_2}'

        self.time_aggr_ds = self.time_aggr_ds.assign(**{name: ('neuron', dprime)})
        return dprime

    def find_discr_index_neurons_shuffled(self, tt_1='sensory', tt_2='sham'):

        assert self.time_aggr_ds is not None 
        responses_both = self.time_aggr_ds.activity.where(self.time_aggr_ds.trial_type.isin([tt_1, tt_2]), drop=True)
        responses_both = responses_both.copy(deep=True)
        responses_both = responses_both.data  # go to numpy format to make sure shuffling works 
        responses_both_shuffled = shuffle_along_axis(a=responses_both, axis=1)  # shuffle along trials, per neuron

        # print('normal: ')
        # print(responses_both.mean(1)[:20])
        # print('\n shuffled:')
        # print(responses_both_shuffled.mean(1)[:20])
        assert np.isclose(responses_both_shuffled.mean(1), responses_both.mean(1), atol=1e-5).all()  # ensure shuffling is done along trial dim only 
        assert responses_both_shuffled.shape[1] == 200, 'not 100 trials per tt??'
        responses_1 = responses_both_shuffled[:, :100]
        responses_2 = responses_both_shuffled[:, 100:]

        mean_1 = responses_1.mean(1)
        mean_2 = responses_2.mean(1)

        var_1 = responses_1.var(1)
        var_2 = responses_2.var(1)

        dprime = np.abs(mean_1 - mean_2) / np.sqrt(var_1 + var_2)
        name = f'dprime_{tt_1}_{tt_2}_shuffled'

        self.time_aggr_ds = self.time_aggr_ds.assign(**{name: ('neuron', dprime)})
        return dprime

    def find_all_discr_inds(self, region='s2', shuffled=False,
                            subtract_pop_av=False):
        if self.verbose > 0:
            print('Creating time-aggregate data set')
        ## make time-averaged data
        self.create_time_averaged_response(sort_neurons=False, region=region,
                                           subtracts_pop_av=subtract_pop_av)
        ## get discr arrays (stored in time_aggr_ds)
        if self.verbose > 0:
            print('Calculating d prime values')
        for tt in self.list_tt:  # get all comparisons vs sham
            if tt != 'sham':
                self.find_discr_index_neurons(tt_1=tt, tt_2='sham')
                if shuffled:
                    self.find_discr_index_neurons_shuffled(tt_1=tt, tt_2='sham')

    def population_tt_decoder(self, region='s2', bool_subselect_neurons=False,
                              decoder_type='LDA', tt_list=['whisker', 'sham'],
                              n_cv_splits=5, verbose=1, subtract_pcs=False,
                              t_min=0.4, t_max=2):
        """Decode tt from pop of neurons.
        Use time av response, region specific, neuron subselection.
        Use CV, LDA?, return mean test accuracy"""
        ## make time-averaged data
        self.create_time_averaged_response(sort_neurons=False, region=region,
                                            subtract_pcs=subtract_pcs,
                                           subtract_pop_av=False, trial_type_list=tt_list,
                                           t_min=t_min, t_max=t_max)
        if verbose > 0:
            print('Time-aggregated activity object created')
        ## activity is now in self.time_aggr_ds

        ## neuron subselection
        assert bool_subselect_neurons is False, 'neuron sub selection not yet implemented'

        ## Prepare data
        tt_labels = self.time_aggr_ds.trial_type.data
        neural_data = self.time_aggr_ds.activity.data.transpose()
        assert tt_labels.shape[0] == neural_data.shape[0]  # number of trials 

        ## prepare CV
        cv_obj = sklearn.model_selection.StratifiedKFold(n_splits=n_cv_splits)
        score_arr = np.zeros(n_cv_splits)

        ## run decoder:
        i_cv = 0
        for train_index, test_index in cv_obj.split(X=neural_data, y=tt_labels):
            if verbose > 0:
                print(f'Decoder Cv loop {i_cv + 1}/{n_cv_splits}')
            neural_train, neural_test = neural_data[train_index, :], neural_data[test_index, :]
            tt_train, tt_test = tt_labels[train_index], tt_labels[test_index]

            ##  select decoder 
            if decoder_type == 'LDA':
                decoder_model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
            elif decoder_type == 'QDA':
                decoder_model = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
            elif decoder_type == 'logistic_regression':
                decoder_model = sklearn.linear_model.LogisticRegression()
            else:
                assert False, 'NOT IMPLEMENTED'

            decoder_model.fit(X=neural_train, y=tt_train)
            score_arr[i_cv] = decoder_model.score(X=neural_test, y=tt_test)
            if verbose > 0:
                print(f'Score: {score_arr[i_cv]}')
            i_cv += 1

        if verbose > 0:
            print(score_arr)

        return score_arr

    def assert_normalisation_procedure(self, tt='sensory'):
        '''This function just does some asserts that show how Rob has done the
        baseline normalization of trials in his pre-processing.'''

        assert tt in self.list_tt, f'choose a trial type that is in {self.list_tt}'

        original_obj = getattr(self.SesObj, {v: k for k, v in self._key_dict.items()}[tt])
        trial_start_frames = original_obj.stim_start_frames[0]
        n_trials = 100

        for i_trial in tqdm(range(n_trials)):
            #dfof[0] because of first (and only) imaging plane
            assert len(original_obj.dfof) == 1
            nonnorm_trial = original_obj.dfof[0][:, (trial_start_frames[i_trial] - original_obj.pre_frames):(trial_start_frames[i_trial] + original_obj.post_frames)]
            nonnorm_trial = nonnorm_trial[self._mask_neurons_keep, :]  # filter neurons

            ## perform baselining that is in interareal_analysis.interarealAnalysis._baselineFluTrial()
            baseline_activity = np.mean(nonnorm_trial[:, :original_obj.pre_frames], axis=1)  # mean per neuron across pre-stim time points
            baseline_activity_stack = np.repeat(baseline_activity, nonnorm_trial.shape[1]).reshape(nonnorm_trial.shape)  # Rob stacks instead of broadcasts
            newlynorm_trial = nonnorm_trial - baseline_activity_stack  # subtract baseline
            newlynorm_trial[:, original_obj.pre_frames:(original_obj.pre_frames + original_obj.duration_frames)] = 0  # set artefact frames to zero

            ## two different ways of getting the normalized data
            norm_trial_1 = self.all_trials[tt][:, :, i_trial]  # first, as saved in robs object
            norm_trial_2 = self.full_ds.where(self.full_ds.trial_type == tt, drop=True).isel(trial=i_trial).activity.data  # second, as I extract them
            assert (norm_trial_1 == norm_trial_2).all()  # ensure they are exactly equal
            
            ## Assert that Robs normalized trials are equal to how I normalise them in this function:
            assert norm_trial_1.shape == nonnorm_trial.shape
            assert norm_trial_1.shape == newlynorm_trial.shape
            assert np.isclose(newlynorm_trial - norm_trial_1, 0, atol=1e-6).all(), np.abs((newlynorm_trial - norm_trial_1)).max()  # do it like this in case of float error

        print('All tests passed without issues!')

    def create_nonbaselinsed_alltrials_object(self):
        
        self.all_trials_nonbaselined = {}
        for tt in self.list_tt:
            original_obj = getattr(self.SesObj, {v: k for k, v in self._key_dict.items()}[tt])
            trial_start_frames = original_obj.stim_start_frames[0]
            n_trials = self.all_trials[tt].shape[2]
            self.all_trials_nonbaselined[tt] = np.zeros_like(self.all_trials[tt])

            for i_trial in range(n_trials):
                #dfof[0] because of first (and only) imaging plane
                assert len(original_obj.dfof) == 1
                nonnorm_trial = original_obj.dfof[0][:, (trial_start_frames[i_trial] - original_obj.pre_frames):(trial_start_frames[i_trial] + original_obj.post_frames)]
                nonnorm_trial = nonnorm_trial[self._mask_neurons_keep, :]  # filter neurons
                nonnorm_trial[:, original_obj.pre_frames:(original_obj.pre_frames + original_obj.duration_frames)] = 0  # set artefact to zero
                self.all_trials_nonbaselined[tt][:, :, i_trial] = nonnorm_trial
            # self.all_trials_nonbaselined[tt] = self.all_trials_nonbaselined[tt][self._mask_neurons_keep, :, :]

class AllSessions():
    '''Class that accumulates data from all sessions (of one of the two sess types)'''
    def __init__(self, sess_type='sens', verbose=1,
                 shuffle_trial_labels=False, shuffle_timepoints=False, 
                 shuffle_all_data=False, prestim_baseline=True,
                 bool_filter_neurons=True, 
                 memory_efficient=False):
        self.sess_type = sess_type
        self.n_sessions = 6  ## hard coding this because this is what the data is like
        self.verbose = verbose 
        self.shuffle_timepoints = shuffle_timepoints
        self.shuffle_trial_labels = shuffle_trial_labels
        self.shuffle_all_data = shuffle_all_data
        self.bool_filter_neurons = bool_filter_neurons
        self.prestim_baseline = prestim_baseline
        self.memory_efficent = memory_efficient

        if self.sess_type == 'sens':
            self._key_dict = {'photostim_s': 'sensory', 
                              'photostim_r': 'random',
                              'whisker_stim': 'whisker',
                              'spont': 'sham'}
            self._list_tt_original = ['photostim_s', 'photostim_r', 'spont', 'whisker_stim']
        elif self.sess_type == 'proj':
            self._key_dict = {'photostim_s': 'projecting', 
                              'photostim_r': 'non_projecting',
                              'spont': 'sham'}
            self._list_tt_original = ['photostim_s', 'photostim_r', 'spont']
        else:
            assert False, 'sess_type not implemented'
        self.list_tt = [self._key_dict[tt] for tt in self._list_tt_original]

        self.create_accumulated_data()
        self.dataset_selector = SimpleSession.dataset_selector
        self.create_time_averaged_response = SimpleSession.create_time_averaged_response
        self.squeeze_coords = lambda tmp_dataset: SimpleSession.squeeze_coords(self=self, tmp_dataset=tmp_dataset)

    def create_accumulated_data(self):
        ## Load individual sessions:
        self.sess_dict = {}
        for i_s in range(self.n_sessions):
            self.sess_dict[i_s] = SimpleSession(verbose=self.verbose, session_id=i_s, 
                                                sess_type=self.sess_type,
                                                shuffle_trial_labels=self.shuffle_trial_labels,
                                                shuffle_timepoints=self.shuffle_timepoints,
                                                shuffle_all_data=self.shuffle_all_data,
                                                prestim_baseline=self.prestim_baseline,
                                                bool_filter_neurons=self.bool_filter_neurons)

        if self.verbose > 0:
            print('Individual sessions loaded')

        ## Do some asserts to make sure all is well
        for ii in range(self.n_sessions):
            assert (self.sess_dict[ii].full_ds.activity.ndim == 3)
            if ii > 0:  # compare to first (and hence all are effectively compared against each other)
                assert (self.sess_dict[0].full_ds.activity.shape[1:] == self.sess_dict[ii].full_ds.activity.shape[1:])
                assert (self.sess_dict[0].full_ds.trial_type == self.sess_dict[ii].full_ds.trial_type).all()  # ensure same trial types per trial
                assert np.allclose(self.sess_dict[0].full_ds.time.data, self.sess_dict[ii].full_ds.time.data, atol=1e-6)  # ensure same time axis

        ## Create new xr.Dataset that concatenates all sessions:
        if self.memory_efficent:
            cc_ds = xr.concat(objs=[self.sess_dict[ii].full_ds.copy(deep=True) for ii in range(self.n_sessions)], 
                              dim='neuron', join='override')  # join='override' from https://github.com/pydata/xarray/issues/3681
            for ii in range(self.n_sessions):
                self.sess_dict[ii] = None
        else:
            cc_ds = xr.concat(objs=[self.sess_dict[ii].full_ds for ii in range(self.n_sessions)], 
                              dim='neuron', join='override')  # join='override' from https://github.com/pydata/xarray/issues/3681
        cc_ds['original_neuron_index'] = cc_ds.activity.neuron  # save original neuron index
        cc_ds['neuron'] = np.arange(cc_ds.activity.neuron.shape[0])  # but make main index uniquely accumulating across sessions
        
        if 'neuron' in cc_ds.trial_type.dims:
            for i_trial in range(cc_ds.trial.shape[0]):
                assert len(np.unique(cc_ds.trial_type[:, i_trial].data)) == 1  # ensure all trial types same structure
            cc_ds = cc_ds.assign(trial_type=cc_ds.trial_type.isel(neuron=0).drop('neuron'))  # is this the best way? probably missing some magic here
        if 'neuron' in cc_ds.frame_array.dims:
            for i_time in range(cc_ds.time.shape[0]):
                assert len(np.unique(cc_ds.frame_array[:, i_time].data)) == 1  # ensure all trial types same structure
            cc_ds = cc_ds.assign(frame_array=cc_ds.frame_array.isel(neuron=0).drop('neuron'))  # is this the best way? probably missing some magic here
        if 'neuron' in cc_ds.artefact_bool.dims:
            for i_time in range(cc_ds.time.shape[0]):
                assert len(np.unique(cc_ds.artefact_bool[:, i_time].data)) == 1  # ensure all trial types same structure
            cc_ds = cc_ds.assign(artefact_bool=cc_ds.artefact_bool.isel(neuron=0).drop('neuron'))  # is this the best way? probably missing some magic here
            
        assert np.sum(np.isnan(cc_ds.activity)) == 0      

        self.full_ds = cc_ds        
        all_vars = list(dict(self.full_ds.variables).keys())                    
        self.coord_dict = {var_name: self.full_ds[var_name].dims for var_name in all_vars}  # original (squeezed) coordinates
        self.datatype_dict = {var_name: self.full_ds[var_name].dtype for var_name in all_vars}
        

def shuffle_along_axis(a, axis):
    ## https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)

def opt_leaf(w_mat, dim=0, link_metric='correlation'):
    '''(from popoff)
    create optimal leaf order over dim, of matrix w_mat.
    see also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering'''
    assert w_mat.ndim == 2
    if dim == 1:  # transpose to get right dim in shape
        w_mat = w_mat.T
    dist = scipy.spatial.distance.pdist(w_mat, metric=link_metric)  # distanc ematrix
    link_mat = scipy.cluster.hierarchy.ward(dist)  # linkage matrix
    if link_metric == 'euclidean':
        opt_leaves = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(link_mat, dist))
        # print('OPTIMAL LEAF SOSRTING AND EUCLIDEAN USED')
    elif link_metric == 'correlation':
        opt_leaves = scipy.cluster.hierarchy.leaves_list(link_mat)
    return opt_leaves, (link_mat, dist)

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_time_axis(ax, time_arr, axis='x', label_list=[-2, 0, 2, 4, 6, 8, 10], rotation=0):
    '''Works with heatmap, where len(time_arr) is the number of data points on that axis.
    Not sure how to capture this in an assert.. ?'''
    if axis == 'x':
        ax.set_xticks([np.argmin(np.abs(time_arr - x)) for x in label_list])
        ax.set_xticklabels(label_list, rotation=rotation);
    elif axis == 'y':
        ax.set_yticks([np.argmin(np.abs(time_arr - x)) for x in label_list])
        ax.set_yticklabels(label_list, rotation=rotation);
    else:
        print(f'WARNING: axis {axis} not recognised when creating time axis')


def plot_pop_av(Ses=None, ax_list=None, region_list=['s2'], sort_trials_per_tt=False,
                plot_trial_av=False):
    pop_act_dict = {}
    if ax_list is None:
        if plot_trial_av:
            fig, ax = plt.subplots(2 * len(region_list), len(Ses.list_tt), 
                                    figsize=(len(Ses.list_tt) * 5, 6 * len(region_list)),
                                    gridspec_kw={'wspace': 0.6, 'hspace': 0.5})

        else:   
            fig, ax = plt.subplots(len(region_list), len(Ses.list_tt), 
                                    figsize=(len(Ses.list_tt) * 5, 3 * len(region_list)),
                                    gridspec_kw={'wspace': 0.6, 'hspace': 0.5})
    for i_r, region in enumerate(region_list):
        if len(region_list) > 1:
            ax_row = ax[i_r]
        else:
            ax_row = ax
        for i_tt, tt in enumerate(Ses.list_tt):
            pop_act_dict[tt] = Ses.dataset_selector(region=region, trial_type_list=[tt], 
                                                    sort_neurons=False)
            plot_data = pop_act_dict[tt].activity.mean('neuron').transpose()
            time_ax = plot_data.time.data
            if sort_trials_per_tt:
                sorting, _ = Ses.sort_neurons(data=plot_data.data, save_sorting=False)  # hijack this function, don't save in class
                plot_data = plot_data.data[sorting, :]  # trials sorted 
                ax_row[i_tt].set_ylabel('sorted trials [#]')
            else:
                ax_row[i_tt].set_ylabel('trials [#]')
            
            sns.heatmap(plot_data, ax=ax_row[i_tt], vmin=-0.5, vmax=0.5, cmap='BrBG',
                        cbar_kws={'label': 'activity'})
            ax_row[i_tt].set_xlabel('time [s]')
            xtl = [-2, 0, 2, 4, 6, 8, 10]
            ax_row[i_tt].set_xticks([np.argmin(np.abs(time_ax - x)) for x in xtl])
            ax_row[i_tt].set_xticklabels(xtl, rotation=0)
            ax_row[i_tt].set_title(f'{tt} {region} population average')
            if plot_trial_av:
                if sort_trials_per_tt:
                    pass
                else:
                    plot_data = plot_data.data
                ax_av = ax[i_r + 2, i_tt]
                ax_av.plot(time_ax, plot_data.mean(0), c=colour_tt_dict[tt], linewidth=2)
                ax_av.set_xlabel('Time (s)')
                ax_av.set_ylabel('Trial-av, pop-av DF/F')
                ax_av.set_title(f'{tt} {region} trial-average')
                despine(ax_av)
    plt.suptitle(f'Session {Ses.session_name_readable}')

def plot_hist_discr(Ses=None, ax=None, max_dprime=None, plot_density=True,
                    plot_shuffled=True, yscale_log=False, show_all_shuffled=False,
                    plot_hist=True, plot_kde=False):
    if ax is None:
        ax = plt.subplot(111)

    if max_dprime is None:
        max_dprime = 0
        for tt in Ses.list_tt:
            if tt != 'sham':  # comparison with sham 
                arr_dprime = getattr(Ses.time_aggr_ds, f'dprime_{tt}_sham').data
                max_dprime = np.maximum(max_dprime, arr_dprime.max())
        max_dprime = np.maximum(max_dprime, 2)

    if (Ses.time_aggr_ds.cell_s1 == 1).all():
        region = 'S1'
    elif (Ses.time_aggr_ds.cell_s1 == 0).all():
        region = 'S2'
    else:
        region = 'S1 and S2'

    ## Assume discr has already been computed 
    bins = np.linspace(0, max_dprime * 1.01, 100)

    arr_dprime_dict = {}
    arr_dprime_shuffled_dict = {}
    kde_log_corr = 1e-6 if yscale_log else 0
    plot_tt_list = [tt for tt in Ses.list_tt if tt != 'sham']
    for i_tt, tt in enumerate(plot_tt_list):
        arr_dprime_dict[tt] = getattr(Ses.time_aggr_ds, f'dprime_{tt}_sham').data
        if plot_hist:
            _ = ax.hist(arr_dprime_dict[tt], bins=bins, alpha=0.7,
                    histtype='step', edgecolor=colour_tt_dict[tt],
                    linewidth=3, density=plot_density, label=f'{tt} vs sham')
        if plot_kde:
            kde_f = scipy.stats.gaussian_kde(arr_dprime_dict[tt])
            x_array = np.linspace(0, max_dprime * 1.01, 1000)
            ax.plot(x_array, kde_f(x_array) + kde_log_corr, c=colour_tt_dict[tt], linewidth=3,
                    alpha=0.7, label=f'{tt} vs sham')

        if plot_shuffled:
            if hasattr(Ses.time_aggr_ds, f'dprime_{tt}_sham_shuffled'):
                if show_all_shuffled or i_tt == 0:  # show only first one if not all
                    arr_dprime_shuffled_dict[tt] = getattr(Ses.time_aggr_ds, f'dprime_{tt}_sham_shuffled').data
                    if plot_hist:
                        _ = ax.hist(arr_dprime_shuffled_dict[tt], bins=bins, alpha=1,
                                histtype='step', edgecolor='grey', 
                                linewidth=3, density=plot_density, label=f'shuffled {tt} vs sham')
                    if plot_kde:
                        kde_f = scipy.stats.gaussian_kde(arr_dprime_shuffled_dict[tt])
                        x_array = np.linspace(0, max_dprime * 1.01, 1000)
                        ax.plot(x_array, kde_f(x_array) + kde_log_corr, c='grey', linewidth=3,
                                alpha=0.7, label=f'shuffled {tt} vs sham')
            else:
                print("Shuffled discr not found!")

    ax.legend(loc='best', frameon=False)
    ax.set_title(f'Discriminating trial types in {region} of\n{Ses.session_name_readable}')
    ax.set_xlabel('d prime')
    if plot_density:
        ax.set_ylabel('density')
    else:
        ax.set_ylabel('number of cells')
    if yscale_log:
        ax.set_yscale('log')
    despine(ax)

def plot_raster_sorted_activity(Ses=None, sort_here=False, create_new_time_aggr_data=False,
                                region='s2', ## region only applicable to creating new data!!
                                plot_trial_type_list=['whisker', 'sham'], verbose=1):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    if create_new_time_aggr_data or (Ses.time_aggr_ds is None):
        ## make time-averaged data
        if verbose > 0:
            print('Creating time aggregated data')
        Ses.create_time_averaged_response(sort_neurons=False, region=region,
                                           subtract_pop_av=False)
    plot_data = Ses.time_aggr_ds.activity.where(Ses.time_aggr_ds.trial_type.isin(plot_trial_type_list), drop=True)
    print(plot_data)
    plot_data = plot_data.data

    tt_list = []
    for tt in Ses.time_aggr_ds.trial_type:
        if str(tt.data) not in tt_list:
            tt_list.append(str(tt.data))

    print(tt_list)

    if sort_here:
        sorted_inds, _ = Ses.sort_neurons(data=plot_data, sorting_method='euclidean')
        # print(sorted_inds)
        plot_data = plot_data[sorted_inds, :]
    sns.heatmap(plot_data, ax=ax, vmin=-0.5, vmax=0.5, 
                cmap='BrBG')
    ax.set_xlabel('Trial')
    ax.set_xticks(list(np.arange(2 * len(plot_trial_type_list) + 1) * 50))
    ax.set_xticklabels(sum([[x * 100, tt_list[x]] for x in range(len(plot_trial_type_list))], []) + [100 * (len(plot_trial_type_list) + 1)], rotation=0)
    if sort_here:
        ax.set_ylabel('Neurons (sorted by Eucl. distance')
    else:
        ax.set_ylabel('neuron')
    ax.set_title('Time-averaged data (average of 2 seconds post-stimulus per trial per neuron)')
    return ax

def bar_plot_decoder_accuracy(scores_dict, dict_sess_type_tt=None, 
                    custom_title='Full population LDA-decoding of trial types vs sham across 6 sessions'):

    if dict_sess_type_tt is None:
        dict_sess_type_tt = {'sens': ['sensory', 'random', 'whisker'],
                             'proj': ['projecting', 'non_projecting']}
    n_tt = 5
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'wspace': 0.5})

    for i_r, region in enumerate(['s1', 's2']):
        ax_curr = ax[i_r]
        
        mean_score_arr, err_score_arr = np.zeros(n_tt), np.zeros(n_tt)
        color_list, label_list = [], []
        i_tt = 0
        for sess_type, tt_test_list in dict_sess_type_tt.items():
            for tt in tt_test_list:
                curr_scores = scores_dict[region][sess_type][tt]
                mean_score_arr[i_tt] = np.mean(curr_scores)
                err_score_arr[i_tt] = np.std(curr_scores) / np.sqrt(len(curr_scores)) * 1.96
                color_list.append(colour_tt_dict[tt])
                label_list.append(label_tt_dict[tt])
                i_tt += 1
                
        ax_curr.bar(x=np.arange(n_tt), height=mean_score_arr - 0.5, yerr=err_score_arr,
                    color=color_list, edgecolor='k', linewidth=2, tick_label=label_list,
                    width=0.8, bottom=0.5)
        ax_curr.set_ylim([0, 1])
        ax_curr.set_xlabel('Trial type')
        ax_curr.set_ylabel('Decoding accuracy')
        ax_curr.text(x=-0.5, y=0.05, s=f'{region.upper()}', fontdict={'weight': 'bold'})
        despine(ax_curr)
        ax_curr.set_xticklabels(ax_curr.get_xticklabels(), rotation=45)
    ax[0].text(y=1.15, x=6, fontdict={'weight': 'bold', 'ha': 'center'},
                s=custom_title)
    # return ax

def plot_pca_time_aggr_activity(Ses, trial_type_list=['whisker', 'sensory', 'random', 'sham'],
                                merge_trial_types_during_pca=True, verbose=0,
                                plot_ci=True, plot_indiv_trials=False, plot_loadings=True,
                                ax=None, ax_bottom=None, region='s2', n_pcs=3,
                                t_min=-1, t_max=6, save_fig=False):
    if ax is None:
        if plot_loadings:
            fig = plt.figure(constrained_layout=False, figsize=(8, 9))
            gs_top = fig.add_gridspec(ncols=2, nrows=2, wspace=0.6, hspace=0.4, top=0.9, left=0.05, right=0.95, bottom=0.35)
            gs_bottom = fig.add_gridspec(ncols=3, nrows=1, wspace=0.6, top=0.22, left=0.05, right=0.95, bottom=0.05)
            ax = np.array([[fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[0, 1])], 
                             [fig.add_subplot(gs_top[1, 0]), fig.add_subplot(gs_top[1, 1])]])
            ax_bottom = [fig.add_subplot(gs_bottom[x]) for x in range(3)]
        else:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw={'wspace': 0.6, 'hspace': 0.6})
    pc_activity_dict = {}

    ## PCA calculation:
    if merge_trial_types_during_pca:
        if trial_type_list != sorted(trial_type_list):
            print('WARNING: trial type list should be sorted alphabetically. doing that now for you')
            trial_type_list = sorted(trial_type_list)
        selected_ds = Ses.dataset_selector(region=region, min_t=t_min, max_t=t_max,
                                        sort_neurons=False, trial_type_list=trial_type_list)  # all trial types
        selected_ds_av = selected_ds.groupby('trial_type').mean('trial')  # mean across trials per trial type
        assert selected_ds_av.activity.ndim == 3
        assert list(selected_ds_av.coords.keys()) == ['neuron', 'time', 'trial_type']
        assert (selected_ds_av.trial_type == trial_type_list).all(), f'Order trial types not correct. in Ds: {selected_ds_av.trial_type}, in arg: {trial_type_list}'  ## double check that order of tts is same as input arg, because xarray will sort trial types alphabetically after groupby. (Though technically it doesnt matter because of list concat by trial_types_list order in lines below)
        n_timepoints_per_trial = len(selected_ds_av.time)

        activity_fit = np.concatenate([selected_ds_av.activity.sel(trial_type=tt).data for tt in trial_type_list], axis=1)  # concat average data per trial along time axis
        for i_tt, tt in enumerate(trial_type_list):
            assert (selected_ds_av.activity.sel(trial_type=tt).data == activity_fit[:, (i_tt * n_timepoints_per_trial):((i_tt + 1) * n_timepoints_per_trial)]).all()
        assert activity_fit.ndim == 2
        assert activity_fit.shape[1] == len(trial_type_list) * n_timepoints_per_trial  # neurons x times
        activity_fit = activity_fit.transpose()
        pca = sklearn.decomposition.PCA(n_components=n_pcs)
        pc_activity = pca.fit_transform(X=activity_fit)
        pc_activity = pc_activity.transpose()
        assert pc_activity.shape[1] == len(trial_type_list) * n_timepoints_per_trial
        expl_var = pca.explained_variance_ratio_
        if verbose > 0:
            print(f'Total var expl of all trial types: {expl_var}')

        for i_tt, tt in enumerate(trial_type_list):  ## (we know order is same because of earlier assert)
            pc_activity_dict[tt] = pc_activity[:, (i_tt * n_timepoints_per_trial):((i_tt + 1) * n_timepoints_per_trial)]

        if plot_ci:
            n_trials_per_tt = 100
            assert len(selected_ds.trial) == int(n_trials_per_tt * len(trial_type_list))
            pc_activity_indiv_trials_dict = {tt: np.zeros((n_pcs, n_timepoints_per_trial, n_trials_per_tt)) for tt in trial_type_list}
            assert pca.components_.shape == (n_pcs, len(selected_ds.neuron))
            for i_tt, tt in enumerate(trial_type_list):
                current_tt_ds = selected_ds.where(selected_ds.trial_type == tt, drop=True)
                pca_score_av = pca.score(X=current_tt_ds.mean('trial').activity.data.transpose())
                if verbose > 1:
                    print(f'PCA score of {tt} = {pca_score_av}')
                for i_trial in range(n_trials_per_tt):
                    pca_score_curr_trial = pca.score(current_tt_ds.activity.isel(trial=i_trial).data.transpose())
                    if verbose > 1:
                        print(pca_score_curr_trial)
                    pc_activity_indiv_trials_dict[tt][:, :, i_trial] = np.dot(pca.components_, current_tt_ds.activity.isel(trial=i_trial))


    else:
        assert False, 'This is not the correct of doing this PCA analysis, hence i stopped improving it'
        for i_tt, tt in enumerate(trial_type_list):
            selected_ds = Ses.dataset_selector(region=region, min_t=t_min, max_t=t_max,
                                        sort_neurons=False, 
                                        trial_type_list=[tt])  # select just this trial type (tt)
            selected_ds_av = selected_ds.mean('trial')  # trial average activity

            n_timepoints_per_trial = len(selected_ds_av.time)
            pca = sklearn.decomposition.PCA(n_components=n_pcs)
            activity_fit = selected_ds_av.activity.data  # neurons x times
            assert activity_fit.shape[1] == n_timepoints_per_trial
            activity_fit = activity_fit.transpose()
            pc_activity = pca.fit_transform(X=activity_fit)
            pc_activity_dict[tt] = pc_activity.transpose()
            expl_var = pca.explained_variance_ratio_
            if verbose > 0:
                print(tt, 'explained var: ', expl_var)

    ## Plotting
    for i_tt, tt in enumerate(trial_type_list):
            
        ax[0, 0].plot(pc_activity_dict[tt][0, :], pc_activity_dict[tt][1, :], marker='o',
                        color=colour_tt_dict[tt], linewidth=2, label=tt, linestyle='-')

        if plot_indiv_trials:
            indiv_activity_time_av = pc_activity_indiv_trials_dict[tt][:, 45:75, :].mean(1)  # pcs x time points x trials
            ax[0, 0].scatter(indiv_activity_time_av[0, :], indiv_activity_time_av[1, :],
                             marker='x', c=colour_tt_dict[tt])

        i_row = 0
        i_col = 1
        for i_plot in range(n_pcs):
            pc_num = i_plot + 1
            curr_ax = ax[i_row, i_col]
        
            curr_ax.plot(selected_ds_av.time, pc_activity_dict[tt][i_plot, :],
                            color=colour_tt_dict[tt], linewidth=2, label=f'EV {str(np.round(expl_var[i_plot], 3))}')

            if plot_ci:
                indiv_activity = pc_activity_indiv_trials_dict[tt][i_plot, :, :]
                ci = np.std(indiv_activity, 1) * 1.96 / np.sqrt(indiv_activity.shape[1])
                curr_ax.fill_between(selected_ds_av.time, pc_activity_dict[tt][i_plot, :] - ci, 
                                    pc_activity_dict[tt][i_plot, :] + ci, color=colour_tt_dict[tt], alpha=0.5)

            i_col += 1
            if i_col == 2:
                i_col = 0
                i_row += 1

            if plot_loadings:
                ax_bottom[i_plot].hist(pca.components_[i_plot], bins=30, color='grey', linewidth=1)
                ax_bottom[i_plot].set_xlabel(f'PC {pc_num} loading')
                ax_bottom[i_plot].set_ylabel('Frequency')
                ax_bottom[i_plot].set_title(f'Loadings of PC {pc_num}', fontdict={'weight': 'bold'})

    ## Cosmetics:
    despine(ax[0, 0])
    ax[0, 0].set_xlabel('PC 1')
    ax[0, 0].set_ylabel('PC 2')
    ax[0, 0].legend(loc='best', frameon=False)
    ax[0, 0].set_title('State space PC 1 vs PC 2', fontdict={'weight': 'bold'})

    i_row = 0
    i_col = 1
    for i_plot in range(n_pcs):
        pc_num = i_plot + 1
        curr_ax = ax[i_row, i_col]
        despine(curr_ax)
        curr_ax.set_xlabel('Time (s)')
        curr_ax.set_ylabel(f'PC {pc_num}')
        if merge_trial_types_during_pca:
            curr_ax.set_title(f'PC {pc_num} activity, {int(np.round(expl_var[i_plot] * 100))}% EV of trial-av.', fontdict={'weight': 'bold'})
        else:
            curr_ax.set_title(f'PC {pc_num} activity', fontdict={'weight': 'bold'})
            curr_ax.legend(loc='best')
        i_col += 1
        if i_col == 2:
            i_col = 0
            i_row += 1
        if plot_loadings:
            despine(ax_bottom[i_plot])
    
    plt.suptitle(f'Trial-average PC traces in {region.upper()} region of {Ses.session_name_readable}', fontdict={'weight': 'bold'})

    if save_fig:
        plt.savefig(f'/home/tplas/repos/Vape/jupyter/thijs/figs/pca_activity__{Ses.sess_type}__{Ses.session_name_readable}_{region.upper()}.pdf', bbox_inches='tight')

def manual_poststim_response_classifier(Ses, region='s2', tt_1='sensory', tt_2='sham',
                                        t_min=1, t_max=2, time_aggr_method='average',
                                        n_shuffles=5000, verbose=1, plot_hist=True, ax=None,
                                        neuron_aggr_method='average'):    
    tt_list = [tt_1, tt_2]
    assert len(tt_list) == 2, 'multi classification not implemented'
    assert neuron_aggr_method in ['average', 'variance'], f'{neuron_aggr_method} not recognised'
    assert time_aggr_method == 'average', 'no other aggretation method than average implemented'
    ## Get data
    ds = Ses.dataset_selector(region=region,
                              min_t=t_min, max_t=t_max,
                            sort_neurons=False, trial_type_list=tt_list)
    n_trials_per_tt = 100
    time_av_responses_dict = {}
    for i_tt, tt in enumerate(tt_list):
        if neuron_aggr_method == 'average':
            time_av_responses_dict[tt] = ds.activity.where(ds.trial_type == tt, drop=True).mean(['neuron', 'time'])
        elif neuron_aggr_method == 'variance':
            time_av_responses_dict[tt] = ds.activity.where(ds.trial_type == tt, drop=True).mean('time').var('neuron')    
        assert len(time_av_responses_dict[tt]) == 100

    ## Get real classification performance
    mean_response_per_tt_dict = {tt: time_av_responses_dict[tt].mean() for tt in tt_list}
    threshold = 0.5 * (mean_response_per_tt_dict[tt_list[0]] + mean_response_per_tt_dict[tt_list[1]])
    correct_classification_dict = {}
    for tt in tt_list:
        if mean_response_per_tt_dict[tt] > threshold:
            correct_classification_dict[tt] = time_av_responses_dict[tt] > threshold 
        else:
            correct_classification_dict[tt] = time_av_responses_dict[tt] < threshold 
        if verbose > 0:
            print(f'Number of correctly classified trials of {tt}: {np.sum(correct_classification_dict[tt])}')

    ## Shuffled classification performance:
    n_correct_class_shuffled_dict = {tt: np.zeros(n_shuffles) for tt in tt_list}
    all_responses = np.concatenate([time_av_responses_dict[tt] for tt in tt_list], axis=0)
    assert len(all_responses) == len(tt_list) * n_trials_per_tt
    for i_shuf in range(n_shuffles):
        random_trial_inds = np.random.permutation(len(all_responses))
        shuffled_responses_all = all_responses[random_trial_inds]
        shuffled_responses_1 = shuffled_responses_all[:n_trials_per_tt]
        shuffled_responses_2 = shuffled_responses_all[n_trials_per_tt:]
        mean_1, mean_2 = shuffled_responses_1.mean(), shuffled_responses_2.mean()
        threshold = 0.5 * (mean_1 + mean_2)
        if mean_1 > threshold:
            correct_sh_resp_1 = shuffled_responses_1 > threshold
            correct_sh_resp_2 = shuffled_responses_2 < threshold
        else:
            correct_sh_resp_1 = shuffled_responses_1 < threshold
            correct_sh_resp_2 = shuffled_responses_2 > threshold
        n_correct_class_shuffled_dict[tt_list[0]][i_shuf] = np.sum(correct_sh_resp_1)
        n_correct_class_shuffled_dict[tt_list[1]][i_shuf] = np.sum(correct_sh_resp_2)
            
    ## Compute p value using two sided z test
    n_cor_real_dict, p_val_dict = {}, {}
    for tt in tt_list:
        n_cor_real_dict[tt] = np.sum(correct_classification_dict[tt])
        mean_n_cor_sh = np.mean(n_correct_class_shuffled_dict[tt])
        std_n_cor_sh = np.std(n_correct_class_shuffled_dict[tt])
        zscore_n_cor = (n_cor_real_dict[tt] - mean_n_cor_sh) / std_n_cor_sh
        p_val_dict[tt] = scipy.stats.norm.sf(np.abs(zscore_n_cor)) * 2
        if verbose > 0:
            print(f'Two-sided p value of {tt} = {p_val_dict[tt]}')

    ## Plot
    if plot_hist:
        if ax is None:
            ax = plt.subplot(111)
        hist_n ,_, __ = ax.hist([n_correct_class_shuffled_dict[tt_list[x]] for x in range(2)], 
                    bins=np.arange(40, 70, 1), 
                    density=True, label=tt_list, color=[colour_tt_dict[tt] for tt in tt_list])
        ax.set_xlabel('percentage correctly classified trials')
        ax.set_ylabel('PDF')
        ax.set_title(f'Distr. of correctly classified SHUFFLED trials\nN_bootstrap={n_shuffles}, {region.upper()}, {Ses.session_name_readable}')
        despine(ax)
        max_vals = np.max(np.array([np.max(hist_n[x]) for x in range(2)]))
        for i_tt, tt in enumerate(tt_list): 
            ax.text(s=u"\u2193" + f' (P = {np.round(p_val_dict[tt], 3)})', 
                    x=n_cor_real_dict[tt], y=max_vals * (1 + 0.1 * (i_tt + 1)),
                    fontdict={'ha': 'left', 'color': colour_tt_dict[tt], 'weight': 'bold'})
        ax.set_ylim([0, max_vals * 1.3])

    return p_val_dict
    
def plot_cross_temp_corr(ds, ax=None, name=''):
    n_trials = len(ds.trial)
    tmpcor = np.stack([np.corrcoef(ds.activity.isel(trial=x).data.transpose()) for x in range(n_trials)])
    meancor = tmpcor.mean(0)

    if ax is None:
        ax = plt.subplot(111)
    sns.heatmap(meancor, ax=ax, cmap=sns.color_palette("cubehelix", as_cmap=True), vmax=0.5, vmin=0)
    create_time_axis(ax=ax, time_arr=ds.time.data, axis='x')
    create_time_axis(ax=ax, time_arr=ds.time.data, axis='y')
    ax.invert_yaxis()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Mean cross-temporal correlation across {n_trials} {name} trials')

def plot_hist_p_vals_manual_decoders(p_val_dict, ax=None):
    if ax is None:
        ax = plt.subplot(111)

    tt_list = ['sensory', 'random', 'projecting', 'non_projecting']
    p_val_arr_dict = {}
    n_ses = 6
    p_val_th = 0.05 / (n_ses * len(tt_list))
    plot_logscale = True

    for i_tt, tt in enumerate(tt_list):
        if tt in p_val_dict.keys():
            p_val_arr_dict[tt] = np.array([p_val_dict[tt][ii][tt] for ii in range(n_ses)])

            random_x_coords = i_tt + np.random.rand(n_ses) * 0.1 - 0.05
            ax.plot(random_x_coords, p_val_arr_dict[tt], '.', label=tt, 
                    c=colour_tt_dict[tt], markersize=15, clip_on=False)

    ax.set_xlabel('Trial type')
    ax.set_ylabel('P value per session')
    ax.set_xticks(np.arange(len(tt_list)))
    ax.set_xticklabels([label_tt_dict[tt] for tt in tt_list], rotation=0)
    despine(ax)
    ax.plot([-0.25, 3.25], [p_val_th, p_val_th], c='k', linestyle=':', label='Significance threshold')
    # ax.legend(loc='best')
    if plot_logscale:
        ax.set_yscale('log')
        ax.text(s='Significance threshold', x=-0.25, y=p_val_th * 1.1, fontdict={'va': 'bottom'})
    else:
        ax.set_ylim([0, 1])

def plot_distr_poststim_activity(ses, ax=None, plot_hist=False, tt_list=['sensory', 'sham'], 
                                 plot_logscale=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    tmpr = ses.dataset_selector(region='s2',
                                sort_neurons=False, trial_type_list=tt_list)
    if plot_hist:
        tmphist = ax.hist([tmpr.activity.where(tmpr.trial_type==tt, drop=True).data[:, 45:75, :].ravel() for tt in tt_list],
                            label=tt_list, bins=np.linspace(-5, 5, 101), color=[colour_tt_dict[tt] for tt in tt_list])
        ax.set_ylabel('Frequency ')

        ## plot difference:
        # plt.plot((tmphist[1][1:] + tmphist[1][:-1]) * 0.5, tmphist[0][0, :] - tmphist[0][1, :])
        # plt.xlabel('difference sensory - sham')
        # plt.ylabel('Frequency')
        # print(tmphist[0].sum(1))
    else:
        ax.set_ylabel('PDF (Gaussian fit)')
        for i_tt, tt in enumerate(tt_list):
            plot_data = tmpr.activity.where(tmpr.trial_type==tt, drop=True).data[:, 45:75, :].ravel()
            mean_data = np.mean(plot_data)
            var_data = np.var(plot_data)
            gauss_fit = scipy.stats.norm(loc=mean_data, scale=np.sqrt(var_data))
            x_data = np.linspace(-2.5, 2.5, 1000)
            ax.plot(x_data, gauss_fit.pdf(x_data), linewidth=3, label=tt, c=colour_tt_dict[tt])

    ax.legend(frameon=False)
    ax.set_xlabel('DF/F activity during 1-3 sec post-stim window\n(per neuron per trial per time point)')
    despine(ax)
    ax.set_xlim([-2.5, 2.5])
    if plot_logscale:
        ax.set_yscale('log')

def smooth_trace(trace, one_sided_window_size=2, fix_ends=True):
    trace = copy.deepcopy(trace)
    window_size = int(2 * one_sided_window_size + 1)
    old_trace = copy.deepcopy(trace)
    trace[one_sided_window_size:-one_sided_window_size] = np.convolve(trace, np.ones(window_size), mode='valid') / window_size

    if fix_ends:
        for i_w in range(one_sided_window_size):
            trace[i_w] = np.mean(old_trace[:(i_w + one_sided_window_size + 1)])
            trace[-(i_w + 1)] = np.mean(old_trace[(-1 * (i_w + one_sided_window_size + 1)):])
    return trace

def smooth_trace_with_artefact(trace, one_sided_window_size=2, fix_ends=True):

    if np.sum(np.isnan(trace)) == 0:
        return smooth_trace(trace=trace, one_sided_window_size=one_sided_window_size,
                            fix_ends=fix_ends)

    else:
        inds_change_nan = np.where(np.diff(np.isnan(trace)))[0]
        assert len(inds_change_nan) == 2, 'only made for 1 nan period'
        inds_start_end_nan = inds_change_nan + 1
        ind_start_nan = inds_start_end_nan[0]
        ind_end_nan = inds_start_end_nan[1]

        new_trace = np.zeros_like(trace) + np.nan
        new_trace[:ind_start_nan] = smooth_trace(trace=trace[:ind_start_nan], 
                                                one_sided_window_size=one_sided_window_size,
                                                fix_ends=fix_ends)
        
        new_trace[ind_end_nan:] = smooth_trace(trace=trace[ind_end_nan:], 
                                                one_sided_window_size=one_sided_window_size,
                                                fix_ends=fix_ends)
        return new_trace


def plot_grand_average(ds, ax=None, tt_list=['sham', 'sensory', 'random'],
                       blank_ps=True, smooth_mean=True, plot_legend=False,
                       p_val_dict=None, first_frame_significance=1,
                       plot_significance=True, test_method='cluster',
                       bottom_sign_bar=0.005, addition_sign_bar=0.001,
                       legend_profile=0, plot_zero=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    elif type(ax) == list:
        ## plot tt per ax in list 
        assert len(ax) == len(tt_list)
        bool_tt_split = True 
        assert plot_significance is False, 'fix colours/position of significance array + ax selection + sens/rand array'
        assert plot_legend is False, 'change ax selection'
    else:
        bool_tt_split = False

    for i_tt, tt in enumerate(tt_list):
        time_ax = ds.activity.time 
        grand_av = ds.activity.where(ds.trial_type == tt).mean(['neuron', 'trial'])
        if blank_ps:
            if tt == 'whisker':
                ps_period = np.logical_and(time_ax >= 0, time_ax <= 1.083)
            elif tt == 'sham':
                ps_period = np.zeros_like(time_ax, dtype=bool)
            else:
                ps_period = np.logical_and(time_ax >= 0, time_ax <= 0.3)
        grand_av[ps_period] = np.nan
        if smooth_mean:
            plot_av = smooth_trace_with_artefact(grand_av)
        else:
            plot_av = grand_av
        total_std = ds.activity.where(ds.trial_type == tt).std(['neuron', 'trial'])
        total_ci = total_std * 1.96 / np.sqrt(len(ds.neuron) * len(ds.trial))
        if bool_tt_split:
            curr_ax = ax[i_tt]
        else:
            curr_ax = ax
        if plot_zero:
            curr_ax.plot(time_ax, np.zeros(len(time_ax)), linestyle=':', c='grey')        
        curr_ax.plot(time_ax, plot_av, 
                    label=label_tt_dict[tt],
                    #  label=(label_tt_dict[tt] if tt != 'sham' or legend_profile == 0 else None), 
                linewidth=2, color=colour_tt_dict[tt])
        curr_ax.fill_between(time_ax, plot_av - total_ci, plot_av + total_ci, alpha=0.3, facecolor=colour_tt_dict[tt])

        curr_ax.set_xlabel('Time (s)')
        curr_ax.set_ylabel(r"Grand average $\Delta$F/F")
        # curr_ax.set_ylim([-0.012, 0.008])
        # curr_ax.set_yticks([-0.01, -0.005, 0, 0.005])
        despine(curr_ax)


    if plot_significance:
        assert test_method in ['uncorrected', 'bonferroni', 'holm_bonferroni', 'cluster'], f'test method {test_method} not recognised.!'
        assert p_val_dict is not None, 'please provide p value dictionary'
        if test_method == 'bonferroni':
            sign_dict = bonferroni_correction(p_val_dict=p_val_dict)
        elif test_method == 'holm_bonferroni':
            sign_dict = holm_bonferroni_correction(p_val_dict=p_val_dict)
        elif test_method == 'cluster':
            sign_dict = suprathreshold_cluster_size_test(p_val_dict=p_val_dict)
        elif test_method == 'uncorrected':
            sign_dict = no_correction(p_val_dict=p_val_dict)
        time_array = ds.time.where(ds.frame_array >= first_frame_significance, drop=True).data  

        i_sign_arr = 0
        for key_tts, sign_array in sign_dict.items():
            if key_tts[:5] == 'sham_':
                color_bar = key_tts[5:]
            else:
                color_bar = None
            plot_significance_array(array=sign_array, color_tt=color_bar, ax=ax, 
                                    bottom_sign_bar=bottom_sign_bar + i_sign_arr * addition_sign_bar,
                                    time_ax=time_array)
            i_sign_arr += 1

    if plot_legend:
        if legend_profile == 0:
            ax.legend(frameon=False, loc='lower left')
        elif legend_profile == 1:
            ax.legend(frameon=False, loc='lower left', ncol=2)
        elif legend_profile == 2:
            ax.legend(frameon=False, loc='upper left')
    
def plot_significance_array(array, ax=None, color_tt=None, time_ax=None, bottom_sign_bar=1,
                            text_s=None, text_x=None, text_y=None, text_c=None):

    if ax is None:
        ax = plt.subplot(111)

    if color_tt is None:
        plot_color = 'k'
    else:
        plot_color = colour_tt_dict[color_tt]

    assert type(array) == np.ndarray
    assert array.dtype == 'bool', (array.dtype, array)
    assert array.ndim == 1

    if time_ax is None:
        time_ax = np.arange(len(array))

    assert len(time_ax) == len(array), f'len time_ax is {len(time_ax)}, len array is {len(array)}'

    ax.plot(time_ax, [bottom_sign_bar if x == 1 else np.nan for x in array],
            linewidth=4, c=plot_color, clip_on=True)
    if text_s is not None and text_x is not None and text_y is not None:
        if text_c is None:
            text_c = plot_color
        ax.text(s=text_s, x=text_x, y=text_y, color=text_c)
        

def compute_dynamic_pvals(ds, tt_list=['sensory', 'random', 'sham'],
                          first_frame=37):
    frame_array = ds.frame_array.data
    last_frame = frame_array[-1]
    assert first_frame in list(frame_array)
    n_frames = last_frame - first_frame + 1
    p_val_array = {}  # will contain array of uncorrected p values
    ds_tt_dict = {}

    for tt in tt_list:
        ds_sel = ds.where(ds.trial_type == tt, drop=True)
        if 'trial' in ds_sel.frame_array.dims:
            ds_sel = ds_sel.assign(frame_array=ds_sel.frame_array.isel(trial=0).drop('trial').astype('int')) 
        ds_tt_dict[tt] = ds_sel

    for i_tt1, tt1 in enumerate(tt_list):
        for i_tt2, tt2 in enumerate(tt_list[i_tt1:]):
            if tt1 != tt2:

                key_tts = f'{tt1}_{tt2}'
                print(key_tts)

                p_val_array[key_tts] = np.zeros(n_frames)
                # for i_frame, frame in enumerate(frame_array):
                i_frame = 0
                for frame in range(first_frame, last_frame + 1):
                    if ds.artefact_bool.where(ds.frame_array == frame, drop=True).data[0]:
                        p_val_array[key_tts][i_frame] = np.nan 
                    else:
                        data1 = ds_tt_dict[tt1].activity.where(ds_tt_dict[tt1].frame_array == frame, drop=True).data.ravel()
                        data2 = ds_tt_dict[tt2].activity.where(ds_tt_dict[tt2].frame_array == frame, drop=True).data.ravel()
                        # _, p_val = scipy.stats.wilcoxon(data1, data2)
                        _, p_val = scipy.stats.ttest_ind(data1, data2)

                        p_val_array[key_tts][i_frame] = p_val 
                    i_frame += 1

    return ds_tt_dict, p_val_array

def no_correction(p_val_dict, alpha=0.05):
    '''Test signficance vs alpha with no multipl comparison correction'''
    sign_dict = {}

    for key, p_val_array in p_val_dict.items():
        sign_dict[key] = p_val_array <= alpha

    return sign_dict

def bonferroni_correction(p_val_dict, alpha=0.05):
    '''Test signficance with classic Bonferroni correction'''
    sign_dict = {}

    for key, p_val_array in p_val_dict.items():
        n_curr = len(p_val_array)
        alpha_corr = alpha / n_curr 
        sign_dict[key] = p_val_array <= alpha_corr

    return sign_dict

def holm_bonferroni_correction(p_val_dict, alpha=0.05):
    '''Test signficance with Holm Bonf correction'''
    sign_dict = {}

    for key, p_val_array in p_val_dict.items():
        n_total = len(p_val_array)
        sign_dict[key] = np.zeros(len(p_val_array), dtype=np.bool)
        inds_p_low_to_high = np.argsort(p_val_array)  # nans go to the end, so don't need to change rest of loop
        assert p_val_array[inds_p_low_to_high[0]] < p_val_array[inds_p_low_to_high[1]]

        for i_ind, ind in enumerate(inds_p_low_to_high):
            n_curr = n_total - i_ind
            alpha_corr = alpha / n_curr 
            if p_val_array[ind] <= alpha_corr:
                sign_dict[key][ind] = True
            else:
                break
    return sign_dict

def group_sizes_true(array):
    '''Get sizes of consecutive True values in array'''
    sizes = np.diff(np.where(np.concatenate(([array[0]], array[:-1] != array[1:], [True])))[0])[::2]  #trick from https://stackoverflow.com/questions/24342047/count-consecutive-occurences-of-values-varying-in-length-in-a-numpy-array
    return sizes

def group_size_threshold_test(array, size_threshold=1, verbose=0):
    '''Check if there are any clusters greater than critical cluster size threshold'''
    sign_array = np.zeros(len(array), dtype=np.bool)
    sizes_array = group_sizes_true(array=array)  # get cluster sizes
    sizes_greater_th = sizes_array >= size_threshold  # check if there are any that are greater
    if verbose > 0:
        print('sizes greater than th: ', size_threshold, sizes_array, sizes_greater_th)
    if np.sum(sizes_greater_th) == 0:  # if none greater, return
        return sign_array
    else:  # else, find them and label indiv time points True
        inds_clusters_sign = np.where(sizes_greater_th)[0]  # which clusters are greater
        inds_changes_array = np.where(np.concatenate(([array[0]], array[:-1] != array[1:], [True])))[0]  # which time points indicate start and end of cluster, from group_sizes_true()
        for ind_sign_cluster in inds_clusters_sign:  # for each cluster, get start and end and set all time point in between as True
            start_ind_array = inds_changes_array[int(2 * ind_sign_cluster)]
            end_ind_array = inds_changes_array[int(2 * ind_sign_cluster + 1)]
            sign_array[start_ind_array:end_ind_array] = True 
        return sign_array

def suprathreshold_cluster_size_test(p_val_dict, alpha=0.05, n_perm=5000):
    '''Test whether a cluster of consecutive signficant p values (wrt uncorrected alpha)
    is significant via permutation test
    
    https://stats.stackexchange.com/questions/196769/multiple-comparison-correction-for-temporally-correlated-tests
    https://www.fil.ion.ucl.ac.uk/spm/doc/papers/NicholsHolmes.pdf
    '''
    ## Extract nan indices, and filter: (exclude nans because they would give spurious False p val bools)
    isnotnan_inds_dict = {key: ~np.isnan(val) for key, val in p_val_dict.items()}
    p_val_nonnan_dict = {key: val[isnotnan_inds_dict[key]] for key, val in p_val_dict.items()}
    sign_dict = {key: np.zeros_like(val, dtype='bool') for key, val in p_val_dict.items()}

    p_val_dict_bool = no_correction(p_val_dict=p_val_nonnan_dict)  # get uncorrected signficant values 
    for key, p_val_array in p_val_dict_bool.items():
        shuffled_largest_cluster = np.zeros(n_perm)
        n_vals = len(p_val_array)
        for i_shuffle in range(n_perm):  # perform permutations of array of p values
            shuffled_inds = np.random.choice(a=n_vals, size=n_vals, replace=False)
            shuffled_pvals = p_val_array[shuffled_inds]
            shuffled_largest_cluster[i_shuffle] = np.max(group_sizes_true(array=shuffled_pvals))  # find largest cluster
        ind_threshold = int(np.round(alpha * n_perm)) # see links; it's the a * N + 1 value, so with zero indexing it becomes a * N
        shuffled_largest_cluster = np.sort(shuffled_largest_cluster)[::-1]
        assert shuffled_largest_cluster[0] >= shuffled_largest_cluster[1], print(shuffled_largest_cluster)
        cluster_size_threshold = shuffled_largest_cluster[ind_threshold]  # critical cluster size
        sign_dict[key][isnotnan_inds_dict[key]] = group_size_threshold_test(array=p_val_array, size_threshold=cluster_size_threshold)  # test real data vs permutation-determined threshold

    return sign_dict

def get_percent_cells_responding(session, region='s1', fdr_rate=0.01, 
                                 pre_window=(-1.2, -0.15), post_window=(0.55, 1.65),
                                 post_window_whisker=(1.1, 2.2),  # whisker has a longer period blanked out after stim (up to 1.08 s post stim)
                                 verbose=0, get_responders_targets=False,
                                 stat_test='wilcoxon'):

    assert len(pre_window) == 2 and len(post_window) == 2, 'pre and post window must be a tuple of length 2'
    assert pre_window[0] < pre_window[1] and post_window[0] < post_window[1], 'pre and post window must be in order'
    assert region in ['s1', 's2'], 'region must be s1 or s2'
    assert pre_window[1] < 0 and post_window[0] > 0, 'pre and post window must be before and after stimulus'
    assert post_window_whisker[1] > post_window[1], 'post window whisker must be after post window'
    if get_responders_targets and region == 's2':
        get_responders_targets = False 
        print('No targets in S2, so not finding responders per targets.')
    sel_data = session.dataset_selector(region=region, sort_neurons=False, min_t=pre_window[0], 
                                        max_t=post_window_whisker[1], deepcopy=True)
    meta_data = {'pre_window': pre_window, 'post_window': post_window, 'post_window_whisker': post_window_whisker,
                 'fdr_rate': fdr_rate, 'region': region, 'get_responders_targets': get_responders_targets,
                 'stat_test': stat_test}
    dff = sel_data.activity
    assert dff.dims == ('neuron', 'time', 'trial')
    dff_pre = dff.where(np.logical_and(sel_data.time >= pre_window[0], sel_data.time <= pre_window[1]), drop=True)
    dff_post = dff.where(np.logical_and(sel_data.time >= post_window[0], sel_data.time <= post_window[1]), drop=True)
    dff_post_whisker = dff.where(np.logical_and(sel_data.time >= post_window_whisker[0], sel_data.time <= post_window_whisker[1]), drop=True)
    assert dff_pre.shape == dff_post.shape, f'dff_pre.shape = {dff_pre.shape}, dff_post.shape = {dff_post.shape}'
    assert dff_pre.shape == dff_post_whisker.shape, f'dff_pre.shape = {dff_pre.shape}, dff_post_whisker.shape = {dff_post_whisker.shape}'
    assert len(sel_data.trial) == len(dff_pre.trial)
    n_timepoints = len(dff_pre.time)
    assert len(dff_pre.time) == len(dff_post.time) and len(dff_pre.time) == len(dff_post_whisker.time)
    meta_data['length_windows'] = n_timepoints
    n_trials = len(dff_pre.trial)
    n_positive_responders = np.zeros(n_trials)
    n_negative_responders = np.zeros(n_trials)
    if get_responders_targets:
        n_positive_target_responders = np.zeros(n_trials)
        n_negative_target_responders = np.zeros(n_trials)

    for i_trial in tqdm(dff_pre.trial):
        current_trial_type = sel_data.sel(trial=i_trial).trial_type.data.item()
        assert type(current_trial_type) == str, f'trial type {current_trial_type} of type {type(current_trial_type)} not recognised'
        if current_trial_type == 'whisker':  # whisker has a longer period blanked out after stim (up to 1.08 s post stim), so use different post window
            dff_post_curr = dff_post_whisker.sel(trial=i_trial)
        else:
            dff_post_curr = dff_post.sel(trial=i_trial)
        dff_pre_curr = dff_pre.sel(trial=i_trial)
        assert dff_post_curr.dims == ('neuron', 'time') and dff_pre_curr.dims == ('neuron', 'time')
        p_vals_neurons = np.zeros(len(dff_pre_curr.neuron))
        for i_neuron, neuron in enumerate(dff_pre_curr.neuron):
            if stat_test == 'wilcoxon':
                p_vals_neurons[i_neuron] = scipy.stats.wilcoxon(dff_pre_curr.sel(neuron=neuron).data, 
                                                                dff_post_curr.sel(neuron=neuron).data)[1]
            elif stat_test == 'ttest':
                p_vals_neurons[i_neuron] = scipy.stats.ttest_ind(dff_pre_curr.sel(neuron=neuron).data, 
                                                                 dff_post_curr.sel(neuron=neuron).data)[1]
            else:
                assert False, f'stat test {stat_test} not recognised'
        significance_neurons, p_vals_neurons_corr, _, _ = multitest.multipletests(p_vals_neurons, 
                                                                                  alpha=fdr_rate, method='fdr_bh',
                                                                                  is_sorted=False, returnsorted=False)
        
        positive_cells = dff_post_curr.mean('time') > dff_pre_curr.mean('time')
        negative_cells = np.logical_not(positive_cells)
        significant_positive_cells = np.logical_and(significance_neurons, positive_cells)
        significant_negative_cells = np.logical_and(significance_neurons, negative_cells)

        n_positive_responders[i_trial] = np.sum(significant_positive_cells)
        n_negative_responders[i_trial] = np.sum(significant_negative_cells)

        if get_responders_targets:
            tt = sel_data.sel(trial=i_trial).trial_type.data
            if tt == 'sensory':
                targets_vector = sel_data.sel(trial=i_trial).targets_sensory.data
            elif tt == 'random':
                targets_vector = sel_data.sel(trial=i_trial).targets_random.data
            elif tt == 'projecting':
                targets_vector = sel_data.sel(trial=i_trial).targets_projecting.data
            elif tt == 'non_projecting':
                targets_vector = sel_data.sel(trial=i_trial).targets_non_projecting.data
            elif tt == 'sham' or tt == 'whisker':
                targets_vector = np.zeros(len(significant_positive_cells), dtype=np.bool)
            else:
                assert False, f'trial type {tt} not recognised'

            assert len(targets_vector) == len(significant_positive_cells)
            assert len(targets_vector) == len(significant_negative_cells)
            targets_vector = targets_vector.astype(np.bool)
            positive_responders_targets = np.logical_and(significant_positive_cells, targets_vector)
            negative_responders_targets = np.logical_and(significant_negative_cells, targets_vector)
            n_positive_target_responders[i_trial] = np.sum(positive_responders_targets) / np.sum(targets_vector) * 100  # percentage 
            n_negative_target_responders[i_trial] = np.sum(negative_responders_targets) / np.sum(targets_vector) * 100

    n_cells = len(dff_pre.neuron)
    percent_positive_responders = n_positive_responders / n_cells * 100
    percent_negative_responders = n_negative_responders / n_cells * 100

    df_results = pd.DataFrame({'percent_positive_responders': percent_positive_responders,
                               'percent_negative_responders': percent_negative_responders,
                               'n_positive_responders': n_positive_responders,
                               'n_negative_responders': n_negative_responders,
                               'trial_type': sel_data.trial_type.data,
                               'trial': sel_data.trial.data})
    df_results['session_name_readable'] = session.session_name_readable
    if get_responders_targets:
        df_results['n_positive_responders_targets'] = n_positive_target_responders
        df_results['n_negative_responders_targets'] = n_negative_target_responders

    return percent_positive_responders, percent_negative_responders, sel_data, df_results, meta_data

def overview_plot_metric_vs_responders(sess_dict, sess_type='sens', 
                                       dict_df_responders=None,
                                       metric='pop_var', zscore_metric=False,
                                       response_type='positive',
                                       append_to_title=''):
    n_sess = 6
    assert sess_type in ['sens', 'proj'], f'sess_type must be sens or proj, not {sess_type}'
    if sess_type == 'sens':
        n_tts = 4
    elif sess_type == 'proj':
        n_tts = 3
    assert n_tts in [3, 4]
    assert len(sess_dict) == n_sess
    assert metric in ['pop_var', 'dot_product_spont_stim'], f'metric can not be {metric}'
    assert response_type in ['positive', 'negative', 'total', 'pre_post_corr_s2'], f'response_type must be positive, negative or total, not {response_type}'
   
    fig = plt.figure(figsize=(4 * n_tts, 6))
    gs_tt = {}

    if n_tts == 3:    
        gs_tt[0] = plt.GridSpec(3, 2, left=0.03, right=0.3, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[1] = plt.GridSpec(3, 2, left=0.36, right=0.63, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[2] = plt.GridSpec(3, 2, left=0.7, right=0.97, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    elif n_tts == 4:
        gs_tt[0] = plt.GridSpec(3, 2, left=0.03, right=0.22, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[1] = plt.GridSpec(3, 2, left=0.28, right=0.47, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[2] = plt.GridSpec(3, 2, left=0.53, right=0.72, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[3] = plt.GridSpec(3, 2, left=0.79, right=0.99, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    else:
        assert False, f'n_tts must be 3 or 4, not {n_tts}'

    ax_dict = {i_tt: {} for i_tt in range(n_tts)}
    save_tt_names = np.zeros((n_tts, n_sess), dtype=np.object)
    for sess_id in range(n_sess):
        curr_ds = sess_dict[sess_id].full_ds
        if sess_type == 'sens':
            assert len(curr_ds.trial_type) == 400, f'len trial type is {len(curr_ds.trial_type)}'
        elif sess_type == 'proj':
            assert len(curr_ds.trial_type) == 300, f'len trial type is {len(curr_ds.trial_type)}'
        
        for i_tt in range(n_tts):
            trial_slice = slice(i_tt * 100, (i_tt + 1) * 100) 
            assert len(np.unique(curr_ds.trial_type[trial_slice])) == 1, f'trial slice {trial_slice} contains multiple trial types'
            tt = curr_ds.trial_type[trial_slice].data[0]
            save_tt_names[i_tt, sess_id] = tt

            curr_df = dict_df_responders[sess_type][sess_id][trial_slice]
            if response_type == 'positive':
                responders = curr_df['percent_positive_responders']
            elif response_type == 'negative':
                responders = curr_df['percent_negative_responders']
            elif response_type == 'total':  
                responders = curr_df['percent_positive_responders'] + curr_df['percent_negative_responders']
            elif response_type == 'pre_post_corr_s2':
                ds_pre = curr_ds.sel(time=slice(-0.6, -0.1)).sel(neuron=np.logical_not(curr_ds.cell_s1)).sel(trial=np.arange((i_tt * 100), ((i_tt + 1) * 100)))
                ds_post = curr_ds.sel(time=slice(0.6, 1.1)).sel(neuron=np.logical_not(curr_ds.cell_s1)).sel(trial=np.arange((i_tt * 100), ((i_tt + 1) * 100)))
                assert len(np.unique(ds_pre.trial_type)) == 1, f'trial slice {trial_slice} contains multiple trial types'
                assert np.unique(ds_pre.trial_type)[0] == tt, f'trial slice {trial_slice} contains multiple trial types'
                ds_pre = ds_pre.mean('time')
                ds_post = ds_post.mean('time')
                cc_trials = np.zeros(ds_pre.dims['trial'])
                for i_trial in range(ds_pre.dims['trial']):
                    cc_trials[i_trial] = np.corrcoef(ds_pre.isel(trial=i_trial).activity, ds_post.isel(trial=i_trial).activity)[0, 1]
                responders = cc_trials  

            if metric == 'pop_var':
                metric_plot = curr_ds.sel(time=slice(-0.6, -0.1)).sel(neuron=curr_ds.cell_s1).mean('time').var('neuron').activity[trial_slice]
                metric_plot = np.log(metric_plot)
            elif metric == 'dot_product_spont_stim':
                tmp_ds = curr_ds.sel(time=slice(-0.6, -0.1)).sel(neuron=curr_ds.cell_s1).sel(trial=np.arange((i_tt * 100), ((i_tt + 1) * 100)))
                assert len(np.unique(tmp_ds.trial_type)) == 1, f'trial slice {trial_slice} contains multiple trial types'
                assert np.unique(tmp_ds.trial_type)[0] == tt, f'trial slice {trial_slice} contains multiple trial types'
                if tt == 'sensory':
                    stim_vector = tmp_ds.targets_sensory.values.astype(int)
                elif tt == 'random':
                    stim_vector = tmp_ds.targets_random.values.astype(int)
                elif tt == 'projecting':    
                    stim_vector = tmp_ds.targets_projecting.values.astype(int)
                elif tt == 'non_projecting':
                    stim_vector = tmp_ds.targets_non_projecting.values.astype(int)
                elif tt == 'sham':
                    stim_vector = np.zeros(tmp_ds.dims['neuron'])

                activity_vector_per_trial = tmp_ds.activity.mean('time')
                activity_vector_per_trial = (activity_vector_per_trial - activity_vector_per_trial.mean('neuron')) #/ activity_vector_per_trial.std('neuron')
                assert len(stim_vector) == activity_vector_per_trial.shape[0], f'len stim vector is {len(stim_vector)}, len activity vector is {activity_vector_per_trial.shape[0]}'
                similarity_array = np.dot(stim_vector, activity_vector_per_trial.values)
                metric_plot = similarity_array
        
            if zscore_metric:
                metric_plot = (metric_plot - metric_plot.mean()) / metric_plot.std()

            pearson_r, pearson_p = scipy.stats.pearsonr(metric_plot, responders)

            row_id = sess_id // 2
            col_id = sess_id % 2
            ax_dict[i_tt][sess_id] = fig.add_subplot(gs_tt[i_tt][row_id, col_id])
            curr_ax = ax_dict[i_tt][sess_id]
            curr_ax.plot(metric_plot, responders, '.', markersize=6, color=colour_tt_dict[tt])
            p_val = rfv.readable_p_significance_statement(pearson_p, n_bonf=6)[1]
            curr_ax.annotate(f'r = {pearson_r:.2f}, {p_val}', xy=(0.05, 1.05), xycoords='axes fraction',
                             weight='bold' if p_val != 'n.s.' else 'normal')
            rfv.despine(curr_ax)
            if zscore_metric:
                curr_ax.set_xlim([-3.5, 3.5])
            else:
                if metric == 'pop_var':
                    curr_ax.set_xlim([-4, -1.5])
                elif metric == 'dot_product_spont_stim':
                    curr_ax.set_xlim([-12, 12]) 
            if row_id == 2:
                if metric == 'pop_var':
                    curr_ax.set_xlabel('Population variance\n(log' + (' zscored)' if zscore_metric else ')'))
                elif metric == 'dot_product_spont_stim':
                    curr_ax.set_xlabel('Similarity spont/stim')
            else:
                curr_ax.set_xticklabels([])
            if col_id == 0:
                if response_type == 'pre_post_corr_s2':
                    curr_ax.set_ylabel('Correlation\npre/post stim S1')
                    curr_ax.set_ylim([-0.4, 0.6])
                else:
                    curr_ax.set_ylabel(f'{response_type} responders (%)')
            else:
                pass

    for i_tt in range(n_tts):
        fig.align_ylabels(list(ax_dict[i_tt].values())[::2])
        assert len(np.unique(save_tt_names[i_tt, :])) == 1, f'trial types are not the same across sessions for tt {i_tt}'
        title_use = save_tt_names[i_tt, 0] + append_to_title
        ax_dict[i_tt][0].set_title(title_use, y=1.15, x=1.25,
                                   fontdict={'fontsize': 14, 'fontweight': 'bold', 'color': colour_tt_dict[save_tt_names[i_tt, 0]]})

def plot_sorted_responders_per_trial_type(dict_df_responders):
    df_tmp_result = pd.concat(list(dict_df_responders.values()))

    list_sessions = df_tmp_result['session_name_readable'].unique()
    list_tt = df_tmp_result['trial_type'].unique()
    n_trials = 100
    mat_sorted_responses = {} 
    for ses in list_sessions:
        mat_sorted_responses[ses] = np.zeros((len(list_tt), n_trials))
        for i_tt, tt in enumerate(list_tt):
            curr_df = df_tmp_result[(df_tmp_result['session_name_readable'] == ses) & (df_tmp_result['trial_type'] == tt)]
            total_responders = curr_df['percent_positive_responders'] + curr_df['percent_negative_responders']
            total_responders = np.sort(total_responders)
            mat_sorted_responses[ses][i_tt, :len(total_responders)] = total_responders

    fig, ax = plt.subplots(3, 2, figsize=(12, 10), gridspec_kw={'wspace': 0.3, 'hspace': 0.5})

    for i_sess, sess in enumerate(list_sessions):
        curr_ax = ax.flatten()[i_sess]
        for i_tt, tt in enumerate(list_tt):
            curr_ax.plot(mat_sorted_responses[sess][i_tt, :], '.-', label=tt, color=colour_tt_dict[tt])
        curr_ax.legend()
        curr_ax.set_title(sess)
        curr_ax.set_xlabel('Sorted trial id')
        curr_ax.set_ylabel('Total responders (%)')

def plot_average_responders_per_trial_type(dict_df_responders, sess_type='sens', ax=None,
                                           list_tt_ignore=[], list_trial_numbers_ignore=[],
                                           stat_test_compare='mannwhitneyu', n_bonf=None,
                                           plot_pos_neg_separately=True, plot_stats=True, plot_non_sham_stats=False,
                                           plot_legend=True, add_y=None):
    
    df_responders_all = pd.concat(list(dict_df_responders[sess_type].values()),)
    df_responders_all = df_responders_all.drop(['n_positive_responders', 'n_negative_responders'], axis=1)
    for col_tmp in ['n_positive_responders_targets', 'n_negative_responders_targets']:
        if col_tmp in df_responders_all.columns:
            df_responders_all = df_responders_all.drop(col_tmp, axis=1)
    df_responders_all['percent_total_responders'] = df_responders_all['percent_positive_responders'] + df_responders_all['percent_negative_responders']
    df_responders_av_sess = df_responders_all.groupby(['session_name_readable', 'trial_type',]).mean()
    df_responders_av_sess = df_responders_av_sess.reset_index()
    df_responders_av_sess_sham = df_responders_av_sess[df_responders_av_sess['trial_type'] == 'sham']
    dict_responders_av_sham = {df_responders_av_sess_sham['session_name_readable'].values[i]: df_responders_av_sess_sham['percent_total_responders'].values[i] for i in range(len(df_responders_av_sess_sham))}
    df_responders_all_normalised = df_responders_all.copy()
    columns_normalise = ['percent_positive_responders', 'percent_negative_responders', 'percent_total_responders']
    for col in columns_normalise:
        df_responders_all_normalised[col] = df_responders_all_normalised[col] / df_responders_all_normalised['session_name_readable'].map(dict_responders_av_sham)

    if len(list_tt_ignore) > 0:
        df_responders_all_normalised = df_responders_all_normalised[~df_responders_all_normalised['trial_type'].isin(list_tt_ignore)]
    if len(list_trial_numbers_ignore) > 0:
        df_responders_all_normalised = df_responders_all_normalised[~df_responders_all_normalised['trial'].isin(list_trial_numbers_ignore)]

    ## Extract total, positive and negative repsonders per trial type
    dict_responders_av = {}
    dict_responders_sem = {}  # standard error mean
    for tt in df_responders_all_normalised['trial_type'].unique():
        curr_df = df_responders_all_normalised[df_responders_all_normalised['trial_type'] == tt]
        dict_responders_av[tt] = {}
        dict_responders_sem[tt] = {}
        for col in ['percent_total_responders', 'percent_positive_responders', 'percent_negative_responders']:
            dict_responders_av[tt][col] = curr_df[col].mean()
            dict_responders_sem[tt][col] = 1.96 * curr_df[col].std() / np.sqrt(len(curr_df[col]))
        assert np.isclose(dict_responders_av[tt]['percent_total_responders'], dict_responders_av[tt]['percent_positive_responders'] + dict_responders_av[tt]['percent_negative_responders']), f'total responders : {dict_responders_av[tt]["percent_total_responders"]}, positive responders: {dict_responders_av[tt]["percent_positive_responders"]}, negative responders: {dict_responders_av[tt]["percent_negative_responders"]}'

    if plot_stats:
        ## Use total to compute statistics between trial types
        if stat_test_compare == 'mannwhitneyu':
            stat_test = scipy.stats.mannwhitneyu
        else:
            assert False, 'not implemeneted'
        p_val_dict = {}
        unique_trial_types = df_responders_all_normalised['trial_type'].unique()
        for itt1, tt1 in enumerate(unique_trial_types):
            for itt2, tt2 in enumerate(unique_trial_types):
                if tt1 == tt2:
                    continue
                if (tt2, tt1) in p_val_dict.keys():
                    continue
                p_val_dict[(tt1, tt2)] = stat_test(df_responders_all_normalised[df_responders_all_normalised['trial_type'] == tt1]['percent_total_responders'],
                                                    df_responders_all_normalised[df_responders_all_normalised['trial_type'] == tt2]['percent_total_responders'])[1]
        if n_bonf is None:
            n_bonf = len(p_val_dict.keys())


    if plot_pos_neg_separately:
        if ax is None:
            fig, ax  = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'wspace': 0.4})
        sns.barplot(data=df_responders_all_normalised, x='trial_type', y='percent_positive_responders', 
                    ci=95, ax=ax[0], palette=colour_tt_dict)
        sns.barplot(data=df_responders_all_normalised, x='trial_type', y='percent_negative_responders', 
                    ci=95, ax=ax[1], palette=colour_tt_dict)
        ax[0].set_ylabel('Pos.responders (norm. %)')
        ax[1].set_ylabel('Neg. responders (norm. %)')
        for i_ax in range(2):
            ax[i_ax].set_xlabel('')
            ax[i_ax].set_ylim([0, 2.5])
            if sess_type == 'proj':
                ax[i_ax].set_xticklabels(ax[i_ax].get_xticklabels(), rotation=30)

    else:
        ## Use positive and negative to plot (stacked vertically) using plt.bar
        if ax is None:
            ax = plt.subplot(111)
        
        x_pos = np.arange(len(unique_trial_types))
        width = 0.65
        bottom_pos = np.zeros(len(unique_trial_types))
        for i_col, col in enumerate(['percent_negative_responders', 'percent_positive_responders']):
            y_pos = np.zeros(len(unique_trial_types))
            for itt, tt in enumerate(unique_trial_types):
                y_pos[itt] = dict_responders_av[tt][col]
            ax.bar(x_pos, y_pos, width, bottom=bottom_pos, 
                   color=[colour_tt_dict[tt] for tt in unique_trial_types],
                   hatch='xx' if i_col == 0 else None, linewidth=1,
                   edgecolor='k',
                   yerr=[dict_responders_sem[tt]['percent_total_responders'] for tt in unique_trial_types] if i_col == 1 else None)
            if plot_legend:
                pos_patch = mpatches.Patch(facecolor='white', label='Positive responders', edgecolor='k')
                neg_patch = mpatches.Patch(facecolor='white', hatch='xx', label='Negative responders', edgecolor='k')
                ax.legend(handles=[pos_patch, neg_patch], handlelength=4, handleheight=2, loc='upper left', frameon=False)

            bottom_pos += y_pos

        # height_text_1 = (bottom_pos[-1] - y_pos[-1]) / 2
        # height_text_2 = (bottom_pos[-1] - y_pos[-1]) / 2 + y_pos[-1]
        # ax.annotate('Neg.', xy=(x_pos[-1] + width / 2, height_text_1), clip_on=False,
        #             ha='left', va='center', rotation=90, color='k', fontsize=12)
        # ax.annotate('Pos.', xy=(x_pos[-1] + width / 2, height_text_2), clip_on=False,
        #             ha='left', va='center', rotation=90, color='k', fontsize=12)

        if plot_stats:
            max_y = np.max(bottom_pos)
            # add_y = max_y * 0.2
            if add_y is None:
                add_y = 0.25
            curr_y = max_y 
            ## Plot statistics (a horizontal line between each pair of trial types, with p value)
            for itt1, tt1 in enumerate(unique_trial_types):
                for itt2, tt2 in enumerate(unique_trial_types):
                    if itt1 >= itt2:
                        continue
                    if 'sham' not in [tt1, tt2] and plot_non_sham_stats is False:
                        continue
                    p_val = p_val_dict[(tt1, tt2)]
                    p_val_readable = rfv.readable_p_significance_statement(p_val, n_bonf=n_bonf)[1]
                    curr_y = curr_y + add_y
                    ax.plot([itt1, itt2], [curr_y, curr_y], 'k-', linewidth=1, clip_on=False, zorder=1)
                    ax.annotate(p_val_readable, xy=((itt1 + itt2) / 2, curr_y - 0.1 * add_y), ha='center', va='top', clip_on=False)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([x.replace('non_projecting', 'non\nprojecting') for x in unique_trial_types])
            ax.set_yticks(np.arange(0, max_y + add_y, 0.5))
            # ax.set_ylim([0, max_y + add_y])
            ax.set_ylabel('Responders (normalised %)')
            rfv.despine(ax)

    return df_responders_all_normalised

def overview_plot_correlations(sess_dict, sess_type='sens', 
                               comparison='pre_vs_post',
                               append_to_title=''):
    n_sess = 6
    assert sess_type in ['sens', 'proj'], f'sess_type must be sens or proj, not {sess_type}'
    if sess_type == 'sens':
        n_tts = 4
    elif sess_type == 'proj':
        n_tts = 3
    assert n_tts in [3, 4]
    assert len(sess_dict) == n_sess
    assert comparison in ['pre_vs_post', 's1_vs_s2']

    fig = plt.figure(figsize=(4 * n_tts, 6))
    gs_tt = {}

    if n_tts == 3:    
        gs_tt[0] = plt.GridSpec(3, 2, left=0.03, right=0.28, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[1] = plt.GridSpec(3, 2, left=0.37, right=0.62, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[2] = plt.GridSpec(3, 2, left=0.72, right=0.97, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    elif n_tts == 4:
        gs_tt[0] = plt.GridSpec(3, 2, left=0.03, right=0.22, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[1] = plt.GridSpec(3, 2, left=0.28, right=0.47, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[2] = plt.GridSpec(3, 2, left=0.53, right=0.72, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        gs_tt[3] = plt.GridSpec(3, 2, left=0.79, right=0.99, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    else:
        assert False, f'n_tts must be 3 or 4, not {n_tts}'

    ax_dict = {i_tt: {} for i_tt in range(n_tts)}
    save_tt_names = np.zeros((n_tts, n_sess), dtype=np.object)
    global_min_x, global_max_x = np.inf, -np.inf
    global_min_y, global_max_y = np.inf, -np.inf
    for sess_id in range(n_sess):
        curr_ds = sess_dict[sess_id].full_ds
        if sess_type == 'sens':
            assert len(curr_ds.trial_type) == 400, f'len trial type is {len(curr_ds.trial_type)}'
        elif sess_type == 'proj':
            assert len(curr_ds.trial_type) == 300, f'len trial type is {len(curr_ds.trial_type)}'
        
        for i_tt in range(n_tts):
            trial_slice = slice(i_tt * 100, (i_tt + 1) * 100) 
            assert len(np.unique(curr_ds.trial_type[trial_slice])) == 1, f'trial slice {trial_slice} contains multiple trial types'
            tt = curr_ds.trial_type[trial_slice].data[0]
            save_tt_names[i_tt, sess_id] = tt

            ds_pre = curr_ds.sel(time=slice(-0.6, -0.1)).sel(trial=np.arange((i_tt * 100), ((i_tt + 1) * 100)))
            if tt == 'whisker':
                ds_post = curr_ds.sel(time=slice(1.1, 1.6)).sel(trial=np.arange((i_tt * 100), ((i_tt + 1) * 100)))
            else:
                ds_post = curr_ds.sel(time=slice(0.6, 1.1)).sel(trial=np.arange((i_tt * 100), ((i_tt + 1) * 100)))
            # return ds_pre
            assert len(np.unique(ds_pre.trial_type)) == 1, f'trial slice {trial_slice} contains multiple trial types'
            assert np.unique(ds_pre.trial_type)[0] == tt, f'trial slice {trial_slice} contains multiple trial types'
            # ds_pre = ds_pre.mean('time')
            # ds_post = ds_post.mean('time')
            ds_pre_s1 = ds_pre.sel(neuron=ds_pre.cell_s1).mean('time')
            ds_post_s1 = ds_post.sel(neuron=ds_post.cell_s1).mean('time')
            ds_pre_s2 = ds_pre.sel(neuron=np.logical_not(ds_pre.cell_s1)).mean('time')
            ds_post_s2 = ds_post.sel(neuron=np.logical_not(ds_post.cell_s1)).mean('time')
   
            if comparison == 'pre_vs_post':
                cc_trials = {x: np.zeros(ds_pre.dims['trial']) for x in ['s1', 's2']}
            elif comparison == 's1_vs_s2':
                cc_trials = {x: np.zeros(ds_pre.dims['trial']) for x in ['pre', 'post']}

            for i_trial in range(ds_pre.dims['trial']):
                if comparison == 'pre_vs_post':
                    cc_trials['s1'][i_trial] = np.corrcoef(ds_pre_s1.isel(trial=i_trial).activity, ds_post_s1.isel(trial=i_trial).activity)[0, 1]
                    cc_trials['s2'][i_trial] = np.corrcoef(ds_pre_s2.isel(trial=i_trial).activity, ds_post_s2.isel(trial=i_trial).activity)[0, 1]
                elif comparison == 's1_vs_s2':
                    cc_trials['pre'][i_trial] = np.corrcoef(ds_pre_s1.isel(trial=i_trial).activity, ds_pre_s2.isel(trial=i_trial).activity)[0, 1]
                    cc_trials['post'][i_trial] = np.corrcoef(ds_post_s1.isel(trial=i_trial).activity, ds_post_s2.isel(trial=i_trial).activity)[0, 1]

            if comparison == 'pre_vs_post':
                stat_x = cc_trials['s1']
                stat_y = cc_trials['s2']
                xlabel = 'Corr. pre/\npost stim S1'
                ylabel = 'Corr. pre/\npost stim S2'
            elif comparison == 's1_vs_s2': 
                stat_x = cc_trials['pre']
                stat_y = cc_trials['post']
                xlabel = 'Corr. pre stim \nS1/S2'
                ylabel = 'Corr. post stim \nS1/S2'
                
            pearson_r, pearson_p = scipy.stats.pearsonr(stat_x, stat_y)
            ## linear regression to get slope:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(stat_x, stat_y)
            assert np.isclose(pearson_p, p_value)
            # return stat_x, stat_y, pearson_r, pearson_p, cc_trials, ds_pre_s1, ds_post_s1, ds_pre_s2, ds_post_s2, curr_ds
            row_id = sess_id // 2
            col_id = sess_id % 2
            ax_dict[i_tt][sess_id] = fig.add_subplot(gs_tt[i_tt][row_id, col_id])
            curr_ax = ax_dict[i_tt][sess_id]
            curr_ax.plot(stat_x, stat_y, '.', markersize=6, color=colour_tt_dict[tt])
            ## plot linear regression line:
            x = np.linspace(np.min(stat_x), np.max(stat_x), 100)
            y = slope * x + intercept
            curr_ax.plot(x, y, color='k')
            p_val = rfv.readable_p_significance_statement(pearson_p, n_bonf=6)[1]
            curr_ax.annotate(f'r = {pearson_r:.2f}, {p_val}', xy=(0.05, 1.05), xycoords='axes fraction',
                             weight='bold' if p_val != 'n.s.' else 'normal')
            curr_ax.annotate(f'sl: {slope:.2f}', xy=(1, 0), xycoords='axes fraction',
                             ha='right', va='bottom')
            rfv.despine(curr_ax)
            # curr_ax.set_xlim([-.5, .7])
            # curr_ax.set_ylim([-.5, .7])
            if global_min_x > np.min(stat_x):
                global_min_x = np.min(stat_x)
            if global_max_x < np.max(stat_x):
                global_max_x = np.max(stat_x)
            if global_min_y > np.min(stat_y):
                global_min_y = np.min(stat_y)
            if global_max_y < np.max(stat_y):
                global_max_y = np.max(stat_y)
            if row_id == 2:
                curr_ax.set_xlabel(xlabel)
            else:
                curr_ax.set_xticklabels([])
            if col_id == 0:
                curr_ax.set_ylabel(ylabel)
            else:
                curr_ax.set_yticklabels([])

    global_min_combined = np.min([global_min_x, global_min_y])
    global_max_combined = np.max([global_max_x, global_max_y])
    for i_tt in range(n_tts):
        for sess_id in range(n_sess):
            curr_ax = ax_dict[i_tt][sess_id]
            # curr_ax.plot([global_min_x, global_max_x], [global_min_x, global_max_x],
            #               color='grey', linestyle='--', zorder=-1)
            # curr_ax.set_xlim([global_min_x, global_max_x])
            # curr_ax.set_ylim([global_min_y, global_max_y])
            curr_ax.plot([global_min_combined, global_max_combined], [global_min_combined, global_max_combined],
                          color='grey', linestyle='--', zorder=-1)
            curr_ax.set_xlim([global_min_combined, global_max_combined])
            curr_ax.set_ylim([global_min_combined, global_max_combined])
        fig.align_ylabels(list(ax_dict[i_tt].values())[::2])
        assert len(np.unique(save_tt_names[i_tt, :])) == 1, f'trial types are not the same across sessions for tt {i_tt}'
        title_use = save_tt_names[i_tt, 0] + append_to_title
        ax_dict[i_tt][0].set_title(title_use, y=1.15, x=1.25,
                                   fontdict={'fontsize': 14, 'fontweight': 'bold', 'color': colour_tt_dict[save_tt_names[i_tt, 0]]})

def exponential_decay(x, a, b, c):
                return a * np.exp(-b * x) + c

def plot_change_target_response(dict_df_responders_s1, verbose=1, save_fig=False):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
    n_trials = 100
    n_sess = 6
    tt_dict = {0 : {}, 1: {}}
    for i_st, st in enumerate(['sens', 'proj']):
        dict_mat_responders = {x: np.zeros((n_trials, n_sess)) for x in range(2)}
        for ii in range(n_sess):
            assert 'n_positive_responders_targets' in dict_df_responders_s1[st][ii].columns
            
            total_responders = dict_df_responders_s1[st][ii].n_positive_responders_targets + dict_df_responders_s1[st][ii].n_negative_responders_targets
            ## NB: these are percentages, not number as the column name suggests (above)

            if st == 'sens':
                trials_part1 = np.where(dict_df_responders_s1[st][ii].trial_type == 'sensory')[0]
                trials_part2 = np.where(dict_df_responders_s1[st][ii].trial_type == 'random')[0]
                tt_dict[i_st] = {0: 'sensory', 1: 'random'}
            elif st == 'proj':
                trials_part1 = np.where(dict_df_responders_s1[st][ii].trial_type == 'projecting')[0]
                trials_part2 = np.where(dict_df_responders_s1[st][ii].trial_type == 'non_projecting')[0]
                tt_dict[i_st] = {0: 'projecting', 1: 'non_projecting'}
            part_1 = total_responders[trials_part1].values
            part_2 = total_responders[trials_part2].values
            assert len(part_1) == len(part_2) == n_trials, f'The following dont match: {len(part_1)}, {len(part_2)}, {n_trials}'
            dict_mat_responders[0][:len(part_1), ii] = part_1
            dict_mat_responders[1][:len(part_2), ii] = part_2

            ax[i_st, 0].plot(part_1, label=f'fold {ii}', c='k', alpha=0.3)
            ax[i_st, 1].plot(part_2, label=f'fold {ii}', c='k', alpha=0.3)

        for col in range(2):
            curr_tt = tt_dict[i_st][col]
            ax[i_st, col].plot(dict_mat_responders[col].mean(1), linewidth=3, c=colour_tt_dict[curr_tt], alpha=1)
          
            x = np.arange(n_trials)
            y = dict_mat_responders[col].mean(1)
            
            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
            r_squared_linear = r_value ** 2
            
            # Fit exponential decay
            popt, pcov = scipy.optimize.curve_fit(exponential_decay, x, y, p0=[1, 1e-3, 1], maxfev=10000)
            y_fit = exponential_decay(x, *popt)
            residuals = y - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared_exp = 1 - (ss_res / ss_tot)
            
            ## Calculate Pearson corr:
            pearson_r, pearson_p = scipy.stats.pearsonr(x, y)

            if verbose > 0:
                # Print R-squared values
                print(f'{st} {col}')
                print(f"R-squared (linear): {r_squared_linear:.2f}, R-squared (exponential): {r_squared_exp:.2f}")
            
            if r_squared_exp < r_squared_linear:
                r_sq_use = r_squared_linear
                ax[i_st, col].plot(np.arange(n_trials) * slope + intercept, c='k', alpha=1, linewidth=2)
            else:
                r_sq_use = r_squared_exp
                ax[i_st, col].plot(x, y_fit, c='k', alpha=1, linewidth=2)
            assert slope < 0, f'slope is {slope}'
            assert pearson_r < 0, f'pearson_r is {pearson_r}'
            pearson_p = 0.5 * pearson_p  # Go from two-tailed to one-tailed (assert above ensures that pearson_r is negative)
            ax[i_st, col].annotate("$R^2$" + f' = {r_sq_use:.2f}, r = {np.round(pearson_r, 2)} ({rfv.readable_p_significance_statement(pearson_p, n_bonf=4)[1]})', 
                            (0.98, 0.95), xycoords='axes fraction', ha='right', va='center', color='k')
            # ax[i_st, col].legend()
            ax[i_st, col].set_xlabel('Trials')
            ax[i_st, col].set_ylabel('Total responding\nstim. S1 neurons (%)')
            rfv.despine(ax[i_st, col])
    ax[0, 0].set_title(f'Sensory')
    ax[0, 1].set_title(f'Random')
    ax[1, 0].set_title(f'Projecting')
    ax[1, 1].set_title(f'Non-projecting')
    rfv.equal_lims_n_axs(ax.flatten())
    for i_lab, lab in enumerate('abcd'):
        ax.flatten()[i_lab].annotate(lab, (-0.23, 1.05), xycoords='axes fraction', weight='bold', fontsize=16)

    if save_fig:
        fig.savefig(f'figs/fig_change_target_response.pdf', bbox_inches='tight')

def load_responders(stat_test_use='wilcoxon', fdr_rate_use = '5e-01'):

    ## Load
    with open(f'results_responders/df_responders_s1_{stat_test_use}_window-16-timepoints_fdr-{fdr_rate_use}.pkl', 'rb') as f:
        dict_df_responders_s1 = pickle.load(f)

    with open(f'results_responders/df_responders_s2_{stat_test_use}_window-16-timepoints_fdr-{fdr_rate_use}.pkl', 'rb') as f:
        dict_df_responders_s2 = pickle.load(f)

    return dict_df_responders_s1, dict_df_responders_s2

def load_responders_different(stat_test_use='wilcoxon', fdr_rate_sens='1e-01', fdr_rate_proj='2e-02'):
    df_s1_sens, df_s2_sens = load_responders(stat_test_use=stat_test_use, fdr_rate_use=fdr_rate_sens)
    df_s1_proj, df_s2_proj = load_responders(stat_test_use=stat_test_use, fdr_rate_use=fdr_rate_proj)

    ## combine:
    df_s1_comb = {'sens': df_s1_sens['sens'], 'proj': df_s1_proj['proj']}
    df_s2_comb = {'sens': df_s2_sens['sens'], 'proj': df_s2_proj['proj']}

    return df_s1_comb, df_s2_comb
    

def plot_effect_fdr_responders(ax=None, plot_std=True, save_fig=False):

    stat_test = 'wilcoxon'
    sweep_fdr = ['1e-02', '2e-02', '5e-02', '1e-01', '3e-01', '5e-01']

    for i_fdr, fdr in enumerate(sweep_fdr):
        fdr_float = float(fdr)
        dict_df_responders_s1, dict_df_responders_s2 = load_responders(stat_test_use=stat_test, fdr_rate_use=fdr)
        dict_df_responders_use = dict_df_responders_s2

        for i_st, st in enumerate(['sens', 'proj']):
            df_concat = pd.concat(list(dict_df_responders_use[st].values()))
            df_tmp_result = df_concat.groupby(['session_name_readable', 'trial_type',]).mean()
            for col_tmp in ['n_positive_responders_targets', 'n_negative_responders_targets', 'n_positive_responders', 'n_negative_responders', 'trial', ]:
                if col_tmp in df_tmp_result.columns:
                    df_tmp_result = df_tmp_result.drop(col_tmp, axis=1)
            df_tmp_result = df_tmp_result.reset_index()
            df_tmp_result['trial_type'] = df_tmp_result['trial_type'].replace('sham', f'sham_{st}')
            
            if i_st == 0:
                df_results_all_tt = df_tmp_result.copy()
            else:
                df_results_all_tt = pd.concat([df_results_all_tt, df_tmp_result])

        df_results_all_tt['fdr'] = fdr_float 
        if i_fdr == 0:
            df_results_all_fdr = df_results_all_tt.copy()
        else:
            df_results_all_fdr = pd.concat([df_results_all_fdr, df_results_all_tt])

    df_results_all_fdr['percent_total_responders'] = df_results_all_fdr['percent_positive_responders'] + df_results_all_fdr['percent_negative_responders']
    ## df_results_all_fdr now has columns: session_name_readable, trial_type, percent_positive_responders, percent_negative_responders, fdr, percent_total_responders
    for k, v in label_tt_dict.items():
        df_results_all_fdr['trial_type'] = df_results_all_fdr['trial_type'].replace(k, v)
    if ax is None:
        if plot_std:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'wspace': 0.4})
            ax_mean, ax_std = ax
            ax_list = [ax_mean, ax_std]
            ax_std.set_ylabel('Std of total S2 responders,\n across sessions (%)')
        else:
            ax_mean = plt.subplot(111)
            ax_list = [ax_mean]
    # sns.lineplot(data=df_results_all_fdr, x='fdr', y='percent_total_responders', hue='trial_type', 
    #             marker='o', ax=ax_mean, palette=colour_tt_dict, ci=95)  # with errorbars 
    sns.lineplot(data=df_results_all_fdr.groupby(['trial_type', 'fdr']).mean(), x='fdr', y='percent_total_responders', hue='trial_type', 
                marker='o', ax=ax_mean, palette=colour_tt_dict, legend=True)
    sns.lineplot(data=df_results_all_fdr.groupby(['trial_type', 'fdr']).std(), x='fdr', y='percent_total_responders', hue='trial_type', 
                marker='o', ax=ax_std, palette=colour_tt_dict, legend=False)
    ax_mean.legend(frameon=False)
    ax_mean.set_ylabel('Total S2 responders,\naverage across sessions (%)')
    for i_ax, ax_curr in enumerate(ax_list):
        ax_curr.annotate('a' if i_ax == 0 else 'b', (-0.23, 1.0), va='top', xycoords='axes fraction', weight='bold', fontsize=16)
        ax_curr.set_xscale('log')
        ax_curr.set_xticks(np.array(sweep_fdr, dtype=np.float))
        # ax_curr.set_xticklabels(sweep_fdr)
        ax_curr.set_xticklabels([float(x) for x in sweep_fdr])
        for perc_plot in [1, 2, 5]:
            ax_curr.plot([float(sweep_fdr[0]), float(sweep_fdr[-1])], [perc_plot, perc_plot], c='grey', linestyle='--', zorder=-1)
        rfv.despine(ax_curr)
        ax_curr.set_xlabel('FDR threshold')

    if plot_std:
        rfv.equal_lims_n_axs(ax_list)
    # ## Plot
    # fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'wspace': 0.4})
    # sns.barplot(data=df_results_all_fdr, x='trial_type', y='percent_positive_responders',
    #             hue='fdr', ci=95, ax=ax[0], palette='viridis')
    # sns.barplot(data=df_results_all_fdr, x='trial_type', y='percent_negative_responders',
    #             hue='fdr', ci=95, ax=ax[1], palette='viridis')


    if save_fig:
        fig.savefig(f'figs/fig_fdr_responders_sweep_{stat_test}.pdf', bbox_inches='tight')

    return df_results_all_fdr
