## Script to compute responders

import decoding_analysis_vis as dav
import pickle
import os 

def compute_responders(save_folder='/home/tplas/repos/S1S2_mechanisms/jupyter/thijs/results_responders', 
                       fdr_rate=0.01, pre_window=(-1.2, -0.15), post_window=(0.55, 1.65),
                     post_window_whisker=(1.1, 2.2), stat_test='wilcoxon'):
    ## Load all data 
    all_sess = {}
    for st in ['sens', 'proj']:
        all_sess[st] = dav.AllSessions(sess_type=st)
     
    ## Compute responders, per region 
    n_sessions = 6
    for region_use in ['s1', 's2']:
        dict_df_responders = {}
        meta_data_prev = None 
        for st in ['sens', 'proj']:
            dict_df_responders[st] = {}
            for ii in range(n_sessions):
                tmp_sess = all_sess[st].sess_dict[ii]
                n_pos_resp, n_neg_resp, tmp_ds, df_results, meta_data_new = dav.get_percent_cells_responding(session=tmp_sess, 
                                                                                get_responders_targets=True if region_use=='s1' else False,
                                                                                region=region_use, fdr_rate=fdr_rate, pre_window=pre_window,
                                                                                post_window=post_window, post_window_whisker=post_window_whisker,
                                                                                stat_test=stat_test)
                dict_df_responders[st][ii] = df_results
                if meta_data_prev is None:
                    meta_data_prev = meta_data_new
                else:
                    assert meta_data_prev == meta_data_new 
                    meta_data_prev = meta_data_new
        
        dict_df_responders['meta_data'] = meta_data_new
        length_windows = meta_data_new['length_windows']
        stat_test = meta_data_new['stat_test']
        filename = f'df_responders_{region_use}_{stat_test}_window-{length_windows}-timepoints.pkl'
        filepath = os.path.join(save_folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(dict_df_responders, f)
    

    all_sess = None 

if __name__ == '__main__':
    compute_responders()