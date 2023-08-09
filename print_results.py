import numpy as np
import json
import pandas as pd
import glob


target_idx = '4'
pth_to_json = f'./TCL/params_target{target_idx}/'
transfer = pd.DataFrame()
warmstart = pd.DataFrame()
base = pd.DataFrame()

for file in glob.glob(f"{pth_to_json}*transfer*.json"):
    with open(file) as json_data:
        data = json.load(json_data)
        data = pd.DataFrame(data)
        data['batch_size'] = int(file.split("_")[-3])
        data['net'] = file.split("_")[-4]
        data['reg'] = file.split("_")[-2]
        data['lr'] = file.split("_")[-1][:-5]
        transfer = pd.concat([transfer, data])

for file in glob.glob(f"{pth_to_json}*warm*.json"):
    with open(file) as json_data:
        data = json.load(json_data)
        data = pd.DataFrame(data)
        data['batch_size'] = int(file.split("_")[-2])
        data['net'] = file.split("_")[-3]
        data['lr'] = file.split("_")[-1][:-5]
        warmstart = pd.concat([warmstart, data])

for file in glob.glob(f"{pth_to_json}*base*.json"):
    with open(file) as json_data:
        data = json.load(json_data)
        data = pd.DataFrame(data)
        data['batch_size'] = int(file.split("_")[-2])
        data['net'] = file.split("_")[-3]
        data['lr'] = file.split("_")[-1][:-5]
        base = pd.concat([base, data])

transfer['how'] = 'transfer'
warmstart['how'] = 'warmstart'
base['how'] = 'base'

def select_keys(df, net='dragonnet', by='err_train'):
    if 'reg' in df.columns:
        return df.loc[df.net == net].drop(labels=['index_train', 'index_test', 'sim_idx','how'], axis=1).groupby(['net', 'reg', 'lr', 'batch_size']).median().sort_values(by=by).index[0]
    else:
        return df.loc[df.net == net].drop(labels=['index_train', 'index_test', 'sim_idx','how'], axis=1).groupby(['net', 'lr', 'batch_size']).median().sort_values(by=by).index[0]

final = pd.DataFrame()

for by in ['regression_loss_train']:

    best_tar = select_keys(transfer, net='tarnet', by=by)
    best_tar_warm = select_keys(warmstart, net='tarnet', by=by)
    best_tar_base = select_keys(base, net='tarnet', by=by)
    transfer_tar = transfer.loc[(transfer['net'] == 'tarnet') & (transfer['reg'] == best_tar[1]) & (transfer['lr'] == best_tar[2]) & (transfer['batch_size'] == best_tar[3])].copy()
    warm_tar = warmstart.loc[(warmstart['net'] == 'tarnet') & (warmstart['lr'] == best_tar_warm[1]) & (warmstart['batch_size'] == best_tar_warm[2])].copy()
    base_tar = base.loc[(base['net'] == 'tarnet') & (base['lr'] == best_tar_base[1]) & (base['batch_size'] == best_tar_base[2])].copy()
    

    all = pd.concat([transfer_tar, warm_tar, base_tar], axis=0)
    #all = pd.concat([transfer_dragon, transfer_tar, warm_dragon, warm_tar, base_notune], axis=0)
    assert all[['index_train', 'index_test', 'sim_idx']].astype(str).drop_duplicates().shape[0] == all.sim_idx.max() + 1
    

    to_tab = all.drop(labels = ['index_test','index_train','sim_idx','reg', 'lr', 'batch_size'],axis = 1).groupby(['net', 'how']).agg(['mean', 'std'])[['dr_err_train', 'ipw_error_train', 'err_train', 'dr_err_test', 'ipw_error_test', 'err_test']]
    

    to_tab['criteria'] = by
    

    final = pd.concat([final, to_tab], axis=0)

print(final)