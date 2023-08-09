import copy

from numpy import load

from semi_parametric_estimation.ate import *


def load_truth(replication, knob,fp):
    """
    loading ground truth data
    """
    if fp is None:
        
        file_path = '../result_target/ihdp3/{}/{}/simulation_outputs.npz'.format(knob, replication)
    else:
        file_path = fp+'ihdp3/{}/{}/simulation_outputs.npz'.format(knob, replication)
    data = load(file_path)
    mu_0 = data['mu_0']
    mu_1 = data['mu_1']

    return mu_1, mu_0


def load_data(knob='default', replication=1, model='baseline', train_test='test',fp = None):
    """
    loading train test experiment results
    """
    if fp is None:
        
        file_path = '../result_target/ihdp3/{}/'.format(knob)
    else:
        file_path = fp+'ihdp3/{}/'.format(knob)
    data = load(file_path + '{}/{}/0_replication_{}.npz'.format(replication, model, train_test))

    return data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), data['g'].reshape(-1, 1), \
           data['t'].reshape(-1, 1), data['y'].reshape(-1, 1), data['index'].reshape(-1, 1), data['eps'].reshape(-1, 1)


def get_estimate(q_t0, q_t1, g, t, y_dragon, index, eps, truncate_level=0.01):
    """
    getting the back door adjustment & TMLE estimation
    """

    psi_n = psi_naive(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
    ipw_n, dr_n = psi_weighting(q_t0, q_t1, g, t, y_dragon, truncate_level=truncate_level)
    psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(q_t0, q_t1, g, t,
                                                                                              y_dragon,
                                                                                              truncate_level=truncate_level)
    return psi_n, psi_tmle, initial_loss, final_loss, g_loss,ipw_n, dr_n


def make_table(train_test='train', n_replication=5,fp=None):
    dict = {'tarnet': {'baseline': {'back_door': 0, }},
            'dragonnet': {'baseline': 0}}
    tmle_dict = copy.deepcopy(dict)
    dr_dict = copy.deepcopy(dict)
    ipw_dict = copy.deepcopy(dict)
    for knob in ['dragonnet', 'tarnet']:
        for model in ['baseline']:
            simple_errors, tmle_errors, dr_errors, ipw_errors = [], [],[],[]
            for rep in range(n_replication):
                q_t0, q_t1, g, t, y_dragon, index, eps = load_data(knob, rep, model, train_test,fp)
                a, b = load_truth(rep, knob,fp)
                mu_1, mu_0 = a[index], b[index]

                truth = (mu_1 - mu_0).mean()

                psi_n, psi_tmle, initial_loss, final_loss, g_loss,ipw_n, dr_n = get_estimate(q_t0, q_t1, g, t, y_dragon, index, eps,
                                                                                 truncate_level=0.01)

                err = abs(truth - psi_n).mean()
                tmle_err = abs(truth - psi_tmle).mean()
                dr_error = abs(truth - dr_n).mean()
                ipw_error =  abs(truth - ipw_n).mean()
                simple_errors.append(err)
                tmle_errors.append(tmle_err)
                dr_errors.append(dr_error)
                ipw_errors.append(ipw_error)
            dict[knob][model] = (np.mean(simple_errors),np.std(simple_errors))
            tmle_dict[knob][model] = np.mean(tmle_errors)
            dr_dict[knob][model] = (np.mean(dr_errors),np.std(dr_errors))
            ipw_dict[knob][model] = (np.mean(ipw_errors),np.std(ipw_errors))
    return dict, tmle_dict,dr_dict,ipw_dict


def main():
    print("************ TRAIN Base *********************")
    dict, tmle_dict,dr_dict,ipw_dict = make_table(train_test='train',fp = '../result_base/')
    print("The back door adjustment result is below")
    print(dict)
    print("The DR result is below")
    print(dr_dict)
    print("The IPW result is below")
    print(ipw_dict)
    

    print("************ TRAIN Warmstart*********************")
    dict, tmle_dict,dr_dict,ipw_dict = make_table(train_test='train',fp = '../result_warmstart/')
    print("The back door adjustment result is below")
    print(dict)
    print("The DR result is below")
    print(dr_dict)
    print("The IPW result is below")
    print(ipw_dict)

    print("************ TRAIN l1tl*********************")
    dict, tmle_dict,dr_dict,ipw_dict = make_table(train_test='train',fp = '../result/')
    print("The back door adjustment result is below")
    print(dict)
    print("The DR result is below")
    print(dr_dict)
    print("The IPW result is below")
    print(ipw_dict)
    

    
    print("************ TEST Base*********************")
    dict, tmle_dict,dr_dict,ipw_dict = make_table(train_test='test',fp = '../result_base/')
    print("The back door adjustment result is below")
    print(dict)
    print("The DR result is below")
    print(dr_dict)
    print("The IPW result is below")
    print(ipw_dict)
    

    # print("the tmle estimator result is this ")



    print("************ TEST warmstart*********************")
    dict, tmle_dict,dr_dict,ipw_dict= make_table(train_test='test',fp = '../result_warmstart/')
    print("The back door adjustment result is below")
    print(dict)
    print("The DR result is below")
    print(dr_dict)
    print("The IPW result is below")
    print(ipw_dict)
    



    # print("the tmle estimator result is this ")
    # print(tmle_dict)


    print("************ TEST l1tl*********************")
    dict, tmle_dict,dr_dict,ipw_dict= make_table(train_test='test',fp = '../result/')
    print("The back door adjustment result is below")
    print(dict)
    print("The DR result is below")
    print(dr_dict)
    print("The IPW result is below")
    print(ipw_dict)
    

    # print("the tmle estimator result is this ")
    # print(tmle_dict)
    

if __name__ == "__main__":
    main()
