# from experiment.models import *
from experiment.models import *
import os
import glob
import argparse
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from experiment.idhp_data import *
import json
import numpy as np
from process_result.ihdp_ate import get_estimate
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def _split_output(yt_hat, t, y, y_scaler, x, index):
    """
        Split output into dictionary for easier use in estimation
        Args:
            yt_hat: Generated prediction
            t: Binary treatment assignments
            y: Treatment outcomes
            y_scaler: Scaled treatment outcomes
            x: Covariates
            index: Index in data

        Returns:
            Dictionary of all needed data
    """
    yt_hat = yt_hat.detach().cpu().numpy()
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())
    var = "average propensity for treated: {} and untreated: {}".format(g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


def train(train_loader, net, optimizer, criterion,valid_loader= None,l1_reg = None):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if l1_reg is not None:
            l1_penalty = l1_reg * sum([p.abs().sum() for p in net.parameters()])
            loss+= l1_penalty
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss

    valid_loss = None
    if valid_loader is not None:
        valid_loss = 0.0
        net.eval()     # Optional when not using Model Specific layer
        for data, labels in valid_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            target = net(data)
            loss = criterion(target,labels)
            
            if l1_reg is not None:
                loss+= l1_reg * sum([p.abs().sum() for p in net.parameters()])
                
            valid_loss += loss
        valid_loss = valid_loss/len(valid_loader)
    return avg_loss / len(train_loader), valid_loss


def train_and_predict_dragons(t, y_unscaled, x, net,seed = 0, targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross, ratio=1., dragon='', val_split=0.2, batch_size=64,lr =1e-3,l1_reg = None):
    """
    Method for training dragonnet and tarnet and predicting new results
    Returns:
        Outputs on train and test data
    """    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    verbose = 0
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_unscaled)
    train_outputs = []
    test_outputs = []


    # Which loss to use for training the network
    if targeted_regularization:
        loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss

    # loss = knob_loss
    # for reporducing the IHDP experimemt

    i = seed
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)
    

    
    
    # Get the data and optionally divide into train and test set
    if ratio == 0:
        train_index = np.arange(x.shape[0])
        test_index = train_index
    else:
        train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=ratio, random_state=seed)
        print(f'test_index {test_index}')
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    yt_train = np.concatenate([y_train, t_train], 1)
    yt_test = np.concatenate([y_test, t_test], 1)

    # Create data loader to pass onto training method
    tensors_train = torch.from_numpy(x_train).float().to(device), torch.from_numpy(yt_train).float().to(device)
    train_set, valid_set = random_split(TensorDataset(*tensors_train),[1-val_split,val_split])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=500)

    import time;
    start_time = time.time()

    # Configuring optimizers
    # Training the networks first for 100 epochs with the Adam optimizer and
    # then for 300 epochs with the SGD optimizer.
    epochs1 = 100
    epochs2 = 300

    # Add L2 regularization to t0 and t1 heads of the network
    optimizer_Adam = optim.Adam([{'params': net.representation_block.parameters()},
                                 {'params': net.t_predictions.parameters()},
                                 {'params': net.t0_head.parameters(), 'weight_decay': 0.01},
                                 {'params': net.t1_head.parameters(), 'weight_decay': 0.01}], lr=lr)
    optimizer_SGD = optim.SGD([{'params': net.representation_block.parameters()},
                               {'params': net.t_predictions.parameters()},
                               {'params': net.t0_head.parameters(), 'weight_decay': 0.01},
                               {'params': net.t1_head.parameters(), 'weight_decay': 0.01}], lr=lr*0.01, momentum=0.9)
    scheduler_Adam = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_Adam, mode='min', factor=0.5, patience=5,
                                                          threshold=1e-8, cooldown=0, min_lr=0)
    scheduler_SGD = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_SGD, mode='min', factor=0.5, patience=5,
                                                         threshold=0, cooldown=0, min_lr=0)

    train_loss = 0

    early_stopper = EarlyStopper(patience=2, min_delta=0.)

    # Adam training run
    for epoch in range(epochs1):

        # Train on data
        train_loss,val_loss = train(train_loader, net, optimizer_Adam, loss,valid_loader = valid_loader,l1_reg = l1_reg)
        
        if early_stopper.early_stop(val_loss):             
            break

        scheduler_Adam.step(val_loss)

    print(f"Adam loss: train -- {train_loss}, validation -- {val_loss}, epoch {epoch}")

    # SGD training run
    
    early_stopper = EarlyStopper(patience=40, min_delta=0.)

    for epoch in range(epochs2):
        # Train on data
        train_loss,val_loss = train(train_loader, net, optimizer_SGD, loss,valid_loader = valid_loader,l1_reg = l1_reg)

        if early_stopper.early_stop(val_loss):             
            break
        scheduler_SGD.step(val_loss)
        

    print(f"SGD loss: train --  {train_loss}, validation -- {val_loss},  epoch {epoch}")

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    yt_hat_test = net(torch.from_numpy(x_test).float().to(device))
    yt_hat_train = net(torch.from_numpy(x_train).float().to(device))

    test_outputs += [_split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]
    train_outputs += [_split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]
    
    train_all_dicts = _split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)
    test_all_dicts = _split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)
    
    psi_n, psi_tmle, initial_loss, final_loss, g_loss,ipw_n, dr_n = get_estimate(train_all_dicts['q_t0'].reshape(-1, 1), train_all_dicts['q_t1'].reshape(-1, 1), train_all_dicts['g'].reshape(-1, 1), train_all_dicts['t'].reshape(-1, 1), train_all_dicts['y'].reshape(-1, 1), train_all_dicts['index'].reshape(-1, 1), train_all_dicts['eps'].reshape(-1, 1),truncate_level=0.01)

    train_dict = {'psi_n':psi_n, 'classification_mse': g_loss,'ipw_n':ipw_n, 'dr_n':dr_n,'regression_loss':regression_loss(torch.tensor(yt_train).to(device),yt_hat_train).cpu().detach(),'BCE':binary_classification_loss(torch.tensor(yt_train).float().to(device),yt_hat_train).cpu().detach().numpy(),'regression_mse':initial_loss,'index':train_all_dicts['index']}
    
    psi_n, psi_tmle, initial_loss, final_loss, g_loss,ipw_n, dr_n = get_estimate(test_all_dicts['q_t0'].reshape(-1, 1), test_all_dicts['q_t1'].reshape(-1, 1), test_all_dicts['g'].reshape(-1, 1), test_all_dicts['t'].reshape(-1, 1), test_all_dicts['y'].reshape(-1, 1), test_all_dicts['index'].reshape(-1, 1), test_all_dicts['eps'].reshape(-1, 1),truncate_level=0.01)

    
    test_dict = {'psi_n':psi_n, 'classification_mse': g_loss,'ipw_n':ipw_n, 'dr_n':dr_n,'regression_loss':regression_loss(torch.tensor(yt_test).to(device),yt_hat_test).cpu().detach(),'BCE':binary_classification_loss(torch.tensor(yt_test).float().to(device),yt_hat_test).cpu().detach().numpy(),'regression_mses':initial_loss,'index':test_all_dicts['index']}
    
    return test_outputs, train_outputs, net,train_dict,test_dict



def run_ihdp(data_base_dir='/Users/claudiashi/data/ihdp_csv', output_dir='~/result/ihdp/',
             knob_loss=dragonnet_loss_binarycross,
             ratio=1., dragon='',lr2  = 1e-3 ,l1_reg = 1e-3, batchsize2 = 64):

    print("the dragon is {}".format(dragon))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))

    final_output = []
    for idx, simulation_file in enumerate(simulation_files):


        x = load_and_format_covariates_ihdp(simulation_file)
        t, y, y_cf, mu_0, mu_1 = load_all_other_crap(simulation_file)
        
        
        
        ##model hyperparmeters
        batchsize = 64
        lr = 1e-3
        test_ratio = 0.5
        val_split = 0.3
        
        
        batchsize2 = batchsize2
        lr2 = lr2
        l1_reg = l1_reg
        
        ############### select target index################
        ##################################################
        target_col_idx = 4
        target_idx0 = np.where(x[:,target_col_idx]==0)[0]
        target_idx1 = np.where(x[:,target_col_idx]==1)[0]
        x_s = x[target_idx0]
        y_s = y[target_idx0]
        t_s = t[target_idx0]
        y_cf_s = y_cf[target_idx0]
        mu_0_s = mu_0[target_idx0]
        mu_1_s = mu_1[target_idx0]
        
        
        x_t = x[target_idx1]
        y_t = y[target_idx1]
        t_t = t[target_idx1]
        y_cf_t = y_cf[target_idx1]
        mu_0_t = mu_0[target_idx1]
        mu_1_t = mu_1[target_idx1]
        


        for is_targeted_regularization in [False]:
            ## step1 ####
            print("Is targeted regularization: {}".format(is_targeted_regularization))
            
            torch.manual_seed(idx)
            if dragon == 'tarnet':
                print('I am here making tarnet')
                net = TarNet(x.shape[1]).to(device)

            elif dragon == 'dragonnet':
                print("I am here making dragonnet")
                net = DragonNet(x.shape[1]).to(device)
            _, _,net,_,_ = train_and_predict_dragons(t_s, y_s, x_s,net,seed = idx,
                                                                   targeted_regularization=is_targeted_regularization,
                                                                   knob_loss=knob_loss, ratio=0, dragon=dragon,
                                                                 val_split=val_split,batch_size=batchsize,lr = lr)

            parm = {}
            for name, param in net.named_parameters():
                param.grad = None
                parm[name]=param.detach()

            if dragon == 'tarnet':
                print('I am here making tarnet')
                net = TarNet_transfer(x.shape[1],parm).to(device)

            elif dragon == 'dragonnet':
                print("I am here making dragonnet")
                net = DragonNet_transfer(x.shape[1],parm).to(device)
            ### step 2 ###
            test_outputs, train_output,net,train_dict,test_dict= train_and_predict_dragons(t_t, y_t, x_t,net,seed = idx,
                                                                   targeted_regularization=is_targeted_regularization,
                                                                   knob_loss=knob_loss, ratio=test_ratio, dragon=dragon,
                                                                   val_split=val_split, batch_size=batchsize2, lr = lr2,l1_reg =l1_reg)          

            for dict in [train_dict,test_dict]:
                truth = (mu_1_t[dict['index']] - mu_0_t[dict['index']]).mean()
                dict['index'] = dict['index'].tolist()
                dict['err'] = abs(truth - dict['psi_n']).mean()
                dict['dr_err'] = abs(truth - dict['dr_n']).mean()
                dict['ipw_error'] = abs(truth - dict['ipw_n']).mean()
            train_dict = {f'{k}_train': v.item() if 'index' not in k else v for k, v in train_dict.items()}
            test_dict = {f'{k}_test': v.item() if 'index' not in k else v for k, v in test_dict.items()}
            train_dict = {**train_dict,**test_dict}
            train_dict['sim_idx'] = idx
            final_output.append(train_dict)

    
    

    with open(f'./params_target{target_col_idx}/experiments_transfer_{dragon}_{batchsize2}_{l1_reg}_{lr2}.json', 'w') as fp:
        json.dump(final_output,fp,indent=2)
    


def turn_knob(data_base_dir='/Users/claudiashi/data/test/', knob='dragonnet',
              output_base_dir='',lr  = 1e-3, l1reg = 1e-4,batchsize = 16):
    output_dir = os.path.join(output_base_dir, knob)

    if knob == 'dragonnet':
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='dragonnet' ,lr2  = lr ,l1_reg = l1reg, batchsize2 = batchsize)

    if knob == 'tarnet':
        run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='tarnet',lr2  = lr ,l1_reg = l1reg, batchsize2 = batchsize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, help="path to directory LBIDD", default="../dat/ihdp/csv/")
    parser.add_argument('--knob', type=str, default='tarnet',
                        help="dragonnet or tarnet")

    parser.add_argument('--output_base_dir', type=str, help="directory to save the output",default="./")

    parser.add_argument('--transfer_lr',type = float,default=0.001)

    parser.add_argument('--l1reg',type = float,default=0.01)

    parser.add_argument('--batchsize',type = int,default=32)

    args = parser.parse_args()
    turn_knob(args.data_base_dir, args.knob, args.output_base_dir,args.transfer_lr, args.l1reg,args.batchsize)


if __name__ == '__main__':
    main()
