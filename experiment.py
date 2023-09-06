import numpy as np
import controller
from tqdm import tqdm
import scipy
import pandas as pd
import os
import shutil



# set experiment params here

num_runs = 3
run_length = 50
exp_name = 'testing'
exp_dir = 'experiments/'
infeed_ratio = 0.25
plots = True
seed = 0


np.random.seed(seed)

# other params that probably shouldnt be changed
num_hits = 10
pdf_std = run_length/num_hits
hit_scale = 10000

 # Set up logging dir
path = exp_dir + exp_name

if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)

param_list = ['num_runs', 'run_length', 'infeed_ratio', 'num_hits', 'pdf_std', 'hit_scale', 'seed']
val_list = [num_runs, run_length, infeed_ratio, num_hits, pdf_std, hit_scale, seed]

lst = list(zip(param_list, val_list))
pd.DataFrame(lst).to_csv(path+'/settings.csv')

# experiment happening here

for i in tqdm(range(num_runs), desc='Experiment', position=0):
   
    # instantiate new controller
    
    C = controller.controller()
    
    # generate new random infeed
    
    infeed = np.ones((run_length, C.num_materials))
    x = np.arange(0, run_length, 1)
    
    hits_1 = np.random.randint(0, run_length, num_hits)
    hits_2 = np.random.randint(0, run_length, num_hits)
    hits_3 = np.random.randint(0, run_length, num_hits)
    
    hit_inds = np.random.randint(0, run_length, (num_hits, C.num_materials))
    for hit in range(num_hits):
        for i in range(C.num_materials):
            infeed[:, i] += scipy.stats.norm.pdf(x, hit_inds[hit, i], pdf_std)*hit_scale

    infeed_norm = np.sum(infeed, axis=1, keepdims=True)
    infeed = infeed/infeed_norm

    # Init system
    x = np.zeros(C.state_dim)
    x[-1] = 1
    opt_traj = np.ones((C.horizon))
    total_score_hist, u_hist, speed_hist = [], [], []
    total_score = 0
    
    # Do optimized run
    for i in tqdm(range(run_length), desc='Run', position=1, leave=False):
        
        # Step sim fwd one time step
        x[:C.num_materials] = infeed[i]* infeed_ratio
        opt_traj = C.solve_opt_traj(x, opt_traj)
        x_new = C.one_step_fwd(x, opt_traj)
        step_score = C.cash_aht(x_new)
        
        # Logging
        total_score += step_score
        total_score_hist.append(total_score)
        u_hist.append(opt_traj[0])
        speed_hist.append(x[-1])
        
        # Set up for next step
        x = x_new
    
    # CSV logging
    lst = list(zip(total_score_hist, u_hist, speed_hist))
    # print(lst)
    df = pd.DataFrame(lst, columns=['total_score_hist', 'u_hist', 'speed_hist'])
    df.to_csv(path+'/opt_run_' + str(i) + '.csv')
    
    mean_speed = np.mean(u_hist)
    
    # Do constant speed run
    
    # Re-init system
    x = np.zeros(C.state_dim)
    x[-1] = 1
    const_traj = np.ones((C.horizon)) * mean_speed
    const_total_score_hist, const_u_hist, const_speed_hist = [], [], []
    total_score = 0
        
     # Do constant speed run
    for i in tqdm(range(run_length)):
        
        # Step sim fwd one time step
        x[:C.num_materials] = infeed[i]* infeed_ratio
        
        x_new = C.one_step_fwd(x, const_traj)
        step_score = C.cash_aht(x_new)
        
        # Logging
        total_score += step_score
        const_total_score_hist.append(total_score)
        const_u_hist.append(const_traj[0])
        const_speed_hist.append(x[-1])
        
        # Set up for next step
        x = x_new
        
    # CSV logging
    lst = list(zip(const_total_score_hist, const_u_hist, const_speed_hist))
    df = pd.DataFrame(lst, columns=['total_score_hist', 'u_hist', 'speed_hist'])
    df.to_csv(path+'/const_run_' + str(i) + '.csv')
    
    