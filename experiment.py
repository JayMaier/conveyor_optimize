import numpy as np
import controller
from tqdm import tqdm
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import scipy
import pandas as pd
import os
import shutil

class Experiment():
    def __init__ (self):
        
        # set experiment params here

        self.num_runs = 25
        self.run_length = 46000
        self.exp_name = '25x_hour_new_infeed'
        self.exp_dir = 'experiments/'
        self.infeed_ratio = 0.27
        # self.seed = 0
        C = controller.controller()
        self.tries = 100000

        # np.random.seed(self.seed)

        # other params that probably shouldnt be changed
        self.num_hits = 10
        self.pdf_std = 0.5*self.run_length/self.num_hits
        self.hit_scale = 10000

        # Set up logging dir
        self.path = self.exp_dir + self.exp_name

        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.mkdir(self.path)

        param_list = ['num_runs', 'run_length', 'infeed_ratio', 'num_hits', 'pdf_std', 'hit_scale']
        val_list = [self.num_runs, self.run_length, self.infeed_ratio, self.num_hits, self.pdf_std, self.hit_scale]

        lst = list(zip(param_list, val_list))
        pd.DataFrame(lst).to_csv(self.path+'/settings.csv')
        
        # generate new random infeed
        
        self.infeed = np.ones((self.num_runs, self.run_length, C.num_materials))
        x = np.arange(0, self.run_length, 1)
            
        for run_num in range(self.num_runs):
            for trie in range(self.tries):
                infeed_run = np.ones((self.run_length, C.num_materials))
                hit_inds = np.random.randint(0, self.run_length, (self.num_hits, C.num_materials))
                for hit in range(self.num_hits):
                    for i in range(C.num_materials):
                        infeed_run[:, i] += scipy.stats.norm.pdf(x, hit_inds[hit, i], self.pdf_std)*self.hit_scale
                
                infeed_norm = np.sum(infeed_run, axis=1, keepdims=True)
                infeed_run = infeed_run/infeed_norm
                self.infeed[run_num] = infeed_run
                q90_10_0 = np.percentile(self.infeed[run_num, :, 0], 90)/np.percentile(self.infeed[run_num, :, 0], 10)
                q90_10_1 = np.percentile(self.infeed[run_num, :, 1], 90)/np.percentile(self.infeed[run_num, :, 1], 10)
                q90_10_2 = np.percentile(self.infeed[run_num, :, 2], 90)/np.percentile(self.infeed[run_num, :, 2], 10)
                q_s = np.array([q90_10_0, q90_10_1, q90_10_2])
                if np.all(q_s <= 4.58) and np.all(q_s >= 3.39):
                    print('got it! ', run_num)
                    break
            np.savetxt(self.path + '/infeed' + str(run_num) + '.csv', self.infeed[run_num], delimiter = ', ')         
                    
        
    def __call__(self, idx):
        
        # experiment happening here

        # instantiate new controller
        
        C = controller.controller()
        
        infeed = self.infeed[idx]
        # Init system
        x = np.zeros(C.state_dim)
        x[-1] = 1
        opt_traj = np.ones((C.horizon))
        total_score_hist, u_hist, speed_hist = [], [], []
        total_score = 0
        
        # Do optimized run
        for i in tqdm(range(self.run_length), position = 0):
            
            # Step sim fwd one time step
            x[:C.num_materials] = infeed[i]* self.infeed_ratio
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
        df.to_csv(self.path+'/opt_run_' + str(idx) + '.csv')
        
        mean_speed = np.mean(u_hist)
        
        # Do constant speed run
        
        # Re-init system
        x = np.zeros(C.state_dim)
        x[-1] = 1
        const_traj = np.ones((C.horizon)) * mean_speed
        const_total_score_hist, const_u_hist, const_speed_hist = [], [], []
        total_score = 0
            
        # Do constant speed run
        for i in range(self.run_length):
            
            # Step sim fwd one time step
            x[:C.num_materials] = infeed[i]* self.infeed_ratio
            
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
        df.to_csv(self.path+'/const_run_' + str(idx) + '.csv')
        

experiment = Experiment()
# with Pool() as pool:
#     pool.map(experiment, range(0, experiment.num_runs))
process_map(experiment, range(0, experiment.num_runs), position=1)