import numpy as np


class controller():
    def __init__(self, mats= 4):
        # init things here
        self.horizon = 20
        
        self.num_materials = 3
        self.belt_length = 62
        self.volume_length = 2
        self.speed_control_ratio = 0.2
        self.num_speeds = 4
        self.prices = np.array([0.306, 0.124, 0.01875]) #units here are cents/unit.  PET: 12.25 c/lb, PP: 5.5 c/lb, 3-7 bale 0.75 c/lb 40 pc/lb
        # self.prices = np.array([1, 0.001, 0.00001875]) #deweight things to make it more clear what is happening
        
        self.speed_min = 0.7
        self.speed_max = 1.33
        self.item_0_start_id = 10
        self.item_0_end_id = 15
        self.item_1_start_id = 25
        self.item_1_end_id = 30
        self.picks_per_sec = 1.5
        self.update_rate = 12
        
        if mats == 2:
            self.horizon = 20
        
            self.num_materials = 2
            self.belt_length = 32
            self.volume_length = 2
            self.speed_control_ratio = 0.2
            self.num_speeds = 4
            
            self.speed_min = 0.7
            self.speed_max = 1.33
            self.item_0_start_id = 10
            self.item_0_end_id = 15
            self.picks_per_sec = 1.5
            self.update_rate = 12
            self.prices = np.array([0.306, 0.01875])
            # self.prices -= 0.1
            
        if mats == 4:
            self.horizon = 20
        
            self.num_materials = 4
            self.belt_length = 42
            self.volume_length = 2
            self.speed_control_ratio = 0.2
            self.num_speeds = 4
            
            self.speed_min = 0.7
            self.speed_max = 1.33
            self.item_0_start_id = 5
            self.item_0_end_id = 10
            self.picks_per_sec = 1.5
            self.update_rate = 12
            self.prices = np.array([0.306, 0.124, 0.062, 0.01875])
            # self.prices -= 0.1
            self.item_1_start_id = 10
            self.item_1_end_id = 15
            
            self.item_2_start_id = 15
            self.item_2_end_id = 20
        
        self.num_volumes = int(self.belt_length/self.volume_length)
        self.state_dim = self.num_materials * self.num_volumes + 1
        
        self.iters = 10
        self.armijo_iters = 10
        self.armijo_backtrack = 0.5
        self.delta = 0.05
        self.learning_rate = 0.05
        
        
        self.A = self.generate_A_matrix()
        
        self.B = np.zeros((self.state_dim))
        self.B[-1] = self.speed_control_ratio
        
        self.picks = []
        self.cum_sorts = 0
        self.U_traj_last = np.ones((self.horizon))

        
        
        
        pick_width_0 = self.item_0_end_id - self.item_0_start_id
        picks_0 = []
        for i in range(pick_width_0):
            picks_0.append(self.item_0_start_id + i)

        self.picks.append(picks_0)
        
        if mats >= 3:
            pick_width_1 = self.item_1_end_id - self.item_1_start_id
            picks_1 = []
            for i in range(pick_width_1):
                picks_1.append(self.item_1_start_id + i)
                
            self.picks.append(picks_1)
            
        if mats >= 4:
            pick_width_2 = self.item_2_end_id - self.item_2_start_id
            picks_2 = []
            for i in range(pick_width_2):
                picks_2.append(self.item_2_start_id + i)
                
            self.picks.append(picks_2)
        
        x_0 = np.zeros((self.state_dim))
        x_0[-1] = 1
        _, self.master_sort_mat = self.rollout_linearize_A(x_0)
        
    def generate_A_matrix(self):
        A = np.zeros((self.state_dim, self.state_dim, self.num_speeds))
        
        for speed in range(self.num_speeds):
            for i in range((self.num_volumes-speed)*self.num_materials):
                A[i+(speed * self.num_materials), i,  speed] = 1
            for j in range(speed):
                for i in range(self.num_materials):
                    A[j*self.num_materials + i, i, speed] = 1
            A[-1, -1, speed] = 1-self.speed_control_ratio
            
        return A
  
    def solve_opt_traj(self, x_init, U_traj_init):
        debug = 0
        
        U_traj_nom = np.ones(self.horizon) * x_init[-1]
        U_traj_nom[:-1] = np.copy(U_traj_init[1:])
        U_traj_nom[-1] = np.copy(U_traj_init[-1])
        grad = np.zeros_like(U_traj_nom)
        deltas = self.delta * np.diag(np.ones_like(U_traj_nom))
        lr = self.learning_rate
        
        base_score = self.rollout(x_init, U_traj_nom)

        eye_traj = np.eye(self.horizon)
        converged = False
        B_new = np.eye(self.horizon)
        # calcualte gradient:
        for i in range(self.iters):
            for dir in range(self.horizon):
                s = self.rollout(x_init, (U_traj_nom + deltas[dir]))
                grad[dir] = (s-base_score)/self.delta
        
        # if first step, do gradient descent:
            
            if i == 0:
                
                update = lr * grad
               
            # otherwise do BFGS step
            else:
                
                y = (grad - last_grad)
                del_x = np.copy(last_step)
                y = y[:, np.newaxis]
                del_x = del_x[:, np.newaxis]

                if np.all(y !=0 ) and np.any(del_x != 0):
                    term1 = np.eye(self.horizon) - ((del_x @ y.T)/(y.T@del_x))
                    term2 = np.copy(last_B)
                    term3 = np.eye(self.horizon) - ((y@del_x.T)/(y.T@del_x))
                    term4 = ((del_x @ del_x.T)/(y.T @ del_x))
                  
                    B_new = np.abs(term1 @ term2 @ term3 + term4)
                    update = B_new @ grad
                    
                      
               
                else:
                    
                    update = lr * grad
                    B_new = np.eye(self.horizon)
            
            
            #armijo rule backtracking line search
            
            for attempt in range(self.armijo_iters):
                
                
                U_traj_hyp = U_traj_nom + update
                
                U_traj_hyp_clipped = U_traj_hyp.clip(self.speed_min, self.speed_max)
                update = U_traj_hyp_clipped - U_traj_nom 
               
                U_traj_hyp = np.copy(U_traj_hyp_clipped)
                hyp_score = self.rollout(x_init, U_traj_hyp)
               
                
                if hyp_score > base_score:
                    
                    break
                elif hyp_score == base_score:   
                    converged = True
                    
                    break
                elif attempt + 1 < self.armijo_iters:
                    update *= self.armijo_backtrack
                    
                else: 
                    
                    update = lr * grad
                    
                    break          
                 
            last_B = np.copy(B_new)
            last_grad = np.copy(grad)
            last_step = U_traj_hyp - U_traj_nom
            
            if np.all(np.abs(last_step) <= 0.000000001):
                
                converged = True
            base_score = hyp_score
           
            U_traj_nom = np.copy(U_traj_hyp)
           
            if converged == True:
                break            
        
        return U_traj_nom
    
    def rollout(self, x, U_traj):
        
        X_traj = np.zeros((self.horizon + 1, self.state_dim))
        score = 0
        
        
        
        for step in range(self.horizon):
            if step == 0:
                
                A_rollout, sort_mat = self.rollout_linearize_A(x)
                X_traj[step] = A_rollout @ x + self.B * U_traj[step]
            else:
                A_rollout, sort_mat = self.rollout_linearize_A(X_traj[step-1])
                X_traj[step] = A_rollout @ X_traj[step - 1] + self.B * U_traj[step]

            score += self.get_state_cost(X_traj[step], sort_mat, step+1)
            
        
        A_rollout, sort_mat = self.rollout_linearize_A(X_traj[self.horizon-1])
        X_traj[self.horizon] = A_rollout @ X_traj[self.horizon-1] + self.B * U_traj[self.horizon-1]
        
        score += self.get_state_cost(X_traj[self.horizon], sort_mat, self.horizon)
            
        
        return score
    
    def rollout_linearize_A(self, x):
        
        
        floor = np.floor(x[-1])
        ceil = floor + 1
        ratio = x[-1] - floor
        A = self.A[:, :, int(floor)] * (1-ratio) + self.A[:, :, int(ceil)] * ratio
        sort_mat = np.zeros((self.state_dim, len(self.picks)))

        
        for mat in range(len(self.picks)):
            sort_vec = np.zeros((self.state_dim))
            
            for idx, i in enumerate(self.picks[mat]):
                
                sort_vec[i*self.num_materials + mat] = 1
            if sort_vec.T @ x > self.picks_per_sec / self.update_rate:
                
                sort_vec *= (self.picks_per_sec/self.update_rate)/(sort_vec.T @ x)
                
            sort_mat[:, mat] = sort_vec
            A = A * (np.ones((self.state_dim)) - sort_vec)
        
        
        return A, sort_mat

    def get_state_cost(self, x, sort_mat, discount, verbose=False, eval = False):

        positive_sort_price_vec = sort_mat @ self.prices[:-1]
        positive_sort_val = positive_sort_price_vec @ x
        
       
        
        negative_sort_price_vec = np.zeros((self.state_dim))
        negative_sort_price_vec[(self.num_volumes-1)*self.num_materials : self.num_volumes*self.num_materials] = self.prices[-1]
        negative_sort_val = negative_sort_price_vec @ x

       
        
        missed_opportunity_vec = (self.master_sort_mat-sort_mat) @ (self.prices[-1] - self.prices[:-1])
        missed_opportunity_val = missed_opportunity_vec @ x
        
        if eval:
            missed_opportunity_val = 0

        
        if verbose:
            print('\n---------------------------------------\npositive sort val: ', positive_sort_val)
            # print('potisive sort price vec: ', positive_sort_price_vec)
            print('x, ', x[-1])
            print('negative sort val: ', negative_sort_val)
            
            print('missed opportunity val: ', missed_opportunity_val)
        
        return (positive_sort_val + negative_sort_val + missed_opportunity_val)/discount

    def one_step_fwd(self, x, U_traj):
        A_lin, _ = self.rollout_linearize_A(x)
        x_next = A_lin @ x + self.B * U_traj[0]
        return x_next
        
    def cash_aht(self, x, verbose = False, eval = False):
        _, sort_mat = self.rollout_linearize_A(x)
        return self.get_state_cost(x, sort_mat, 1, verbose, eval)
         
 