from multiprocessing import Pool
import numpy as np

params = 0

class Engine(object):
    def __init__(self, parameters):
        self.parameters = parameters
    def __call__(self, idx):
        a = np.ones((10)) * idx
        np.savetxt(str(idx) + '.csv', a)
      
engine = Engine(params) 
inds = []
for i in range(10):
    inds.append(i)
    
           
with Pool() as pool:
    pool.map(engine, inds)