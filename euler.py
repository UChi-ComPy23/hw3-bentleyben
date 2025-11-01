"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d
from warnings import warn

class ForwardEuler(scipy.integrate.OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, vectorized = False, support_complex = False, h = None, **kwargs):
        #initialize the parent class
        super(ForwardEuler, self).__init__(fun, t0, y0, t_bound, vectorized, support_complex=support_complex, **kwargs)
        
        #store attributes
        if h is None:
            h = (t_bound - t0) / 100
        self.h = h
        self.direction = 1
        self.njev = 0
        self.nlu = 0       

    def _step_impl(self):
        t = self.t
        y = self.y
        h = self.h 

        function_eval = self.fun(t,y) #evaluate function at t and y
        y_new = y + h * function_eval #steps y forward based on h and evaluated function value 
        t_new = t + h #steps t forward h 

        self.t = t_new #sets t equal to new t
        self.y = y_new #sets y equal to new y

        if (self.t - self.t_bound) >= 0: #tests if the t is still within the boundary 
            self.status = "finished"

        return True, None

    def _dense_output_impl(self):
        return ForwardEulerOutput(self.t, self.y, self.h, self.fun(self.t, self.y))
    
class ForwardEulerOutput(DenseOutput):
    def __init__(self, t_old, y_old, h, f_eval):
        super().__init__(t_old, t_old + h) #initializes using old t and t (t = t_old + h)
        self.t_old = t_old 
        self.y_old = y_old
        self.h = h
        self.f_eval = f_eval

    def _call_impl(self, t):
        #y(t) = y_n + (t - t_n) * f(t_n, y_n)
        dt = t - self.t_old
        return self.y_old + dt * self.f_eval