#models
import numpy as np

class environ():
    """
    Environmental model
    Source: https://www.sfu.ca/~ssurjano/environ.html
    params: [M,D, L, t]
    """
    def __init__(self, params, s, t):
        self.M = params[0]
        self.D = params[1]
        self.L = params[2]
        self.tau = params[3]
        self.s = s
        self.t = t
        self.ds = len(s)
        self.dt = len(t)
        self.dY = self.ds*self.dt
        #output Y
        self.Y = np.zeros([self.ds,self.dt])


    
    def f(self,s,t) -> float:
        def C(s,t)-> float:
            a1 =self.M/np.sqrt(4*np.pi*self.D*t)
            a2 = np.exp(-(np.square(s))/(4*self.D*t)) 
            if (self.tau < t):
                a3 = self.M/np.sqrt(4*np.pi*self.D*(t-self.tau))
                a4 = np.exp(-(np.square((s-self.L)))/(4*self.D*(t-self.tau)))
            else:
                a3 = 0
                a4 = 0
            c = a1*a2 + a3*a4
            return c
        f_eval = np.sqrt(4*np.pi)*C(s,t)
        return f_eval
    
    def run(self)-> np.ndarray:
        for i in range(self.ds):
            for j in range(self.dt):
                self.Y[i,j] = self.f(self.s[i],self.t[j])
        y_vector = self.Y.flatten()
        return y_vector
         

if __name__ == "__main__":
    print("This is the models.py file.")
         