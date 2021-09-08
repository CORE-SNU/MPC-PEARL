import sys
import numpy as np
from scipy.special import erfinv
import casadi
import forcespro
import forcespro.nlp
import os.path
#import time

class MPC_forces:
    def __init__(self, nx=3, ny=2, nu=2, T=10, Q=None, Qf=None, R=None, S=None, ulb=None, uub=None, xlb=None, xub=None, deltaulb=None, deltauub=None, L=None, lr=None, N_table=3, N_human=0, dt=0.05, safe_rad = 0.1):
        self.ny = ny
        self.nx = nx
        self.nu = nu
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.S = S
        self.uub = uub
        self.ulb = ulb
        self.xub = xub[:2]
        self.xlb = xlb[:2]
        self.deltaulb = deltaulb
        self.deltauub = deltauub
        self.L = L
        self.lr = lr
        self.T = T
        self.N_table = N_table
        self.N_human = N_human
        self.dt = dt
        self.safe_rad = safe_rad
        self.N = 100
        self.delta = 0.01
        self.num_z = 1
        self.nvar = nx + nu + self.num_z
        if os.path.exists("MPC_forces"):
            # print("Solver Exists!")
            self.solver = forcespro.nlp.Solver.from_directory("./MPC_forces")
        else:
            self.solver = self.generate_pathplanner()
        
    def dynamics(self, z):
        # single-integrator system
        # z_t = (u_t^{(1)},..., u_t^{(N)},    x_t^{(N)}, ...,x_t^{(N)},    \Delta u_t^{(1)},...,\Delta u_t^{(N)})
        # TODO : do we need casadi here?
        u = casadi.SX.sym('u', self.nu, 1)
        x = casadi.SX.sym('x', self.nx, 1)
        #deltau = casadi.SX.sym('deltau', self.nu, 1)
        beta = casadi.SX.sym('beta')
        u = z[0:self.nu]
        x = z[self.nu+self.num_z:self.nu+self.num_z+self.nx]
        #deltau = z[self.nu+self.nx:self.nu+self.nx+self.nu]
        beta = casadi.atan(self.lr * casadi.tan(u[1]) / self.L)
        return casadi.vertcat(
                              #deltau[0]+u[0],
                              #deltau[1]+u[1],
                              x[0]+self.dt*u[0]*casadi.cos(beta + x[2]),
                              x[1]+self.dt*u[0]*casadi.sin(beta + x[2]),
                              x[2]+self.dt*u[0]*casadi.tan(u[1])*casadi.cos(beta) / self.L)
        '''
        return casadi.vertcat(deltau[0]+u[0],
                      deltau[1]+u[1],
                      x[0]+self.dt*u[0]*(1 - (beta + x[2])**2 / 2),
                      x[1]+self.dt*u[0]*(x[2] + beta),
                      x[2]+self.dt*u[0]*(x[2] * casadi.cos(beta) * u[1]) / self.L)
        '''

    def objective(self, z, p):
        # p : for goal trajectories
        u = casadi.SX.sym('u', self.nu, 1)
        x = casadi.SX.sym('x', self.nx, 1)
        
        #deltau = casadi.SX.sym('deltau', self.nu, 1)    
        goal = casadi.SX.sym('goal', self.ny, 1)
        obj = 0.
        
        u = z[0:self.nu]
        x = z[self.nu + self.num_z:self.nu+self.num_z +self.nx]
        #deltau = z[self.nu+self.nx:self.nu+self.nx+self.nu]
        goal = p[0:self.ny]
        Q = casadi.diag(p[self.ny:2*self.ny])
        R = casadi.diag(p[2*self.ny: 2*self.ny+self.nu])
        # quadratic objective function
        obj += casadi.dot((Q @ (x[0:self.ny] - goal)), x[0:self.ny] - goal) + casadi.dot((R @ u), u) 
        #+ casadi.dot((self.S @ deltau), deltau)
        return obj


    def objectiveN(self, z, p):
        x = casadi.SX.sym('x', self.nx, 1)
        goal = casadi.SX.sym('goal', self.ny, 1)
        obj = 0.
        x = z[self.nu+self.num_z:self.nu+self.num_z+self.nx]
        goal = p[0:self.ny]
        Qf = casadi.diag(p[2*self.ny+self.nu: 3*self.ny+self.nu])
        # quadratic objective function
        obj += casadi.dot((Qf @ (x[0:2] - goal)), x[0:2] - goal)
        return obj


#    def inequality(self, z, p):
#        alpha = 0.95
#        f = []
#        x = casadi.SX.sym('x', self.nx, 1)
#        x = z[self.nu : self.nu+self.nx]
#        z_cvar = z[self.nu+self.nx : self.nu+self.nx+self.N_human]
#        s = casadi.reshape(z[self.nu+self.nx+self.N_human:self.nu+self.nx+self.N_human+self.N_human*self.N],(self.N, self.N_human))
#        xo = casadi.SX.sym('xo', self.ny, self.N_table)
#        wo = casadi.SX.sym('wo', self.N_table)
#        sample = casadi.SX.sym('xh', self.ny, self.N_human, self.N)
#        wh = casadi.SX.sym('wh', self.N_human)
#        w = casadi.SX.sym('w', 1)
#        w = p[3*self.ny+self.nu+ self.N_table*self.ny]
#        #flag = p[-self.N_human:]
#
#        xo = casadi.reshape(p[3*self.ny+self.nu:3*self.ny+self.nu+self.N_table*self.ny],(self.ny, self.N_table))
#        wo = p[3*self.ny+self.nu+ self.N_table*self.ny + 1: 3*self.ny+self.nu+ self.N_table*self.ny + 1+self.N_table]
#
#        xh = casadi.reshape(p[3*self.ny+self.nu+self.ny*self.N_table+2+self.N_table+self.N_human : 3*self.ny+self.nu+self.ny*self.N_table+2+self.N_table+self.N_human + self.N_human * self.ny * self.N], self.ny, self.N*self.N_human)
#        sample = [xh[:,i*self.N_human:(i+1)*self.N_human] for i in range(self.N)] 
#            
#        wh = p[3*self.ny+self.nu+self.N_table*self.ny + 1 +self.N_table+1:3*self.ny+self.nu+self.N_table*self.ny + 1 +self.N_table+1+ self.N_human]
# 
#        safety = p[3*self.ny+self.nu + self.N_table*self.ny + 1 + self.N_table]
#        for j in range(self.N_table):
#            f.append(casadi.dot(x[0:2]-xo[:,j], x[0:2]-xo[:,j]) - (safety+w+wo[j])**2) 
#       
#        for j in range(self.N_human):
#            cvar_cost =  self.delta - z_cvar[j] - 1/(self.N*(1-alpha))*casadi.sum1(s[:,j])
#            f.append(cvar_cost)
#            for i in range(self.N):
#                pos_diff = x[0:2]-sample[i][:,j]
#                pos_diff_norm2 = casadi.dot(pos_diff, pos_diff)
#                f.append(s[i,j] - (w + wh[j] + safety)**2 + pos_diff_norm2 + z_cvar[j])
#        
#            #+ (1 - flag[j]) * pos_diff_norm2[0] - (safety+w+wh[j])**2)
#        return f
#    
    def inequality(self, z, p):
        alpha = 0.99
        f = []
        x = casadi.SX.sym('x', self.nx, 1)
        x = z[self.nu +self.num_z: self.nu+self.num_z+self.nx]
        z_cvar = z[self.nu: self.nu+self.num_z]
        xo = casadi.SX.sym('xo', self.ny, self.N_table)
        wo = casadi.SX.sym('wo', self.N_table)
        sample = casadi.SX.sym('xh', self.ny, self.N_human, self.N)
        wh = casadi.SX.sym('wh', self.N_human)
        w = casadi.SX.sym('w', 1)
        w = p[3*self.ny+self.nu+ self.N_table*self.ny]
        #flag = p[-self.N_human:]

        xo = casadi.reshape(p[3*self.ny+self.nu:3*self.ny+self.nu+self.N_table*self.ny],(self.ny, self.N_table))
        wo = p[3*self.ny+self.nu+ self.N_table*self.ny + 1: 3*self.ny+self.nu+ self.N_table*self.ny + 1+self.N_table]

        xh = casadi.reshape(p[3*self.ny+self.nu+self.ny*self.N_table+2+self.N_table+self.N_human : 3*self.ny+self.nu+self.ny*self.N_table+2+self.N_table+self.N_human + self.N_human * self.ny * self.N], self.ny, self.N*self.N_human)
        sample = [xh[:,i*self.N_human:(i+1)*self.N_human] for i in range(self.N)] 
            
        wh = p[3*self.ny+self.nu+self.N_table*self.ny + 1 +self.N_table+1:3*self.ny+self.nu+self.N_table*self.ny + 1 +self.N_table+1+ self.N_human]
 
        safety = p[3*self.ny+self.nu + self.N_table*self.ny + 1 + self.N_table]
        for j in range(self.N_table):
            f.append(casadi.dot(x[0:2]-xo[:,j], x[0:2]-xo[:,j]) - (safety+w+wo[j])**2) 
       
#        dist = casadi.SX.sym('dist', self.N, self.N_human)     
#        pos_diff_norm2 = casadi.SX.sym('pos_diff_norm2', self.N, self.N_human)
#        for j in range(self.N_human):
#            for i in range(self.N):
#                pos_diff_norm2[i,j] = casadi.dot(x[0:2]-sample[i][:,j],x[0:2]-sample[i][:,j])
#                dist[i,j] = (w + wh[j] + safety)**2 - pos_diff_norm2[i,j]
#            f.append(self.delta - z_cvar[j] - 1/(self.N)*1/(1-alpha)*casadi.sum1(casadi.fmax(dist[:,j]-z_cvar[j],0)))
                    #+ (1 - flag[j]) * pos_diff_norm2[0,j] - (safety+w+wh[j])**2)
            
        dist = casadi.SX.sym('dist', self.N, self.N_human)  
        maxit = casadi.SX.sym('maxit', self.N)
        for i in range(self.N):
            for j in range(self.N_human):
                dist[i,j] = (w + wh[j] + safety)**2 - casadi.dot(x[0:2]-sample[i][:,j], x[0:2]-sample[i][:,j])
            maxit[i] = casadi.mmax(dist[i,:])
        f.append(self.delta - z_cvar - 1/(self.N)*1/(1-alpha)*casadi.sum1(casadi.fmax(maxit-z_cvar,0)))
#            #+ (1 - flag[j]) * pos_diff_norm2[0] - (safety+w+wh[j])**2)
        return f


    def generate_pathplanner(self):
        # Model Definition
        # ----------------

        # Problem dimensions
        model = forcespro.nlp.SymbolicModel(self.T)
        model.nvar = self.nvar
        model.neq = self.nx
        #+self.nu
        model.npar = 3*self.ny+self.nu + self.N_table*self.ny + 1 + self.N_table + 1 + self.N_human + self.ny*self.N_human*self.N
        #+self.N_human
        model.nh[0] = 0;
        model.nh[1:] = self.N_table+self.num_z
    
        model.objective = self.objective
        model.objectiveN = self.objectiveN
        model.eq = self.dynamics
        # Indices on LHS of dynamical constraint - for efficiency reasons, make
        # sure the matrix E has structure [0 I] where I is the identity matrix.
        #model.E = np.concatenate([np.zeros((self.nx, self.nu)),np.eye(self.nx)], axis=1)
        model.E = np.concatenate([np.zeros((self.nx, self.nu + self.num_z)), np.eye(self.nx)], axis=1)
        model.ubidx = list(range(self.nu)) + list(range(self.nu+self.num_z,self.nu+self.num_z+self.nx-1))
        model.lbidx = list(range(self.nu)) + list(range(self.nu+self.num_z,self.nu+self.num_z+self.nx-1))
        #+list(range(self.nx+self.nu+self.N_human,self.nx+self.nu+self.N_human+self.N_human*self.N))
        for k in range(1, model.N):
           model.ineq[k] = self.inequality
           model.hl[k] = np.zeros(self.N_table+self.num_z)
    
        # Initial condition on vehicle states x
        model.xinitidx = range(self.nu+self.num_z, self.nu+self.num_z+self.nx); # use this to specify on which variables initial conditions


        # Solver generation
        # -----------------

        # Set solver options
        codeoptions = forcespro.CodeOptions('MPC_forces')
        codeoptions.nlp.ad_tool = 'casadi-3.5.1'
        codeoptions.maxit = 2000
        # codeoptions.maxit = 20000
#        codeoptions.nlp.TolStat = 1e-6     # inf norm tol. on stationarity
#        codeoptions.nlp.TolEq = 1e-6       # tol. on equality constraints
#        codeoptions.nlp.TolIneq = 1e-6      # tol. on inequality constraints
#        codeoptions.nlp.TolComp = 1e-6   # tol. on complementarity
#        codeoptions.accuracy.ineq = 1e-4
#        codeoptions.accuracy.eq = 1e-4
#        codeoptions.accuracy.mu = 1e-4
#        codeoptions.accuracy.rdgap = 1e-4
        codeoptions.printlevel = 0
        codeoptions.optlevel = 3
        codeoptions.cleanup = True
        codeoptions.timing = 1
        codeoptions.overwrite = 1
        #codeoptions.threadSafeStorage = True
        codeoptions.nlp.stack_parambounds = True
        codeoptions.forcenonconvex = 1
        codeoptions.noVariableElimination = 1
        codeoptions.parallel = 1
        codeoptions.nlp.linear_solver = 'symm_indefinite_fast'
        

        solver = model.generate_solver(options=codeoptions)

        return solver

    def plan_pol(self, x0, goal, Q, R, Qf, table_rad, human_rad, xtable_list, xhuman_pred, xhuman_pred_cov, count, flag):
        #flag = np.zeros(self.N_human)
        #print(dist_to_goal, max_dist, dist_to_goal/max_dist, Q)
        #start_time = time.time()
        self.deltaulb = []
        self.deltauub = []
        robot_rad = self.lr
        xtable = [xtable1.center for xtable1 in xtable_list]
        #xhuman_pred_cov = np.clip(xhuman_pred_cov, 1e-5, np.inf)
        
        samples = xhuman_pred + xhuman_pred_cov**0.5 * np.random.randn(self.N, self.T, self.N_human, self.ny)
        samples = np.transpose(samples, (1, 0, 2, 3))
        samples = np.reshape(samples, (self.T, self.N*self.N_human*self.ny))
        
        #xtable[3] = xtable[3] - [0, 0.25]
        # print(table_rad + robot_rad + self.safe_rad)
        xinit = x0
        problem = {"x0": np.zeros((self.T, self.nvar)),
                   "xinit": xinit}
        
        #zlb = (1)*[-np.inf]
        #zub = (1)*[np.inf]
        #problem["lb"] = np.concatenate((self.ulb+self.deltaulb , np.tile(self.ulb + self.xlb + self.deltaulb, self.T - 1)))
        #problem["ub"] = np.concatenate((self.uub+self.deltauub, np.tile(self.uub + self.xub + self.deltauub, self.T - 1)))
        problem["lb"] = np.tile(self.ulb + self.xlb + self.deltaulb, self.T)
        problem["ub"] = np.tile(self.uub + self.xub + self.deltauub, self.T)
        #print(xhuman_pred, np.reshape(xhuman_pred,(T,N_human*2)))
        tiled = np.tile(np.concatenate((goal, Q, R, Qf, # 2
                                 np.reshape(xtable,(self.N_table*self.ny,)), 
                                 np.array([robot_rad]),
                                 np.repeat(table_rad,self.N_table),
                                 np.array([self.safe_rad]),
                                 np.repeat(human_rad,self.N_human))), (self.T,1))
             
        stacked = np.hstack((tiled, samples))
#                             np.tile(flag, (self.T, 1))))

        problem["all_parameters"] = np.reshape(stacked,(stacked.shape[0]*stacked.shape[1]))
        #problem["all_parameters"] = np.tile(goal, (T,))
        output, exitflag, info = self.solver.solve(problem)
       
        
        Xout = np.zeros((self.nx, self.T))
        Uout = np.zeros((self.nu, self.T))
        for t in range(self.T):
            temp = output['x{0:02d}'.format(t+1)]
            Xout[:, t] = temp[self.nu+self.num_z:self.nu+self.num_z+self.nx]      # position of human j at time i
            Uout[:, t] = temp[0:self.nu]
        #print("* t=%d: %d - %f sec" % (count, exitflag, info.solvetime))
        #end_time = time.time()
        '''
        
        if not exitflag == 1:
            if any(flag):
                print('GP error')
            else:
                print('Infeasible')
        '''
        
        return Xout, Uout[:, 0], exitflag

