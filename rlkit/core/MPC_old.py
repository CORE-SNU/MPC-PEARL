import sys
import numpy as np
import casadi
import forcespro
import forcespro.nlp
import os.path

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
        self.xub = xub
        self.xlb = xlb
        self.deltaulb = deltaulb
        self.deltauub = deltauub
        self.L = L
        self.lr = lr
        self.T = T
        self.N_table = N_table
        self.N_human = N_human
        self.dt = dt
        self.safe_rad = safe_rad
        self.nvar = nx + 2*nu
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
        deltau = casadi.SX.sym('deltau', self.nu, 1)
        beta = casadi.SX.sym('beta')
        u = z[0:self.nu]
        x = z[self.nu:self.nu+self.nx]
        deltau = z[self.nu+self.nx:self.nu+self.nx+self.nu]
        beta = casadi.atan(self.lr * casadi.tan(u[1]) / self.L)
        return casadi.vertcat(deltau[0]+u[0],
                              deltau[1]+u[1],
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
        deltau = casadi.SX.sym('deltau', self.nu, 1)    
        goal = casadi.SX.sym('goal', self.ny, 1)
        obj = 0.
        
        u = z[0:self.nu]
        x = z[self.nu:self.nu+self.nx]
        deltau = z[self.nu+self.nx:self.nu+self.nx+self.nu]
        goal = p[0:self.ny]
        # quadratic objective function
        obj += casadi.dot((self.Q @ (x[0:self.ny] - goal)), x[0:self.ny] - goal) + casadi.dot((self.R @ u), u) + casadi.dot((self.S @ deltau), deltau)
        return obj


    def objectiveN(self, z, p):
        x = casadi.SX.sym('x', self.nx, 1)
        goal = casadi.SX.sym('goal', self.ny, 1)
        obj = 0.
        x = z[self.nu:self.nu+self.nx]
        goal = p[0:self.ny]
        # quadratic objective function
        obj += casadi.dot((self.Qf @ (x[0:2] - goal)), x[0:2] - goal)
        return obj


    def inequality(self, z, p):
        f = []
        x = casadi.SX.sym('x', self.nx, 1)
        x = z[self.nu:self.nu+self.nx]
        xo = casadi.SX.sym('xo', self.ny, self.N_table)
        wo = casadi.SX.sym('wo', self.N_table)
        xh = casadi.SX.sym('xh', self.ny, self.N_human)
        wh = casadi.SX.sym('wh', self.N_human)
        w = casadi.SX.sym('w', 1)
        w = p[self.ny + self.N_table*self.ny]
        for i in range(self.N_table):
            xo[:, i] = p[self.ny + i*self.ny : self.ny + (i+1)*self.ny]
            wo[i] = p[self.ny + self.N_table*self.ny + 1 + i]
        for i in range(self.N_human):
            xh[:, i] = p[self.ny + self.ny*self.N_table+2+self.N_table+self.N_human+i*self.ny : self.ny + self.ny*self.N_table+2+self.N_table+self.N_human + (i+1)*self.ny]
            wh[i] = p[self.ny + self.N_table*self.ny + 1 +self.N_table+1+ i]
        safety = p[self.ny + self.N_table*self.ny + 1 + self.N_table]
        for j in range(self.N_table):
            f.append(casadi.dot(x[0:2]-xo[:,j], x[0:2]-xo[:,j]) - (1.2*(w+wo[j]))**2)
        for j in range(self.N_human):
            f.append(casadi.dot(x[0:2]-xh[:,j], x[0:2]-xh[:,j]) - (1.2*(w+wh[j]))**2)
        return f


    def generate_pathplanner(self):
        # Model Definition
        # ----------------

        # Problem dimensions
        model = forcespro.nlp.SymbolicModel(self.T)
        model.nvar = self.nvar
        model.neq = self.nx+self.nu
        model.npar = self.ny + self.N_table*self.ny + 1 + self.N_table + 1 + self.N_human + self.ny*self.N_human;
        model.nh[0] = 0;
        model.nh[1:] = self.N_table+self.N_human
    
        model.objective = self.objective
        model.objectiveN = self.objectiveN
        model.eq = self.dynamics
        # Indices on LHS of dynamical constraint - for efficiency reasons, make
        # sure the matrix E has structure [0 I] where I is the identity matrix.
        model.E = np.concatenate([np.eye(self.nx+self.nu),np.zeros((self.nx+self.nu,self.nu))], axis=1)
        model.ubidx = range(model.nvar)
        model.lbidx = range(model.nvar)
        for k in range(1, model.N):
           model.ineq[k] = self.inequality
           model.hl[k] = np.zeros((self.N_table+self.N_human, 1))
    
        # Initial condition on vehicle states x
        model.xinitidx = range(self.nu, self.nu+self.nx); # use this to specify on which variables initial conditions


        # Solver generation
        # -----------------

        # Set solver options
        codeoptions = forcespro.CodeOptions('MPC_forces')
        codeoptions.nlp.ad_tool = 'casadi-3.5.1'
        codeoptions.maxit = 200000
        # codeoptions.maxit = 20000
        codeoptions.printlevel = 0
        codeoptions.optlevel = 3
        codeoptions.cleanup = False
        codeoptions.timing = 1
        codeoptions.overwrite = 1
        codeoptions.threadSafeStorage = True
        codeoptions.nlp.stack_parambounds = True

        solver = model.generate_solver(options=codeoptions)

        return solver

    def plan_pol(self, x0, goal, table_rad, human_rad, xtable_list, xhuman_pred, count):
        robot_rad = self.L
        xtable = [xtable1.center for xtable1 in xtable_list]
        
        xinit = x0
        problem = {"x0": np.zeros((self.T, self.nvar)),
                   "xinit": xinit}
        problem["lb"] = np.concatenate((self.ulb+self.deltaulb , np.tile(self.ulb + self.xlb + self.deltaulb, self.T - 1)))
        problem["ub"] = np.concatenate((self.uub+self.deltauub, np.tile(self.uub + self.xub + self.deltauub, self.T - 1)))
  
        #print(xhuman_pred, np.reshape(xhuman_pred,(T,N_human*2)))
        tiled = (np.tile(np.concatenate((goal, # 2
                                 np.reshape(xtable,(self.N_table*self.ny,)), 
                                 np.array([robot_rad]),
                                 np.repeat(table_rad,self.N_table),
                                 np.array([self.safe_rad]),
                                 np.repeat(human_rad,self.N_human))), (self.T,1)))
                                  
        stacked = np.hstack((tiled,np.reshape(xhuman_pred,(self.T, self.N_human*2))))

        problem["all_parameters"] = np.reshape(stacked,(stacked.shape[0]*stacked.shape[1]))

        #problem["all_parameters"] = np.tile(goal, (T,))
        output, exitflag, info = self.solver.solve(problem)
        #assert exitflag == 1, "bad exitflag"        
        # print("* t=%d: %d - %f sec" % (count, exitflag, info.solvetime))
    
        Xout = np.zeros((self.nx, self.T))
        Uout = np.zeros((self.nu, self.T))
        for t in range(self.T):
            temp = output['x{0:02d}'.format(t+1)]
            Xout[:, t] = temp[self.nu:self.nu+self.nx]      # position of human j at time i
            Uout[:, t] = temp[0:self.nu]

        # print(Uout)
        
        return Xout, Uout[:, 0], exitflag

