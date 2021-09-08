import sys
import numpy as np
import casadi
import forcespro
import forcespro.nlp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os.path

ny = 2
nx = 3
nu = 2
T = 15
nvar = nx+2*nu
bounds = np.array([[-1, -1, -np.pi], [1, 1, np.pi]])
N_table=3
N_human=0
dt = 0.1

def dynamics(z):
    # single-integrator system
    # z_t = (u_t^{(1)},..., u_t^{(N)},    x_t^{(N)}, ...,x_t^{(N)},    \Delta u_t^{(1)},...,\Delta u_t^{(N)})
    # TODO : do we need casadi here?
    u = casadi.SX.sym('u', nu, 1)
    x = casadi.SX.sym('x', nx, 1)
    deltau = casadi.SX.sym('deltau', nu, 1)
    u = z[0:nu]
    x = z[nu:nu+nx]
    deltau = z[nu+nx:nu+nx+nu]
    f = []
    for j in range(nu):
        f.append(deltau[j]+u[j])
    f.append(x[0]+dt*u[0]*casadi.cos(x[2]))
    f.append(x[1]+dt*u[0]*casadi.sin(x[2]))
    f.append(x[2]+dt*u[1])
    return np.array(f)                           # ddelta/dt = phi


def objective(z, p):
    # p : for goal trajectories
    u = casadi.SX.sym('u', nu, 1)
    x = casadi.SX.sym('x', nx, 1)
    deltau = casadi.SX.sym('deltau', nu, 1)    
    goal = casadi.SX.sym('goal', ny, 1)
    obj = 0.
    Q = np.eye(ny)
    R = 0.01*np.eye(nu)
    u = z[0:nu]
    x = z[nu:nu+nx]
    deltau = z[nu+nx:nu+nx+nu]
    goal = p[0: ny]
    # quadratic objective function
    obj += casadi.dot((Q @ (x[0:2] - goal)), x[0:2] - goal) + casadi.dot((R @ deltau), deltau)
    return obj


def objectiveN(z, p):
    x = casadi.SX.sym('x', nx, 1)
    goal = casadi.SX.sym('goal', ny, 1)
    obj = 0.
    Qf = np.eye(ny)
    R = 0.01*np.eye(nu)
    x = z[nu:nu+nx]
    goal = p[0:ny]
    # quadratic objective function
    obj += casadi.dot((Qf @ (x[0:2] - goal)), x[0:2] - goal)
    return obj


def inequality(z, p):
    f = []
    x = casadi.SX.sym('x', nx, 1)
    x = z[nu:nu+nx]
    xo = casadi.SX.sym('xo', ny, N_table)
    wo = casadi.SX.sym('wo', N_table)
    xh = casadi.SX.sym('xh', ny, N_human)
    wh = casadi.SX.sym('wh', N_human)
    w = casadi.SX.sym('w', 1)
    w = p[ny + N_table*ny]
    for i in range(N_table):
        xo[:, i] = p[ny + i*ny : ny + (i+1)*ny]
        wo[i] = p[ny + N_table*ny + 1 + i]
    for i in range(N_human):
        xh[:, i] = p[ny + ny*N_table+2+N_table+N_human+i*ny : ny + ny*N_table+2+N_table+N_human + (i+1)*ny]
        wh[i] = p[ny + N_table*ny + 1 +N_table+1+ i]
    safety = p[ny + N_table*ny + 1 + N_table]
    for j in range(N_table):
        f.append(casadi.dot(x[0:2]-xo[:,j], x[0:2]-xo[:,j]) - (w+wo[j]+safety)**2) 
    for j in range(N_human):
        f.append(casadi.dot(x[0:2]-xh[:,j], x[0:2]-xh[:,j]) - (w+wh[j]+safety)**2) 
    return f


def generate_pathplanner():
    # Model Definition
    # ----------------

    # Problem dimensions
    model = forcespro.nlp.SymbolicModel(T)
    model.nvar = nvar
    model.neq = nx+nu
    model.npar = ny + N_table*ny + 1 + N_table + 1 + N_human + ny*N_human;
    model.nh[0] = 0;
    model.nh[1:] = N_table+N_human
    
    model.objective = objective
    model.objectiveN = objectiveN
    model.eq = dynamics
    # Indices on LHS of dynamical constraint - for efficiency reasons, make
    # sure the matrix E has structure [0 I] where I is the identity matrix.
    model.E = np.concatenate([np.eye(nx+nu),np.zeros((nx+nu,nu))], axis=1)
    model.ubidx = range(model.nvar)
    model.lbidx = range(model.nvar)
    for k in range(1, model.N):
        model.ineq[k] = inequality
        model.hl[k] = np.zeros((N_table+N_human, 1))
    
    # Initial condition on vehicle states x
    model.xinitidx = range(nu, nu+nx); # use this to specify on which variables initial conditions


    # Solver generation
    # -----------------

    # Set solver options
    codeoptions = forcespro.CodeOptions('MPC_policy')
    codeoptions.nlp.ad_tool = 'casadi-3.5.1'
    codeoptions.maxit = 20000
    codeoptions.printlevel = 0
    codeoptions.optlevel = 3
    codeoptions.cleanup = False
    codeoptions.timing = 1
    codeoptions.overwrite = 1
    codeoptions.nlp.stack_parambounds = True

    solver = model.generate_solver(options=codeoptions)

    return solver


def updatePlots_pol(Xout):

    fig = plt.gcf()
    ax = fig.gca()
    #objects = ax.findobj() 

    #objects[N_table].remove()
    #ax.get_lines().pop(-1).remove()
    plt.plot(Xout[0,:],Xout[1,:], 'k-')                                # robot trajectory
    #ax.add_patch(plt.Circle((Xout[0, k], Xout[1, k]), 0.01,color='r'))     # robot position

    plt.pause(1)


def createPlot(x0, xtable, goal, robot_rad, table_rad):
    """Creates a plot and adds the initial data provided by the arguments"""

    # Create empty plot
    plt.close('all')
    fig = plt.figure()
    plt.clf()
    ax = plt.gca()

    # Plot trajectory
    for i in range(N_table):
        ax.add_patch(plt.Circle((xtable[i][0], xtable[i][1]),table_rad,color='b'))      # tables
    plt.plot(goal[0], goal[1], 'k.')                                            # goals
    plt.plot(x0[0], x0[1], 'k.')                                               # goals
    plt.plot([x0[0], goal[0]], [x0[1], goal[1]],
                 '--', color=(0.8, 0.8, 0.8))                                           # robot initial position to goal
    plt.plot(x0[0], x0[1], 'k-')                                              # robot trajectory
    ax.add_patch(plt.Circle((x0[0], x0[1]), robot_rad, color='r'))            # robot position

    plt.xlim(bounds[:,0])
    plt.ylim(bounds[:,1])
    ax.set_axisbelow(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_axisbelow(True)
    ax.grid(True)


def main():
    #  Code Generation
    if os.path.exists("MPC_policy"):
        # print("Solver Exists!")
        solver = forcespro.nlp.Solver.from_directory("./MPC_policy")
    else:
        solver = generate_pathplanner()

    # Simulation
    # ----------
    robot_rad = 0 # human radius

    safe_rad = 0.01 # safety margin for human-to-human and human-to-table collision avoidance

    # umin, umax = [-0.08, -0.08], [0.08, 0.08] # control limits
    xtable = [xtable1.center for xtable1 in xtable_list]
    table_rad = xtable_list[0].radii
    human_pos = [human1.center for human1 in human_list]
    human_vel = [human1.vel for human1 in human_list]
    human_rad = human_list[0].radii
    #print(human_vel)
    xmin, xmax = [bounds[0,0] + robot_rad, bounds[0,1] + robot_rad], [bounds[1,0] - robot_rad, bounds[1,1] - robot_rad] # state limits
    umin, umax = [-0.08, -0.08], [0.08, 0.08]                    # speed difference limit
    deltaumin, deltaumax = [-0.02, -0.02], [0.02, 0.02] 
    """
    # rejection sampling
    xtable = []
    for i in range(N_table):
        flag = False
        while not flag:
            temp = bounds[0] + table_rad + (bounds[1] - bounds[0] - 2*table_rad)*np.random.rand(nx)
            if i == 0:
                flag = True
                xtable.append(temp)               
            elif all(((temp - xtable)**2).sum(1) > (2*table_rad)**2):
                flag = True
                xtable.append(temp)               

    x0 = []
    goal = []
    for i in range(N_human):
        flag = False
        while not flag:
            temp = bounds[0] + robot_rad + (bounds[1] - bounds[0] - 2*robot_rad)*np.random.rand(nx)
            if i == 0:
                if all(((temp - xtable)**2).sum(1) > (robot_rad + table_rad)**2):
                    flag = True
                    x0.append(temp)               
            elif all(((temp - x0)**2).sum(1) > (2*robot_rad)**2) and all(((temp - xtable)**2).sum(1) > (robot_rad + table_rad)**2):
                flag = True
                x0.append(temp) 
        flag = False
        while not flag:
            temp = bounds[0] + robot_rad + (bounds[1] - bounds[0] - 2*robot_rad)*np.random.rand(nx)
            if all(((temp - xtable)**2).sum(1) > (robot_rad + table_rad)**2):
                flag = True
                goal.append(temp)               
    """
    xinit = x0
    problem = {"x0": np.zeros((T, nvar)),
               "xinit": xinit}
    problem["lb"] = np.concatenate(np.tile(umin + xmin + deltaumin, T))
    problem["ub"] = np.concatenate(np.tile(umax + xmax + deltaumax, T))

    problem["all_parameters"] = np.tile(np.concatenate((goal,
                                                        np.reshape(xtable,(N_table*nx,)),
                                                        np.array([robot_rad]),
                                                        np.repeat(table_rad,N_table),
                                                        np.array([safe_rad]))), (T,))

    output, exitflag, info = solver.solve(problem)
    # assert exitflag == 1, "bad exitflag"
    print("FORCES took {} iterations and {} seconds to solve the problem.\n"\
        .format(info.it, info.solvetime))
    
    #createPlot(x0, xtable, goal, robot_rad, table_rad)
    """
    Xout = np.zeros((nx, T, N_human))
    temp = np.zeros((nvar, T))
    for i in range(T):
        temp= output['x{0:02d}'.format(i+1)]
        Xout[:, i] = temp[nu: nu+nx]
        if i == T-1:
            plt.show()
        else:
            plt.draw()
    """

def plan_pol(x0, xobs_list, ntable, nhuman, goal):
#  Code Generation
    N_table = ntable
    N_human = nhuman

    if os.path.exists("MPC_policy"):
        # print("Solver Exists!")
        solver = forcespro.nlp.Solver.from_directory("./MPC_policy")
    else:
        solver = generate_pathplanner()

    # Simulation
    # ----------
    #table_rad = 0.1 # table radius
    robot_rad = 0 # human radius
    safe_rad = 0.1 # safety margin for human-to-human and human-to-table collision avoidance
    # umin, umax = [-0.08, -0.08], [0.08, 0.08] # control limits
    xtable_list = xobs_list[0:N_table]
    xtable = [xtable1.center for xtable1 in xtable_list]
    table_rad = xtable_list[0].radii
    
    if N_human==0:
       xhuman_list = [];
       xhuman = []
       xhuman_rad = [];
       xhuman_pred = np.zeros((T,N_human,2));
    else:
       xhuman_list = xobs_list[N_table:N_table+N_human]
       #print(xhuman_list)
       xhuman = [xhuman1.center for xhuman1 in xhuman_list]
       xhuman_rad = xhuman_list[0].radii
       xhuman_vel = [xhuman1.vel for xhuman1 in xhuman_list]
       xhuman_pred = np.zeros((T, N_human,2));
       for i in range(N_human):
          xhuman_pred[0,i,:]=np.array(xhuman[i])
          for k in range(T-1):
             xhuman_pred[k+1,i,:] = xhuman_pred[k,i,:]+xhuman_vel[i]
       #print(xhuman_pred)

    xmin, xmax = [bounds[0,0] + robot_rad, bounds[0,1] + robot_rad, bounds[0,2]], [bounds[1,0] - robot_rad, bounds[1,1] - robot_rad, bounds[1,2]] # state limits
    umin, umax = [-1, -0.8], [1, 0.8]                    # speed difference limit
    deltaumin, deltaumax = [-0.1, -0.1], [0.1, 0.1]
    
    xinit = x0
    problem = {"x0": np.zeros((T, nvar)),
               "xinit": xinit}
    problem["lb"] = np.concatenate((umin+deltaumin , np.tile(umin + xmin + deltaumin, T - 1)))
    problem["ub"] = np.concatenate((umax+deltaumax, np.tile(umax + xmax + deltaumax, T - 1)))

    #print(problem["x0"], problem["xinit"], problem["lb"], problem["ub"])
    
  
    #print(xhuman_pred, np.reshape(xhuman_pred,(T,N_human*2)))
    tiled = (np.tile(np.concatenate((goal,
                                 np.reshape(xtable,(N_table*ny,)),
                                 np.array([robot_rad]),
                                 np.repeat(table_rad,N_table),
                                 np.array([safe_rad]),
                                 np.repeat(xhuman_rad,N_human))), (T,1)))
                                 
    stacked=np.hstack((tiled,np.reshape(xhuman_pred,(T,N_human*2))))
    problem["all_parameters"] = np.reshape(stacked,(stacked.shape[0]*stacked.shape[1]))
    
    #problem["all_parameters"] = np.tile(goal, (T,))
    output, exitflag, info = solver.solve(problem)
    # assert exitflag == 1, "bad exitflag"
    # print("FORCES took {} iterations and {} seconds to solve the problem.".format(info.it, info.solvetime))
    
    # createPlot(x0, xtable, goal, robot_rad, table_rad)

    Xout = np.zeros((nx, T))
    Uout = np.zeros((nu, T))
    for t in range(T):
        temp = output['x{0:02d}'.format(t+1)]
        Xout[:, t] = temp[nu:nu+nx]      # position of human j at time i
        Uout[:, t] = temp[0:nu]
        # updatePlots(Xout, robot_rad, i)
        # if i == 19:
        #     plt.show()
        # else:
        #   plt.draw()
    return Xout, Uout


if __name__ == "__main__":
    main()
