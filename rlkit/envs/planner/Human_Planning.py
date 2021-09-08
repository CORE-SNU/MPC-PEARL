import sys
import numpy as np
import casadi
import forcespro
import forcespro.nlp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os.path

nx = 2
nu = 2
N_human = 6
N_table = 5
T = 10  # length of planning horizon
nvar = (nx + nu) * N_human + nu * N_human
bounds = np.array([[-5, -5],
                   [5, 5]])


def dynamics(z):
    # single-integrator system
    # z_t = (u_t^{(1)},..., u_t^{(N)},    x_t^{(N)}, ...,x_t^{(N)},    \Delta u_t^{(1)},...,\Delta u_t^{(N)})
    # TODO : do we need casadi here?
    f = []
    u = casadi.SX.sym('u', nu, N_human)
    x = casadi.SX.sym('x', nx, N_human)
    deltau = casadi.SX.sym('deltau', nu, N_human)
    for i in range(N_human):
        u[:, i] = z[i * nu: (i + 1) * nu]  # velocity
        x[:, i] = z[N_human * nu + i * nx: N_human * nu + (i + 1) * nx]  # position
        deltau[:, i] = z[
                       N_human * nu + N_human * nx + i * nu: N_human * nu + N_human * nx + (i + 1) * nu]  # acceleration
        for j in range(nu):
            f.append(deltau[j, i] + u[j, i])
    for i in range(N_human):
        for j in range(nx):
            f.append(x[j, i] + u[j, i])

    return np.array(f)  # ddelta/dt = phi


def objective(z, p):
    # p : for goal trajectories
    u = casadi.SX.sym('u', nu, N_human)
    x = casadi.SX.sym('x', nx, N_human)
    deltau = casadi.SX.sym('deltau', nu, N_human)
    goal = casadi.SX.sym('goal', nx, N_human)
    obj = 0.
    Q = casadi.diag(p[nx * N_human + N_table * nx + N_human + N_table + 1:
                      nx * N_human + N_table * nx + N_human + N_table + 1 + nx])
    R = casadi.diag(p[nx * N_human + N_table * nx + N_human + N_table + 1 + nx:
                      nx * N_human + N_table * nx + N_human + N_table + 1 + nx + nu])
    for i in range(N_human):
        u[:, i] = z[i * nu: (i + 1) * nu]
        x[:, i] = z[N_human * nu + i * nx: N_human * nu + (i + 1) * nx]
        deltau[:, i] = z[N_human * nu + N_human * nx + i * nu: N_human * nu + N_human * nx + (i + 1) * nu]
        goal[:, i] = p[i * nx: (i + 1) * nx]
        # quadratic objective function
        obj += casadi.dot((Q @ (x[:, i] - goal[:, i])), x[:, i] - goal[:, i]) + casadi.dot((R @ deltau[:, i]),
                                                                                           deltau[:, i])
    return obj / N_human


def objectiveN(z, p):
    x = casadi.SX.sym('x', nx, N_human)
    goal = casadi.SX.sym('goal', nx, N_human)
    obj = 0.
    Qf = casadi.diag(p[nx * N_human + N_table * nx + N_human + N_table + nx + nu + 1:
                       nx * N_human + N_table * nx + N_human + N_table + 1 + nx + nu + nx])
    for i in range(N_human):
        x[:, i] = z[N_human * nu + i * nx: N_human * nu + (i + 1) * nx]
        goal[:, i] = p[i * nx: (i + 1) * nx]
        obj += casadi.dot((Qf @ (x[:, i] - goal[:, i])), x[:, i] - goal[:, i])
    return obj / N_human


def inequality(z, p):
    f = []
    x = casadi.SX.sym('x', nx, N_human)
    xo = casadi.SX.sym('xo', nx, N_table)
    w = casadi.SX.sym('w', N_human)
    wo = casadi.SX.sym('wo', N_table)
    for i in range(N_human):
        x[:, i] = z[N_human * nu + i * nx: N_human * nu + (i + 1) * nx]
        w[i] = p[N_human * nx + N_table * nx + N_table + i]
    for i in range(N_table):
        xo[:, i] = p[N_human * nx + i * nx: N_human * nx + (i + 1) * nx]
        wo[i] = p[N_human * nx + N_table * nx + i]
    safety = p[N_human * nx + N_table * nx + N_table + N_human]
    for i in range(N_human):
        for j in range(N_human):
            if i != j:
                f.append(casadi.dot(x[:, i] - x[:, j], x[:, i] - x[:, j]) - (w[i] + w[j] + safety) ** 2)

    for i in range(N_human):
        for j in range(N_table):
            f.append(casadi.dot(x[:, i] - xo[:, j], x[:, i] - xo[:, j]) - (w[i] + wo[j] + safety) ** 2)
    return f


def generate_pathplanner():
    # Model Definition
    # ----------------

    # Problem dimensions
    model = forcespro.nlp.SymbolicModel(T)
    model.nvar = nvar
    model.neq = nx * N_human + nu * N_human
    model.npar = nx * N_human + N_table * nx + N_human + N_table + 1 + nx + nu + nx
    model.nh[0] = 0
    model.nh[1:] = N_table * N_human + N_human * (N_human - 1)

    model.objective = objective
    model.objectiveN = objectiveN
    model.eq = dynamics
    # Indices on LHS of dynamical constraint - for efficiency reasons, make
    # sure the matrix E has structure [0 I] where I is the identity matrix.
    model.E = np.concatenate([np.eye(model.nvar - nu * N_human), np.zeros((model.nvar - nu * N_human, nu * N_human))],
                             axis=1)
    model.ubidx = range(model.nvar)
    model.lbidx = range(model.nvar)
    for k in range(1, model.N):
        model.ineq[k] = inequality
        model.hl[k] = np.zeros((N_table * N_human + N_human * (N_human - 1), 1))

    # Initial condition on vehicle states x
    model.xinitidx = range(N_human * nu,
                           N_human * nu + N_human * nx)     # use this to specify on which variables initial conditions

    # Solver generation
    # -----------------

    # Set solver options
    codeoptions = forcespro.CodeOptions('Human_Motion_Gen')
    codeoptions.nlp.ad_tool = 'casadi-3.5.1'
    codeoptions.maxit = 2000
    codeoptions.printlevel = 0
    codeoptions.optlevel = 3
    codeoptions.cleanup = False
    codeoptions.timing = 1
    codeoptions.overwrite = 1
    codeoptions.nlp.stack_parambounds = True
    codeoptions.parallel = 1
    solver = model.generate_solver(options=codeoptions)

    return solver

def updatePlots(Xout, human_rad, k):
    fig = plt.gcf()
    ax = fig.gca()
    objects = ax.findobj()

    for i in range(N_human):
        objects[i + N_table].remove()
        ax.get_lines().pop(-1).remove()

    for i in range(N_human):
        plt.plot(Xout[0, :k + 1, i], Xout[1, :k + 1, i], 'k-')  # robot trajectory
        ax.add_patch(plt.Circle((Xout[0, k, i], Xout[1, k, i]), human_rad, color='r'))  # robot position

    plt.pause(0.2)

def createPlot(x0, xtable, goal, human_rad, table_rad):
    """Creates a plot and adds the initial data provided by the arguments"""

    # Create empty plot
    plt.close('all')
    fig = plt.figure()
    plt.clf()
    ax = plt.gca()

    # Plot trajectory
    for i in range(N_table):
        ax.add_patch(plt.Circle((xtable[i][0], xtable[i][1]), table_rad, color='b'))  # tables
    for i in range(N_human):
        plt.plot(goal[i][0], goal[i][1], 'k.')  # goals
        plt.plot(x0[i][0], x0[i][1], 'k.')  # goals
        plt.plot([x0[i][0], goal[i][0]], [x0[i][1], goal[i][1]],
                 '--', color=(0.8, 0.8, 0.8))  # robot initial position to goal
    for i in range(N_human):
        plt.plot(x0[i][0], x0[i][1], 'k-')  # robot trajectory
        ax.add_patch(plt.Circle((x0[i][0], x0[i][1]), human_rad, color='r'))  # robot position

    plt.xlim(bounds[:, 0])
    plt.ylim(bounds[:, 1])
    ax.set_axisbelow(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_axisbelow(True)
    ax.grid(True)


def main():
    #  Code Generation
    if os.path.exists("Human_Motion_Gen"):
        # print("Solver Exists!")
        solver = forcespro.nlp.Solver.from_directory("./Human_Motion_Gen")
    else:
        solver = generate_pathplanner()

    # Simulation
    # ----------
    table_rad = 0.5  # table radius
    human_rad = 0.25  # human radius
    safe_rad = 0.05  # safety margin for human-to-human and human-to-table collision avoidance
    u_ub = []
    u_lb = []
    for t in range(N_human):
        lim = .25 * np.random.rand() + .05
        u_ub += [lim, lim]
        u_lb += [-lim, -lim]

    xmin, xmax = [bounds[0, 0] + human_rad, bounds[0, 1] + human_rad], [bounds[1, 0] - human_rad,
                                                                        bounds[1, 1] - human_rad]  # state limit
    deltaumin, deltaumax = [-0.05, -0.05], [0.05, 0.05]  # speed difference limit
    Q, Qf, R = np.array([1, 1]), np.array([1, 1]), np.array([0.01, 0.01])  # cost weights

    # rejection sampling
    xtable = []
    for i in range(N_table):
        flag = False
        while not flag:
            temp = bounds[0] + table_rad + (bounds[1] - bounds[0] - 2 * table_rad) * np.random.rand(nx)
            if i == 0:
                flag = True
                xtable.append(temp)
            elif all(((temp - xtable) ** 2).sum(1) > (2 * table_rad) ** 2):
                flag = True
                xtable.append(temp)

    x0 = []
    goal = []
    for i in range(N_human):
        flag = False
        while not flag:
            temp = bounds[0] + human_rad + (bounds[1] - bounds[0] - 2 * human_rad) * np.random.rand(nx)
            if i == 0:
                if all(((temp - xtable) ** 2).sum(1) > (human_rad + table_rad) ** 2):
                    flag = True
                    x0.append(temp)
            elif all(((temp - x0) ** 2).sum(1) > (2 * human_rad) ** 2) and all(
                    ((temp - xtable) ** 2).sum(1) > (human_rad + table_rad) ** 2):
                flag = True
                x0.append(temp)
        flag = False
        while not flag:
            temp = bounds[0] + human_rad + (bounds[1] - bounds[0] - 2 * human_rad) * np.random.rand(nx)
            if all(((temp - xtable) ** 2).sum(1) > (human_rad + table_rad) ** 2):
                flag = True
                goal.append(temp)

    Xout = np.zeros((nx, 41, N_human))
    Uout = np.zeros((nu, 40, N_human))
    Xout1 = np.zeros((nx, T, N_human))
    Uout1 = np.zeros((nu, T, N_human))

    # assert exitflag == 1, "bad exitflag"

    createPlot(x0, xtable, goal, human_rad, table_rad)
    for j in range(N_human):
        Xout[:, 0, j] = x0[j]

    fig = plt.gcf()
    ax = fig.gca()
    for t in range(40):
        xinit1 = []
        for j in range(N_human):
            xinit1.append(Xout[:, t, j])
        xinit = np.reshape(xinit1, (1, N_human * nx))
        # print(xinit)
        problem = {"x0": np.zeros((T, nvar)),
                   "xinit": xinit}
        problem["lb"] = np.concatenate(
            (u_lb + N_human * deltaumin, np.tile(u_lb + N_human * xmin + N_human * deltaumin, T - 1)))
        problem["ub"] = np.concatenate(
            (u_ub + N_human * deltaumax, np.tile(u_ub + N_human * xmax + N_human * deltaumax, T - 1)))

        problem["all_parameters"] = np.tile(np.concatenate((np.reshape(goal, (N_human * nx,)),
                                                            np.reshape(xtable, (N_table * nx,)),
                                                            np.repeat(table_rad, N_table),
                                                            np.repeat(human_rad, N_human),
                                                            np.array([safe_rad]), Q, R, Qf)), (T,))
        output, exitflag, info = solver.solve(problem)  # solve MPC
        print(output.keys())
        # print("FORCES took {} iterations and {} seconds to solve the problem.\n"\
        # .format(info.it, info.solvetime))

        for k in range(T):
            temp = output['x{0:02d}'.format(k + 1)]
            for j in range(N_human):
                Xout1[:, k, j] = temp[N_human * nu + j * nx: N_human * nu + (j + 1) * nx]
                Uout1[:, k, j] = temp[j * nu: (j + 1) * nu]
        for j in range(N_human):
            plt.plot(Xout1[0, :, j], Xout1[1, :, j], 'k-')

        Xout[:, t + 1, :] = Xout1[:, 1, :]

        Uout[:, t, :] = Uout1[:, 0, :]
        for j in range(N_human):
            ax.get_lines().pop(-1).remove()

        updatePlots(Xout, human_rad, t)
        print("updated")
        if t == 39:
            print("showed")
            plt.show()
        else:
            plt.draw()


def plan(xtable, x0, goal, ep_len, u_lb, u_ub):
    #  Code Generation
    if os.path.exists("Human_Motion_Gen"):
        # print("Solver Exists!")
        solver = forcespro.nlp.Solver.from_directory("./Human_Motion_Gen")
    else:
        solver = generate_pathplanner()

    # Simulation
    # ----------
    table_rad = 0.5  # table radius
    human_rad = 0.25  # human radius
    safe_rad = 0.05  # safety margin for human-to-human and human-to-table collision avoidance
    # umin, umax = [-0.08, -0.08], [0.08, 0.08] # control limits

    # randomly generated control constraint
    """
    u_ub = []
    u_lb = []
    for t in range(N_human):
        lim = .05 * np.random.rand() + .01
        u_ub += [lim, lim]
        u_lb += [-lim, -lim]
    """
    xmin, xmax = [bounds[0, 0] + human_rad, bounds[0, 1] + human_rad], [bounds[1, 0] - human_rad,
                                                                        bounds[1, 1] - human_rad]  # state limits
    deltaumin, deltaumax = [-0.05, -0.05], [0.05, 0.05]  # speed difference limit
    Q, Qf, R = np.array([1, 1]), np.array([1, 1]), np.array([0.01, 0.01])  # cost weights

    Xout = np.zeros((nx, ep_len + 1, N_human))
    Uout = np.zeros((nu, ep_len, N_human))
    Xout1 = np.zeros((nx, T, N_human))
    Uout1 = np.zeros((nu, T, N_human))

    # assert exitflag == 1, "bad exitflag"

    # createPlot(x0, xtable, goal, human_rad, table_rad)
    Xout[:, 0, :] = x0
    for t in range(ep_len):
        xinit1 = []
        for j in range(N_human):
            xinit1.append(Xout[:, t, j])
        xinit = np.reshape(xinit1, (1, N_human * nx))
        problem = {"x0": np.zeros((T, nvar)),
                   "xinit": xinit
                   }
        problem["lb"] = np.concatenate(
            (u_lb + N_human * deltaumin, np.tile(u_lb + N_human * xmin + N_human * deltaumin, T - 1)))
        problem["ub"] = np.concatenate(
            (u_ub + N_human * deltaumax, np.tile(u_ub + N_human * xmax + N_human * deltaumax, T - 1)))

        problem["all_parameters"] = np.tile(np.concatenate((np.reshape(goal, (N_human * nx,)),
                                                            np.reshape(xtable, (N_table * nx,)),
                                                            np.repeat(table_rad, N_table),
                                                            np.repeat(human_rad, N_human),
                                                            np.array([safe_rad]), Q, R, Qf)), (T,))
        output, exitflag, info = solver.solve(problem)
        # print("took {} iterations and {} seconds to solve the problem."\
        # .format(info.it, info.solvetime))

        for k in range(T):
            temp = output['x{0:02d}'.format(k + 1)]
            for j in range(N_human):
                Xout1[:, k, j] = temp[N_human * nu + j * nx: N_human * nu + (j + 1) * nx]
                Uout1[:, k, j] = temp[j * nu: (j + 1) * nu]
        Xout[:, t + 1, :] = Xout1[:, 1, :]
        Uout[:, t, :] = Uout1[:, 0, :]
        # updatePlots(Xout, human_rad, i)
        # print("updated")
        # if t == ep_len-1:
        #     print("showed")
        #     plt.show()
        # else:
        #     plt.draw()

    return Xout, Uout


if __name__ == "__main__":
    main()
