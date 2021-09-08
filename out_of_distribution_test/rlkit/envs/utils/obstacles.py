import numpy as np


class Obstacle:
    def __init__(self):
        self.is_static = None

    def contains(self, pt) -> bool:
        # implementation of membership operation
        # return True if a given point belongs to the obstacle
        raise NotImplementedError

    def entry_time(self, pt, ray) -> float:
        # return the first hitting time of the ray
        # $\inf\{t \geq 0 : p_t \in O \}$
        # In general, computation of the entry time reduces to a simple convex optimization problem when $O$ is convex
        # This is because $p_t$ is affine w.r.t. $t$
        raise NotImplementedError


class DynamicObstacle(Obstacle):
    def __init__(self, *args):
        super(DynamicObstacle, self).__init__()
        self.is_static = False

    def sim(self):
        # one-step simulation of dynamic obstacle
        # must be implemented if the obstacle is dynamic
        raise NotImplementedError

    def contains(self, pt) -> bool:
        raise NotImplementedError

    def entry_time(self, pt, ray) -> float:
        raise NotImplementedError


class StaticObstacle(Obstacle):
    def __init__(self, *args):
        super(StaticObstacle, self).__init__()
        self.is_static = True

    def contains(self, pt) -> bool:
        return Obstacle.contains(self, pt)

    def entry_time(self, pt, ray) -> float:
        return Obstacle.entry_time(self, pt, ray)


class CircularObstacle(Obstacle):
    def __init__(self, center, radii):
        super(CircularObstacle, self).__init__()
        self.center = center
        self.radii = radii

    def contains(self, pt) -> bool:
        return True if ((pt - self.center) ** 2).sum() <= (1.3*self.radii) ** 2 else False
        
    def contains_mpc(self, pt) -> bool:
        return True if ((pt - self.center) ** 2).sum() <= (2*self.radii) ** 2 else False
        
    def entry_time(self, pt, ray) -> float:
        # computation of hitting time when the obstacle is simply a circle
        # the implementation is independent of the dimension of the ambient space
        if self.contains(pt):
            # if the starting point is in the circle, the entry time is 0 by def
            return 0.
        a = (ray**2).sum()
        b = ((self.center - pt) * ray).sum()
        c = ((self.center - pt) ** 2).sum() - self.radii ** 2

        d = b**2 - a * c
        if d < 0:
            # if discriminant = 0, the ray does not meet the circle
            return np.inf
        else:
            # solve quadratic equation to find out the entry time
            # a t^2 - 2b t + c = 0
            # if all solutions are negative, the entry time is $\infty$ by def
            sol = [(b - d**.5) / a, b + d**.5 / a]
            return min([t for t in sol if t > 0], default=np.inf)


class Human(CircularObstacle, DynamicObstacle):
    # person whose motion is modelled as random process
    def __init__(self, center, radii):
        super(Human, self).__init__(center, radii)
        self.vel = None
        self.pos_curve = None
        self.vel_curve = None
        self.step = None
        self.reverse = False

    def set_path(self, x_path, u_path=None):
        # set pre-computed trajectory
        self.pos_curve = x_path     # shape = (2, T + 1) : [p(0), ... , p(T-1), p(T)]
        self.vel_curve = u_path     # shape = (2, T) :     [v(0), ... , v(T-1)]

    def sim(self):
        # $dX_t = v dt + \sigma dBt$
        if not self.reverse:
            self.step += 1
            if self.step >= self.pos_curve.shape[1]:
                self.reverse = True
                self.step -= 1
        else:
            self.step -= 1
            if self.step < 0:
                self.reverse = False
                self.step += 1
        self.center = self.pos_curve[:, self.step]
        if self.vel_curve is not None:
            self.vel = self.vel_curve[:, self.step]

    def init(self):
        self.step = 0
        self.reverse = False
        self.center = self.pos_curve[:, self.step]

    def contains(self, pt) -> bool:
        return CircularObstacle.contains(self, pt)

    def entry_time(self, pt, ray) -> float:
        return CircularObstacle.entry_time(self, pt, ray)


class Table(CircularObstacle, StaticObstacle):
    # table modelled as circular obstacle
    def __init__(self, center, radii):
        super(Table, self).__init__(center, radii)


def intersect(o1: Obstacle, o2: Obstacle) -> bool:
    if isinstance(o1, CircularObstacle) and isinstance(o2, CircularObstacle):
        c1, r1 = o1.center, o1.radii
        c2, r2 = o2.center, o2.radii
        return True if ((c1 - c2)**2).sum() <= (r1 + r2)**2 else False
    else:
        raise NotImplementedError


if __name__ == '__main__':

    p = np.array([-1., -.8])
    r = .1
    v = np.array([.3, 1.])
    sig = .2
    trj = np.zeros((20, 2))
    person = Human(center=p, radii=r)

    pp = np.array([-.8, -.2])
    rr = .2
    table = Table(center=pp, radii=rr)
    for i in range(20):
        # print(person.center)
        print(intersect(person, table))
        trj[i] = person.center
        person.sim()

    import matplotlib.pyplot as plt

    fig = plt.gcf()
    ax = fig.gca()
    plt.plot(trj[:, 0], trj[:, 1])
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    obstacle = plt.Circle(table.center,
                          table.radii,
                          color='tab:gray',
                          )
    ax.add_artist(obstacle)
    # plt.show()

