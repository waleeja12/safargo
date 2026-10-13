
import numpy as np

ZONES = ["Saddar", "Committee Chowk", "Satellite Town", "Murree Rd (PMAS-AAUR)", "Chaklala",
         "Bahria Town", "Faizabad", "Islamabad F-7", "Islamabad G-10", "Dhok Hassu"]
NZ = len(ZONES)
COORD = np.array([[0,0],[2,2],[4,3],[6,1],[8,-1],[12,-3],[6,6],[9,10],[11,8],[3,-3]], float)
DIST = np.sqrt(((COORD[:, None] - COORD[None]) ** 2).sum(-1))   # distance in km

# Demand per zone per 5-min step for 4 time blocks: morning, midday, evening, night
BASE = np.array([
    [1.0, 1.8, 2.2, 0.8],   # Saddar
    [0.8, 1.2, 1.6, 0.6],   # Committee Chowk
    [0.7, 1.0, 1.2, 0.5],   # Satellite Town
    [2.4, 1.0, 0.9, 0.2],   # Murree Rd (university)
    [0.5, 0.7, 0.8, 0.4],   # Chaklala
    [0.4, 0.6, 0.9, 0.7],   # Bahria Town
    [0.9, 0.9, 1.1, 0.4],   # Faizabad
    [0.6, 1.4, 2.0, 0.9],   # F-7
    [0.4, 0.6, 0.8, 0.4],   # G-10
    [0.3, 0.3, 0.4, 0.2]])  # Dhok Hassu (low-demand zone, tests fairness)

STEPS = 192          # 06:00 to 22:00 in 5-minute steps
NV = 30              # number of vehicles
REV, FUEL_KM = 4.5, 0.25   # reward units: Rs 450 per ride, Rs 25 per km (divided by 100)


def block(t):
    return min(3, t // 48)


class Env:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, rain=None):
        self.t = 0
        self.rain = int(self.rng.random() < 0.3) if rain is None else rain
        self.loc = self.rng.integers(0, NZ, NV)
        self.free_at = np.zeros(NV, int)
        self.queue = [[] for _ in range(NZ)]        # ages of waiting passengers
        self.last_served = np.zeros(NZ, int)
        self.stats = dict(rides=0, cancelled=0, requests=0, wait=[], km=0.0)
        self.zone_req = np.zeros(NZ)
        self.zone_srv = np.zeros(NZ)
        return self

    def hotspot(self):
        q = np.array([len(x) for x in self.queue])
        return int(q.argmax()) if q.max() > 0 else NZ

    def state(self, v):
        return (self.loc[v], block(self.t), self.rain, self.hotspot())

    def idle(self):
        return [v for v in range(NV) if self.free_at[v] <= self.t]

    def step(self, actions):
        """actions: dict {vehicle: target zone}. Returns (reward dict, done)."""
        rew = {v: 0.0 for v in range(NV)}
        t = self.t
        # 1) move vehicles (costs fuel, takes 1-2 steps)
        for v, a in actions.items():
            d = DIST[self.loc[v], a]
            if d > 0:
                self.stats['km'] += d
                rew[v] -= FUEL_KM * d
                self.free_at[v] = t + (1 if d < 7 else 2)
                self.loc[v] = a
        # 2) new passengers arrive
        lam = BASE[:, block(t)] * (1.4 if self.rain else 1.0)
        for z, n in enumerate(self.rng.poisson(lam)):
            self.queue[z] += [0] * n
            self.stats['requests'] += n
            self.zone_req[z] += n
        # 3) match idle vehicles to waiting passengers in the same zone
        glob_pen = 0.0
        for z in range(NZ):
            vs = [v for v in range(NV) if self.loc[v] == z and self.free_at[v] <= t]
            while vs and self.queue[z]:
                v = vs.pop(0)
                age = self.queue[z].pop(0)
                self.stats['rides'] += 1
                self.zone_srv[z] += 1
                self.stats['wait'].append(age * 5 + 2)
                bonus = 1.0 if t - self.last_served[z] >= 6 else 0.0   # fairness bonus
                self.last_served[z] = t
                rew[v] += REV + bonus
                dur = self.rng.integers(2, 5) + (1 if self.rain else 0)
                self.free_at[v] = t + dur
                self.loc[v] = self.rng.choice(NZ)      # trip ends in any zone
            # 4) passengers wait or cancel after 15 minutes
            new_q = []
            for age in self.queue[z]:
                if age + 1 >= 3:
                    self.stats['cancelled'] += 1
                    glob_pen += 2.0
                else:
                    new_q.append(age + 1)
                    glob_pen += 0.1
            self.queue[z] = new_q
        for v in rew:
            rew[v] -= glob_pen / NV                    # shared team penalty
        self.t += 1
        return rew, self.t >= STEPS


# ---------------- Baseline policies ----------------
def random_policy(env, v, rng):
    return int(rng.integers(NZ))

BUSIEST = int(BASE.sum(1).argmax())

def busiest_policy(env, v, rng):
    return BUSIEST                       # everyone goes to the historically busiest zone

def stay_policy(env, v, rng):
    return int(env.loc[v])


# ---------------- Q-Learning agent ----------------
class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, seed=0):
        self.Q = np.zeros((NZ, 4, 2, NZ + 1, NZ))   # zone, time, rain, hotspot, action
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)

    def act(self, s, eps):
        if self.rng.random() < eps:                  # explore
            return int(self.rng.integers(NZ))
        q = self.Q[s]                                # exploit
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def update(self, s, a, r, s2, done):
        target = r + (0 if done else self.gamma * self.Q[s2].max())
        self.Q[s + (a,)] += self.alpha * (target - self.Q[s + (a,)])


def run_episode(env, chooser, learner=None, eps=0.0):
    env.reset()
    total = 0.0
    while True:
        acts, states = {}, {}
        for v in env.idle():
            s = env.state(v)
            acts[v] = chooser(env, v, s, eps)
            states[v] = s
        rew, done = env.step(acts)
        total += sum(rew.values())
        if learner is not None:
            for v, s in states.items():
                learner.update(s, acts[v], rew[v], env.state(v), done)
        if done:
            break
    st = env.stats
    zr = env.zone_srv / np.maximum(env.zone_req, 1)
    return dict(reward=total, rides=st['rides'], cancelled=st['cancelled'],
                requests=st['requests'],
                wait=float(np.mean(st['wait'])) if st['wait'] else 0.0,
                km=st['km'], min_zone=float(zr.min()), zr=zr)


def train(episodes=1500, seed=0, alpha=0.1, gamma=0.9):
    agent = QAgent(alpha, gamma, seed)
    env = Env(seed)
    curve = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 * (0.995 ** ep))         # epsilon-greedy decay
        r = run_episode(env, lambda e, v, s, x: agent.act(s, x), agent, eps)
        curve.append(r['reward'])
    return agent, curve


def evaluate(chooser, n=40, seed=1000):
    env = Env(seed)
    res = [run_episode(env, chooser, None, 0.0) for _ in range(n)]
    keys = ['reward', 'rides', 'cancelled', 'requests', 'wait', 'km', 'min_zone']
    out = {k: float(np.mean([r[k] for r in res])) for k in keys}
    out['zr'] = np.mean([r['zr'] for r in res], axis=0)
    return out