"""Example call:
python sir.py params/sir-light.yaml '{"outdir": "/tmp/foo"}'"""

from utils import *
import math, shutil
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

import numba
from numba import int32, int64
from numba.experimental import jitclass

from repast4py import core, space, schedule, logging, random
from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType

import spatial, vis

model = None # The simulation model is accessible from any point in the code

@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]

spec = [
    ('mo', int32[:]),
    ('no', int32[:]),
    ('xmin', int32),
    ('ymin', int32),
    ('ymax', int32),
    ('xmax', int32)
]

@jitclass(spec)
class GridNghFinder:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.mo = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def find(self, x, y):
        xs = self.mo + x
        ys = self.no + y

        xd = (xs >= self.xmin) & (xs <= self.xmax)
        xs = xs[xd]
        ys = ys[xd]

        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)

class Human(core.Agent):
    """Class representing the individuals.
    Each individual can be in one of three states: susceptible, infected, or recovered
    """
    TYPE = 0

    def __init__(self, a_id: int, rank: int,
                 sirstate: int=STATE.S, infcountdown: int=0, stepsize: int=1):
        super().__init__(id=a_id, type=Human.TYPE, rank=rank)
        self.sirstate = sirstate
        self.infcountdown = infcountdown # Countdown, in steps, to get recovered
        self.justinfected = False # Indicates if it has just got infected
        self.stepsize = stepsize

    def save(self) -> Tuple:
        """Save the state of the agent, so it can be transferred to another rank."""
        return (self.uid, self.sirstate, self.infcountdown,
                self.justinfected)

    def step_movement(self):
        """Movement step. We randomly pick an angle and move in that direction."""
        h = self.stepsize
        spacept = model.space.get_location(self)
        ang = np.random.rand() * (2 * np.pi)
        stepx, stepy = np.cos(ang) * h, np.sin(ang) * h
        model.move(self, spacept.x + stepx, spacept.y + stepy)
        space_pt = model.space.get_location(self)
        return (space_pt)

    def step_infection(self):
        """Infection step. If the agent is not infected or it has just got
        infected, there is nothing to do. If it is infected, it find the
        contacts nearby and infect them with a given probability.
        Also in this step, it decreases the countdown to get recovered."""
        if self.sirstate != STATE.I:
            return
        elif self.justinfected:
            self.justinfected = False
            return

        # Naive way of finding neighbours
        neighbours = []
        loc = model.space.get_location(self)
        for ag in model.context.agents():
            if ag.sirstate != STATE.S: # Can just infect Susceptibles
                continue
            loc2 = model.space.get_location(ag)
            dist = np.linalg.norm(loc.coordinates - loc2.coordinates)
            if dist < model.contactradius:
                neighbours.append(ag)

        # Infect with probability @model.probinf
        if len(neighbours) > 0:
            mask = np.random.rand(len(neighbours)) < model.probinf
            for ag in np.array(neighbours)[mask]:
                ag.sirstate = STATE.I
                ag.infcountdown = model.inftime
                ag.justinfected = True

        if self.infcountdown == 0:
            self.sirstate = STATE.R # Infected -> recovered

        self.infcountdown -= 1

agent_cache = {}


def restore_agent(agent_data: Tuple):
    """Re-create an agent. It reads the data from save()"""
    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank

    if uid in agent_cache:
        h = agent_cache[uid]
    else:
        h = Human(uid[0], uid[2])
        agent_cache[uid] = h

    h.sirstate = agent_data[1]
    h.infcountdown = agent_data[2]
    h.justinfected = agent_data[3]
    return h


@dataclass
class Counts:
    """Record the count of individuals by state."""
    susceptibles: int = 0
    infected: int = 0
    recovered: int = 0


class Model:
    """Simulation model"""
    def __init__(self, comm, params):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()
        self.probinf = params['probinf']
        self.inftime = params['inftime']
        worldarea = params['world.width'] * params['world.height']
        self.contactradius = 50
        self.stepsize = 2 # Agent step size

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        if params['posfile']:
            self.runner.schedule_repeating_event(1.1, 1, self.log_agents)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        box = space.BoundingBox(0, params['world.width'], 0,
                                params['world.height'], 0, 0)
        self.grid = space.SharedGrid('grid', bounds=box,
                                     borders=BorderType.Sticky,
                                     occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)
        self.space = space.SharedCSpace('space', bounds=box,
                                        borders=BorderType.Sticky,
                                        occupancy=OccupancyType.Multiple,
                                        buffer_size=2, comm=comm,
                                        tree_threshold=100)
        self.context.add_projection(self.space)
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        self.counts = Counts()
        logcountspath = pjoin(params['outdir'], params['countsfile'])
        cols = ['tick', 'agent_id', 'agent_type', 'agent_uid_rank',
                'posx', 'posy', 'sirstate']
        if params['posfile']:
            logpospath = pjoin(params['outdir'], params['posfile'])
            self.agent_logger = logging.TabularLogger(comm, logpospath, cols)
        loggers = logging.create_loggers(self.counts, op=MPI.SUM,
                                         rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm,
                                                logcountspath)
        world_size = comm.Get_size()

        local_bounds = self.space.get_local_bounds()
        i = 0
        for k0, sirstate in zip(['s0', 'i0', 'r0'], [STATE.S, STATE.I, STATE.R]):
            mm = params[k0]

            total_human_count = params[k0]
            m = int(total_human_count / world_size)
            if self.rank < total_human_count % world_size:
                m += 1

            if sirstate == STATE.I:
                countdowns = [self.inftime] * m # 'Full' countdown at the beginning
            else:
                countdowns = [0] * m

            for j in range(m):
                h = Human(i, self.rank, sirstate, countdowns[j], self.stepsize)
                self.context.add(h)
                x = random.default_rng.uniform(local_bounds.xmin,
                                               local_bounds.xmin + local_bounds.xextent)
                y = random.default_rng.uniform(local_bounds.ymin,
                                               local_bounds.ymin + local_bounds.yextent)
                self.move(h, x, y)
                i += 1

    def at_end(self):
        """Procedures at the end of the simulation."""
        self.data_set.close()

    def move(self, agent, x, y):
        """Update the space and the grid wrt to the agent's location."""
        self.space.move(agent, cpt(x, y))
        self.grid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))

    def step(self):
        """Simulation model step."""
        tick = self.runner.schedule.tick
        self.log_counts(tick)
        self.context.synchronize(restore_agent)

        for h in self.context.agents(Human.TYPE):
            pt = h.step_movement()
            h.step_infection()

    def log_agents(self):
        """Log the state and position of each agent in each step."""
        tick = self.runner.schedule.tick
        for agent in self.context.agents():
            pt = model.space.get_location(agent)
            self.agent_logger.log_row(tick, agent.id, agent.TYPE, agent.uid_rank,
                                      pt.x, pt.y, agent.sirstate)

        self.agent_logger.write()

    def run(self):
        """Simulation model execution"""
        self.runner.execute()

    def remove_agent(self, agent):
        """Remove an agent"""
        self.context.remove(agent)

    def log_counts(self, tick):
        """"""
        agents = self.context.agents()
        self.counts.susceptibles = self.counts.infected = self.counts.recovered = 0
        for ag in agents:
            if ag.sirstate == STATE.S:
                self.counts.susceptibles += 1
            elif ag.sirstate == STATE.I:
                self.counts.infected += 1
            elif ag.sirstate == STATE.R:
                self.counts.recovered += 1

        self.data_set.log(tick)

def run(params: Dict):
    """Instatiates the simulation model and assigns it to a global variable."""
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()

if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)

    np.random.seed(params['random.seed']) # Use the same seed for Numpy

    outdir = params['outdir']
    os.makedirs(outdir, exist_ok=True)

    shutil.copy(args.parameters_file, outdir)
    readmepath = create_readme(sys.argv, outdir)

    t0 = time.time()
    run(params)
    open(readmepath, 'a').write(f'Elapsed time: {time.time() - t0}')

    # pospath = pjoin(params['outdir'], params['posfile'])
    # vis.plot_positions(pospath, params['outdir'])
    # vis.plot_counts(pospath, params['outdir'])

    info('FINISHED')
