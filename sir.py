from utils import *
import math
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

import vis

model = None


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
    """The Human Agent

    Args:
        a_id: a integer that uniquely identifies this Human on its starting rank
        rank: the starting MPI rank of this Human.
    """

    TYPE = 0

    def __init__(self, a_id: int, rank: int,
                 sirstate: int, infcountdown: int):
        super().__init__(id=a_id, type=Human.TYPE, rank=rank)
        self.sirstate = sirstate
        self.infcountdown = infcountdown
        self.justinfected = False

    def save(self) -> Tuple:
        return (self.uid, self.sirstate, self.infcountdown,
                self.justinfected)

    def step(self):
        # MOVE
        h = model.agstepsize
        spacept = model.space.get_location(self)
        ang = np.random.rand() * (2 * np.pi)
        stepx, stepy = np.cos(ang) * h, np.sin(ang) * h
        model.move(self, spacept.x + stepx, spacept.y + stepy)

        if self.sirstate == STATE.I and (not self.justinfected):
            # INFECT

            #NAIVE WAY OF FINDING NEIGHBOURS
            neighbours = []
            loc = model.space.get_location(self)
            for ag in model.context.agents():
                if ag.sirstate != STATE.S: # Can just infect Susceptibles
                    continue
                loc2 = model.space.get_location(ag)
                dist = np.linalg.norm(loc.coordinates - loc2.coordinates)
                if dist < model.contactradius:
                    neighbours.append(ag)

            # INFECTING
            if len(neighbours) > 0:
                mask = np.random.rand(len(neighbours)) < model.probinf
                for ag in np.array(neighbours)[mask]:
                    ag.sirstate = STATE.I
                    ag.infcountdown = model.inftime
                    ag.justinfected = True

            # RECOVER
            if self.infcountdown == 0:
                self.sirstate = STATE.R
            self.infcountdown -= 1
        else:
            self.justinfected = False

        space_pt = model.space.get_location(self)
        return (space_pt)

agent_cache = {}


def restore_agent(agent_data: Tuple):
    """Creates an agent from the specified agent_data.

    This is used to re-create agents when they have moved from one MPI rank to another.
    The tuple returned by the agent's save() method is moved between ranks, and restore_agent
    is called for each tuple in order to create the agent on that rank. Here we also use
    a cache to cache any agents already created on this rank, and only update their state
    rather than creating from scratch.

    Args:
        agent_data: the data to create the agent from. This is the tuple returned from the agent's save() method
                    where the first element is the agent id tuple, and any remaining arguments encapsulate
                    agent state.
    """
    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank

    if uid in agent_cache:
        h = agent_cache[uid]
    else:
        h = Human(uid[0], uid[2])
        agent_cache[uid] = h

    # restore the agent state from the agent_data tuple
    h.sirstate = agent_data[1]
    h.infcountdown = agent_data[2]
    h.justinfected = agent_data[3]
    return h


@dataclass
class Counts:
    """Dataclass used by repast4py aggregate logging to record
    the number of Humans and Zombies after each tick.
    """
    humans: int = 0


class Model:
    def __init__(self, comm, params):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()
        self.probinf = params['probinf']
        # self.probrec = params['probrec']
        self.inftime = params['inftime']
        worldarea = params['world.width'] * params['world.height']
        # self.contactradius = .001 * worldarea
        self.contactradius = 50
        self.agstepsize = .0001 * worldarea

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_repeating_event(1.1, 1, self.log_agents)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.grid = space.SharedGrid('grid', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)
        self.space = space.SharedCSpace('space', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)
        self.context.add_projection(self.space)
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        self.counts = Counts()
        logstatespath = pjoin(params['outdir'], params['statesfile'])
        logcountspath = pjoin(params['outdir'], params['countsfile'])
        self.agent_logger = logging.TabularLogger(comm,
                                                  logstatespath,
                                                  ['tick', 'agent_id', 'agent_type',
                                                   'agent_uid_rank',
                                                   'posx', 'posy', 'sirstate'])
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, logcountspath)

        world_size = comm.Get_size()

        total_human_count = params['s0'] + params['i0'] + params['r0']
        pp_human_count = int(total_human_count / world_size)
        if self.rank < total_human_count % world_size:
            pp_human_count += 1

        local_bounds = self.space.get_local_bounds()
        i = 0
        for k0, sirstate in zip(['s0', 'i0', 'r0'], [STATE.S, STATE.I, STATE.R]):
            m = params[k0]
            if sirstate == STATE.I:
                # countdowns = np.random.randint(1, self.inftime, size=m)
                countdowns = [self.inftime] * m
            else:
                countdowns = [0] * m

            for j in range(m):
                h = Human(i, self.rank, sirstate, countdowns[j])
                self.context.add(h)
                x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
                y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
                self.move(h, x, y)
                i += 1

    def at_end(self):
        self.data_set.close()

    def move(self, agent, x, y):
        self.space.move(agent, cpt(x, y))
        self.grid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))

    def step(self):
        tick = self.runner.schedule.tick
        self.log_counts(tick)
        self.context.synchronize(restore_agent)

        dead_humans = [] # TODO: change the order: infected first
        for h in self.context.agents(Human.TYPE):
            pt = h.step()

    def log_agents(self):
        tick = self.runner.schedule.tick
        for agent in self.context.agents():
            pt = model.space.get_location(agent)
            self.agent_logger.log_row(tick, agent.id,
                                      agent.TYPE,
                                      agent.uid_rank,
                                      pt.x, pt.y,
                                      agent.sirstate)

        self.agent_logger.write()

    def run(self):
        self.runner.execute()

    def remove_agent(self, agent):
        self.context.remove(agent)

    def log_counts(self, tick):
        # Get the current number of zombies and humans and log
        num_agents = self.context.size([Human.TYPE])
        self.counts.humans = num_agents[Human.TYPE]
        self.data_set.log(tick)

        # Do the cross-rank reduction manually and print the result
        if tick % 10 == 0:
            human_count = np.zeros(1, dtype='int64')
            self.comm.Reduce(np.array([self.counts.humans], dtype='int64'), human_count, op=MPI.SUM, root=0)


def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()

if __name__ == "__main__":
    parser = create_args_parser() #TODO: uncomment this later
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    # params = init_params('./params.yaml', args.parameters)
    # params = init_params('./params.yaml', '')

    np.random.seed(params['random.seed']) # Use the same seed in np.random
    outdir = params['outdir']
    os.makedirs(outdir, exist_ok=True)
    run(params)
    pospath = pjoin(params['outdir'], params['statesfile'])
    vis.plot_positions(pospath, params['outdir'])
    vis.plot_counts(pospath, params['outdir'])
    print('FINISHED')
