import numpy as np
import os, sys
from options import AMTOptions
import scripts.clean_supervisor
from random import shuffle



def load_rollouts(clean, rand, r_rng, f_rng, outfile, name = None):
    supervised_dir = AMTOptions.supervised_dir
    if name is not None:
        supervised_dir += name + "_rollouts/"
    rollouts = [x[0] for x in os.walk(supervised_dir)]
    val = [rollout_dir for rollout_dir in rollouts if rollout_dir != supervised_dir]
    supervisor_dirs = [rollout_dir for rollout_dir in rollouts if rollout_dir != supervised_dir]
    rol_num = lambda x: int(x[x.find('s/supervised') + len('s/supervised'):])
    supervisor_dirs.sort(key = rol_num)
    if rand:
        print "randomizing"
        shuffle(supervisor_dirs)
    if len(supervisor_dirs) < np.max(r_rng):
        return True
    supervisor_dirs = supervisor_dirs[r_rng[0]:r_rng[1]]

    for dirname in supervisor_dirs:
        deltas = []
        deltas_file = open(dirname + '/deltas.txt', 'r')
        if clean:
            states = []
            states_file = open(dirname + '/states.txt', 'r')

        i = 0
        if clean:
            for delta, state in zip(deltas_file, states_file):
                if (i >= f_rng[0] and i <= f_rng[1]) or f_rng[1] == -1:
                    deltas.append(delta)
                    states.append(state)
                i += 1
        else:
            for delta in deltas_file:
                if (i >= f_rng[0] and i <= f_rng[1]) or f_rng[1] == -1:
                    deltas.append(delta)
                i += 1
        # print deltas, states
        deltas_file.close()
        if clean:
            states_file.close()

        if clean:
            new_deltas = scripts.clean_supervisor.clean_repeats(states, deltas)
        else:
            new_deltas = deltas
        for delta in new_deltas:
            outfile.write(delta)
    return False


if __name__ == "__main__":
    """Usage: No command line args uses all rollouts
2 command line args specifies to use the first argv[1] rollouts, and the first argv[2] frames
3 command line args specifies to use the range argv[1] - argv[2]-1 rollouts (inclusive), and the first argv[3] frames
4 command line args specifies to use the range argv[1] - argv[2]-1 rollouts (inclusive), and the range argv[3] - argv[4] frames
For all rollouts/all frames, specify -1 is valid
    """
    outfile = open('supervised_deltas.txt', 'w+')
    clean = True
    rand = False
    if len(sys.argv) == 3:
        num_rol = int(sys.argv[1])
        num_frame = int(sys.argv[2])

        load_rollouts(clean, rand, (0, num_rol), (0, num_frame), outfile)

    elif len(sys.argv) == 4:
        start_rol = int(sys.argv[1])
        end_rol = int(sys.argv[2])
        num_frame = int(sys.argv[3])

        load_rollouts(clean, rand, (start_rol, end_rol), (0, num_frame), outfile)

    elif len(sys.argv) == 4:
        start_rol = int(sys.argv[1])
        end_rol = int(sys.argv[2])
        start_frame = int(sys.argv[2])
        end_frame = int(sys.argv[2])

        load_rollouts(clean, rand, (start_rol, end_rol), (start_frame, end_frame), outfile)

    else:
        # pass
        load_rollouts(clean, rand, (0, -1), (0, -1), outfile)

    outfile.close()
