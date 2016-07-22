# clean data

import numpy as np
import os, sys
from options import AMTOptions

def array_state(state):
    state = state.split(' ')
    valued = []
    for val in state[1:]:
        valued.append(float(val[:val.find('\n')]))
    return valued

def array_delta(delta):
    delta = delta.split(' ')
    valued = []
    for val in delta[1:]:
        valued.append(float(val[:val.find('\n')]))
    delta = [0] * 6
    delta[0] = valued[0]
    delta[2] = valued[1]
    return delta

def clean_repeats(states, deltas):
    last_states = []
    new_deltas = []

    distance = 99999
    i = 0
    for state,delta in zip(states, deltas):
        state = np.array(array_state(state))
        state[0] = min(AMTOptions.ROTATE_UPPER_BOUND, state[0])
        state[0] = max(AMTOptions.ROTATE_LOWER_BOUND, state[0])

        state[2] = min(AMTOptions.EXTENSION_UPPER_BOUND, state[2])
        state[2] = max(AMTOptions.EXTENSION_LOWER_BOUND, state[2])
        if len(last_states) > 3:
            distance = np.sum(abs(np.array([state - last for last in last_states])))
            last_states.pop(0)
        last_states.append(state)
        i += 1
        if distance > .01:
            new_deltas.append(delta)
    return new_deltas

def test_files():
    supervised_dir = AMTOptions.supervised_dir
    rollouts = [x[0] for x in os.walk(supervised_dir)]
    val = [rollout_dir for rollout_dir in rollouts if rollout_dir != supervised_dir]
    txt_files = [rollout_dir + '/deltas.txt' for rollout_dir in rollouts if rollout_dir != supervised_dir]
    rol_num = lambda x: int(x[x.find('s/supervised') + len('s/supervised'):x.find('/deltas.txt')])
    state_files = [rollout_dir + '/states.txt' for rollout_dir in rollouts if rollout_dir != supervised_dir]
    s_rol_num = lambda x: int(x[x.find('s/supervised') + len('s/supervised'):x.find('/states.txt')])
    txt_files.sort(key = rol_num)
    state_files.sort(key = s_rol_num)
    txt_files = txt_files[62:63]
    state_files = state_files[62:63]
    # print [rol_num(x) for x in txt_files]
    states = []
    deltas = []
    for filename in txt_files:
        readfile = open(filename, 'r')
        for line in readfile:
            deltas.append(line)
        readfile.close()
    for filename in state_files:
        readfile = open(filename, 'r')
        for line in readfile:
            states.append(line)
        readfile.close()

    new_deltas = clean_repeats(states, deltas)

    # print deltas
