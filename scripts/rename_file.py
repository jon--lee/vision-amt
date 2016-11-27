from options import AMTOptions
import sys, os, time, cv2, argparse
import numpy as np



def color(frame): 
    color_frame = cv2.resize(frame.copy(), (250, 250))
    cv2.imwrite('get_jp.jpg',color_frame)
    color_frame= cv2.imread('get_jp.jpg')
    return color_frame

def rename_supervised(name, first, second, test):
    if name:
        start_path = AMTOptions.supervised_dir + name + "_rollouts/"
    else:
        start_path = AMTOptions.supervised_dir
    try:
        os.makedirs(start_path)
    except OSError as e:
        pass

    target_path = start_path + 'supervised' + first + '/'

    rollout_name = 'supervised' + second
    rollout_path =  start_path + rollout_name +'/'
    try:
        os.makedirs(rollout_path)
    except OSError as e:
        pass
    print "Saving rollout to " + rollout_path + "..."
    deltas_file = open(rollout_path + 'deltas.txt', 'w')
    states_file = open(rollout_path + 'states.txt', 'w')

    deltas = open(target_path + 'deltas.txt', 'r')
    states = open(target_path + 'states.txt', 'r')
    frames = [target_path + fle for fle in list(os.walk(target_path))[0][2] if fle.find('.jpg') != -1]
    print frames

    template = np.load(target_path + 'template.npy')

    np.save(rollout_path + 'template', template)



    i = 0
    for frame, delta, state in zip(frames, deltas, states):
        filename = name + '_' + rollout_name + '_frame_' + str(i) + '.jpg'
        delta = delta[:delta.find('supervised')+len('supervised')] + second + delta[delta.find('_frame'):]
        state = state[:state.find('supervised')+len('supervised')] + second + state[state.find('_frame'):]
        deltas_file.write(delta)
        states_file.write(state)
        frame = cv2.imread(frame)
        cv2.imwrite(rollout_path + filename, frame)
        cv2.imwrite(AMTOptions.colors_dir + filename, color(frame))
        i += 1

def adapt_test(test):
    if len(test) != 0:
        test = test+'_test/'

def rename_rollout(name, first, second, test):
    testft = adapt_test(test)
    if name:
        start_path = AMTOptions.rollout_dir + name + "_rollouts/"  + testft
    else:
        start_path = AMTOptions.rollout_dir
    try:
        os.makedirs(start_path)
    except OSError as e:
        pass

    target_path = start_path + "rollout" + first + '/'

    rollout_name = 'rollout' + second
    rollout_path =  start_path + rollout_name +'/'
    try:
        os.makedirs(rollout_path)
    except OSError as e:
        pass
    print "Saving rollout to " + rollout_path + "..."
    deltas_file = open(rollout_path + 'deltas.txt', 'w')
    states_file = open(rollout_path + 'states.txt', 'w')

    deltas = open(target_path + 'deltas.txt', 'r')
    states = open(target_path + 'states.txt', 'r')
    frames = [target_path + fle for fle in list(os.walk(target_path))[0][2] if fle.find('.jpg') != -1]
    print frames



    i = 0
    for frame, delta, state in zip(frames, deltas, states):
        filename = name + '_' + rollout_name + '_frame_' + str(i) + '.jpg'
        delta = delta[:delta.find('supervised')+len('supervised')] + second + delta[delta.find('_frame'):]
        state = state[:state.find('supervised')+len('supervised')] + second + state[state.find('_frame'):]
        deltas_file.write(delta)
        states_file.write(state)
        frame = cv2.imread(frame)
        cv2.imwrite(rollout_path + filename, frame)
        cv2.imwrite(AMTOptions.colors_dir + filename, color(frame))
        i += 1

def rename_test(name, test_source, test_destination):
    testsc = adapt_test(test_source)
    testdt = adapt_test(test_destination)
    for rol_num in range(30):
        if name:
            src_path = AMTOptions.rollout_dir + name + "_rollouts/"  + testsc
        else:
            src_path = AMTOptions.rollout_dir
        try:
            os.makedirs(start_path)
        except OSError as e:
            pass
        if name:
            dst_path = AMTOptions.rollout_dir + name + "_rollouts/"  + testdt
        else:
            dst_path = AMTOptions.rollout_dir
        try:
            os.makedirs(start_path)
        except OSError as e:
            pass

        target_path = src_path + "rollout" + first + '/'

        rollout_name = 'rollout' + second
        rollout_path =  dst_path + rollout_name +'/'
        try:
            os.makedirs(rollout_path)
        except OSError as e:
            pass
        print "Saving rollout to " + rollout_path + "..."
        deltas_file = open(rollout_path + 'deltas.txt', 'w')
        states_file = open(rollout_path + 'states.txt', 'w')

        deltas = open(target_path + 'deltas.txt', 'r')
        states = open(target_path + 'states.txt', 'r')
        frames = [target_path + fle for fle in list(os.walk(target_path))[0][2] if fle.find('.jpg') != -1]
        print frames



        i = 0
        for frame, delta, state in zip(frames, deltas, states):
            filename = name + '_' + rollout_name + '_frame_' + str(i) + '.jpg'
            delta = delta[:delta.find('rollout')+len('rollout')] + second + delta[delta.find('_frame'):]
            state = state[:state.find('rollout')+len('supervised')] + second + state[state.find('_frame'):]
            deltas_file.write(delta)
            states_file.write(state)
            frame = cv2.imread(frame)
            cv2.imwrite(rollout_path + filename, frame)
            cv2.imwrite(AMTOptions.colors_dir + filename, color(frame))
            i += 1

if __name__ == '__main__':
    print sys.argv[1], sys.argv[2], sys.argv[3]
    name = sys.argv[1]
    target = sys.argv[2]
    source = sys.argv[3]
    try:
        test = sys.argv[4]
    except IndexError as e:
        test = ''
    rename_supervised(name, source, target, test)
