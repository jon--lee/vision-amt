from options import AMTOptions
import sys, os, time, cv2, argparse



def color(frame): 
    color_frame = cv2.resize(frame.copy(), (250, 250))
    cv2.imwrite('get_jp.jpg',color_frame)
    color_frame= cv2.imread('get_jp.jpg')
    return color_frame


if __name__ == '__main__':
    print sys.argv[1], sys.argv[2], sys.argv[3]
    name = sys.argv[1]
    first = sys.argv[2]
    second = sys.argv[3]
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



    i = 0
    for frame, delta, state in zip(frames, deltas, states):
        filename = name + '_' + rollout_name + '_frame_' + str(i) + '.jpg'
        delta = delta[:delta.find('supervised')+len('supervised')] + second + delta[delta.find('_frame'):]
        state = state[:state.find('supervised')+len('supervised')] + second + delta[state.find('_frame'):]
        deltas_file.write(delta)
        states_file.write(state)
        frame = cv2.imread(frame)
        cv2.imwrite(rollout_path + filename, frame)
        cv2.imwrite(AMTOptions.colors_dir + filename, color(frame))
        i += 1