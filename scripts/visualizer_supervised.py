#visualizer

import cv2
import time
from options import Options, AMTOptions
import numpy as np
from Net.tensor import inputdata, net3,net4,net5,net6, net6_c
import matplotlib.pyplot as plt

def convert_state(state):
    # converts a state into a length six list, 
    # differentiates between four number and six number states
    split = state.split(' ')
    if len(split) == 5:
        new_state = [0.0] * 6
        new_state[0] = float(split[1])
        new_state[2] = float(split[2])
        return new_state
    elif len(split) == 7:
        # print "state", state, "state", string_state(state), "fin"
        return string_state(state)
    print "len(state): ", len(state), state

def convert_delta(delta):
    # converts a delta 
    delta = delta.split(' ')
    new_delta = [0.0] * 6
    new_delta[0] = float(delta[1])
    new_delta[2] = float(delta[2])
    # print delta
    return new_delta

def command_to_line(delta, state):
    # print convert_delta(delta), state
    state_next = np.array(state) + np.array(exagerrate_delta(delta))
    state_start = state
    # print "result", state_to_pixel(state_start)[0], state_to_pixel(state_next)[0]
    return state_to_pixel(state_start)[0], state_to_pixel(state_next)[0] 

def states_to_line(state_start, state_next):
    return state_to_pixel(state_start)[0], state_to_pixel(state_next)[0] 


def rescale_sup(deltas):
    # deltas[0] = deltas[0]*0.00754716981132
    # deltas[1] = deltas[1]*0.004
    deltas[0] = deltas[0]*0.02666666666667
    deltas[1] = deltas[1]*0.006

    # deltas[2] = 0.0
    # deltas[3] = 0.0
    return deltas

def exagerrate_delta(delta):
    return np.array(delta) * 2

def feed_to_net(sess, net, image):
    grey_frame = cv2.resize(image.copy(), (250, 250))
    grey_frame = np.reshape(grey_frame, (250, 250, 3))
    net_delta = net.output(sess, grey_frame,channels=3)
    rescaled_net = rescale_sup(net_delta.copy())
    delta = [0] * 6
    delta[0] = rescaled_net[0]
    delta[2] = rescaled_net[1]
    # print "net", delta, net_delta
    return delta

def state_to_pixel(state):
    # takes in a robot state and converts it into two pixel locations
    # The first is the location at the right edge of the screen
    # the second is the location at the center of the research 

    # ### Test angles
    #hardcoded dimensions
    # L = 420 #size in pixels of the viewscreen
    # base_ext = .385 # The extention
    # ang_offset = np.pi + .28#0.26760734641 # the offset of horizontal from the angle
    L = 420 #size in pixels of the viewscreen
    base_ext = .495 # The extention
    ang_offset = np.pi + .45#0.26760734641 # the offset of horizontal from the angle
    bx, by = 270,180 #metersToPixels(.3), metersToPixels(.2)
    ###

    grip_ang = -(float(state[0]) - ang_offset)
    grip_ext = (float(state[2]) + base_ext)
    # print grip_ang, grip_ext

    grip_post = np.array([metersToPixels(grip_ext*np.cos(grip_ang)),metersToPixels(grip_ext*np.sin(grip_ang))])
    grip_L = np.array([bx,bx*np.sin(grip_ang)]) #the start position of the gripper

    translation = np.array([bx + 420, by])
    rotation = np.array([[-1, 0], [0, -1]])
    grip_post = np.dot(rotation, grip_post - translation)
    grip_L = np.dot(rotation, grip_L - translation)
    grip_post = (int(grip_post[0]), int(grip_post[1]))
    grip_L = (int(grip_L[0]), int(grip_L[1]))

    return grip_post, grip_L

def pixelstoMeters(val):
    return 0.5461/420*val

def metersToPixels(val):
    return 420/0.5461*val


def draw_result(img, state, color = (0,0,255), thick = 2):
     
    # grip_ang = np.arctan2(d_vec[1]-base[0], d_vec[0]-base[1])
    # grip_post = (int(grip_pos[0]), int(grip_pos[1]))
    state_start, state_end = state

    # gc_est = [gc_pos[0], gc_pos[0] * np.tan(gc_ang) + base[1]]
    cv2.line(img, state_start, state_end, color, thick)
    # cv2.circle(img, state_start, thick, (color[0] + 255, color[1], color[2]))

def display_result(imgs, names=['figure']): 
    while 1:
        i = 0
        for img in imgs:
            cv2.imshow(names[i],img) # draws a line along the length of the gripper
            i += 1
        a = cv2.waitKey(30)
        if a == 27:
            # cv2.destroyAllWindows()
            break
        time.sleep(.005)



def string_state(state):
    split = state.split(' ')
    # print state
    labels = np.array( [ float(x) for x in split[1:] ] )
    # print labels
    return labels

def bound_state(state):
    # print state
    state[2] = min(state[2], AMTOptions.EXTENSION_UPPER_BOUND)
    state[0] = min(state[0], AMTOptions.ROTATE_UPPER_BOUND)
    state[0] = max(state[0], AMTOptions.ROTATE_LOWER_BOUND)
    return state

def normalize_supervisor(delta):
    # assumes a six state delta
    delta = np.array(delta)
    delta[0] = delta[0] / .0266666666666666667
    delta[1] = delta[1] / .006
    return delta

def normalize_rollout(delta):
    delta = np.array(delta)
    delta[0] = delta[0] / .11
    delta[1] = delta[1] / .1
    return delta

def trajectory_distribution(options, rollout_lst, f_rng):
    initial_image = np.zeros((420, 420, 3))
    for rollout_num in rollout_lst:
        state_path = options.rollouts_dir + '/rollout'+str(rollout_num) +'/states.txt'
        states = open(state_path, 'r')
        first = True
        last = None
        i = 0
        for stv in states:
            if first:
                last = bound_state(string_state(stv))
                print last
                first = False
                continue

            state = bound_state(string_state(stv))
            # print type(state), last, state
            sup_change = states_to_line(last, state)


            state_pts = state_to_pixel(state)
            if i > f_rng[0] and (i < f_rng[1] or f_rng[1] == -1):
                draw_result(initial_image, sup_change, color = (0,255,0), thick = 1 )
                # draw_result(img, state_pts, color = (255,0,0))
                # draw_result(img, net_change, color = (0,255,0))
                # draw_result(img, sup_change, color = (0,0,255))
            # display_result([img])
            if i > f_rng[1] and f_rng[1] != -1:
                break
            i += 1
            last = state

    display_result([initial_image], ['trajectory distribution'])
    cv2.imwrite("test_dist.jpg", initial_image)

def compare_supervisor_net_rol(options, net, sess, rollout_lst, f_rng = (0,-1)):
    for rollout_num in rollout_lst:
        image_path = options.supervised_dir + 'supervised'+str(rollout_num) +'/supervised'+str(rollout_num) +'_frame_'
        state_path = options.supervised_dir + '/supervised'+str(rollout_num) +'/states.txt'
        states = open(state_path, 'r')
        deltas_path = options.supervised_dir + 'supervised'+str(rollout_num) +'/deltas.txt'
        deltas = open(deltas_path, 'r')


        sup_policy = cv2.imread(image_path + str(0) + '.jpg')
        net_policy = cv2.imread(image_path + str(0) + '.jpg')
        combined_policy = cv2.imread(image_path + str(0) + '.jpg')

        i = 0
        for stv, delta in zip(states, deltas):
            state = stv
            img = cv2.imread(image_path + str(i) + '.jpg')
            state = bound_state(string_state(stv))
            delta = convert_delta(delta)
            sup_change = command_to_line(delta, state)

            net_delta = feed_to_net(sess, net, img)
            net_change = command_to_line(net_delta, state)

            # print normalize_supervisor(delta)
            # print normalize_supervisor(net_delta)
            supervisor_value = [0] * 2
            supervisor_value[0] = delta[0]
            supervisor_value[1] = delta[2]
            net_value = [0] * 2
            net_value[0] = net_delta[0]
            net_value[1] = net_delta[2]

            state_pts = state_to_pixel(state)
            print i, " ", state_pts[0], " ",state
            if i > f_rng[0] and (i < f_rng[1] or f_rng[1] == -1):
                draw_result(net_policy, net_change, color = (0,255,0))
                draw_result(sup_policy, sup_change, color = (0,0,255))
                draw_result(combined_policy, net_change, color = (0,255,0))
                draw_result(combined_policy, sup_change, color = (0,0,255))
                # draw_result(img, state_pts, color = (255,0,0))
                # draw_result(img, net_change, color = (0,255,0))
                # draw_result(img, sup_change, color = (0,0,255))
            # display_result([img])
            if i > f_rng[1] and f_rng[1] != -1:
                break
            i += 1
        cv2.imwrite("worst_test_ang" + str(rollout_num) + ".jpg", combined_policy)
        states.close()
        deltas.close()
        print "rollout number: ", rollout_num
        # print "error rate: ", rollout_num
        # display_result([net_policy,sup_policy, combined_policy], ['net', 'supervisor', 'both'])
        # plt.plot(distances)
        # plt.show(block=False)
        # plt.close()
        # print rollout_num - r_rng[0]



def compare_supervisor_net(options, net, sess, r_rng, f_rng = (0,-1)):
    frame_differences = [np.array((0.0, 0.0)) for i in range(100)]
    frame_totals = [0.0 for i in range(100)]
    ang_rates = []
    fore_rates = []
    rates = []
    costs = []
    for rollout_num in range(r_rng[0],r_rng[1]):
        image_path = options.supervised_dir + 'supervised'+str(rollout_num) +'/supervised'+str(rollout_num) +'_frame_'
        state_path = options.supervised_dir + '/supervised'+str(rollout_num) +'/states.txt'
        states = open(state_path, 'r')
        deltas_path = options.supervised_dir + 'supervised'+str(rollout_num) +'/deltas.txt'
        deltas = open(deltas_path, 'r')


        sup_policy = cv2.imread(image_path + str(0) + '.jpg')
        net_policy = cv2.imread(image_path + str(0) + '.jpg')
        combined_policy = cv2.imread(image_path + str(0) + '.jpg')

        error_costs = 0.0
        error_ang = 0.0
        error_fore = 0.0
        i = 0
        window = []
        distances = []
        for stv, delta in zip(states, deltas):
            state = stv
            img = cv2.imread(image_path + str(i) + '.jpg')
            state = bound_state(string_state(stv))
            delta = convert_delta(delta)
            # print delta
            sup_change = command_to_line(delta, state)

            net_delta = feed_to_net(sess, net, img)
            net_change = command_to_line(net_delta, state)

            # print normalize_supervisor(net_delta)
            supervisor_value = [0] * 2
            supervisor_value[0] = delta[0]
            supervisor_value[1] = delta[2]
            supervisor_norm = normalize_supervisor(supervisor_value)
            net_value = [0] * 2
            net_value[0] = net_delta[0]
            net_value[1] = net_delta[2]
            net_norm = normalize_supervisor(net_value)
            net_value = np.array(net_value)
            supervisor_value = np.array(supervisor_value)
            error_costs += np.linalg.norm(net_norm - supervisor_norm)
            error_ang += np.abs(supervisor_value[0] - net_value[0])
            error_fore += np.abs(supervisor_value[1] - net_value[1])
            # window.append(np.linalg.norm(net_value - supervisor_value))
            # if len(window) > 1:
            #     window.pop(0)
            #     distances.append(np.sum(window))
            # if np.sum(window) > 0:
            #     num_errs += 1
            # print window
            # print 'Magnitude difference: ', np.linalg.norm(net_value - supervisor_value)

            frame_differences[i] += np.array((np.abs(supervisor_value[0] - net_value[0]), np.abs(supervisor_value[1]-net_value[1])))
            frame_totals[i] += np.linalg.norm(net_norm-supervisor_norm)
            state_pts = state_to_pixel(state)
            if i > f_rng[0] and (i < f_rng[1] or f_rng[1] == -1):
                draw_result(net_policy, net_change, color = (0,255,0))
                draw_result(sup_policy, sup_change, color = (0,0,255))
                draw_result(combined_policy, net_change, color = (0,255,0))
                draw_result(combined_policy, sup_change, color = (0,0,255))
            #     draw_result(img, state_pts, color = (255,0,0))
            #     draw_result(img, net_change, color = (0,255,0))
            #     draw_result(img, sup_change, color = (0,0,255))
            # display_result([img])
            if i > f_rng[1] and f_rng[1] != -1:
                break
            i += 1
        # display_result([net_policy,sup_policy, combined_policy], ['net', 'supervisor', 'both'])
        # plt.plot(distances)
        # plt.show(block=False)
        # plt.close()
        # print rollout_num - r_rng[0]
        rates.append(error_costs/float(i))
        ang_rates.append(error_ang/float(i))
        fore_rates.append(error_fore/float(i))
        print "rollout number: ", rollout_num, " rate: ", error_costs/float(i), " fore: ", error_ang/float(i), " ang: ", error_fore/float(i)
    rate_indices = np.argsort(ang_rates)
    fore_indices = np.argsort(fore_rates)
    indices = np.argsort(rates)
    rates.sort()
    ang_rates.sort()
    fore_rates.sort()
    print "total trajectory costs"
    print indices + r_rng[0]
    print "angle trajectory costs"
    print rate_indices + r_rng[0]
    print "forward trajectory costs"
    print fore_indices + r_rng[0]
    plt.plot(ang_rates)
    plt.plot(fore_rates)
    # plt.plot(rates)
    plt.show()
    plt.close()
    plt.plot(rates)
    plt.show()
    for i in range(100):
        frame_differences[i] = -frame_differences[i]/(r_rng[0]-r_rng[1])
    plt.plot(frame_differences)
    plt.show()
    for i in range(100):
        frame_totals[i] = -frame_totals[i]/(r_rng[0]-r_rng[1])
    plt.plot(frame_totals)
    plt.show()

def compare_policies(path_image, r_rng, f_rng, paths_first_states, paths_first_deltas, paths_second_states, paths_second_deltas, options):
    '''
    Displays the policy on the first state of the first set of data
    '''

    i = 0
    for pf_states, pf_deltas, ps_states, ps_deltas in zip(paths_first_states, paths_first_deltas, paths_second_states, paths_second_deltas):
        image_path = assign_path(path_image, r_rng[i], options)
        img = cv2.imread(image_path)
        f_states = open(pf_states, 'r')
        f_deltas = open(pf_deltas, 'r')
        s_states = open(ps_states, 'r')
        s_deltas = open(ps_deltas, 'r')
        j = 0
        for f_state, f_delta, s_state, s_delta in zip(f_states, f_deltas, s_states, s_deltas):
            print f_state, f_delta, s_state, s_delta
            f_state = bound_state(convert_state(f_state))
            f_delta = convert_delta(f_delta)
            s_state = bound_state(convert_state(s_state))
            s_delta = convert_delta(s_delta)

            f_change = command_to_line(f_delta, f_state)
            s_change = command_to_line(s_delta, s_state)
            if (j < f_rng[1] and j > f_rng[0]) or f_rng[1] == -1:
                draw_result(img, f_change, color = (0,0,255))
                draw_result(img, s_change, color = (0,255,0))
            j += 1
        f_states.close()
        f_deltas.close()
        s_states.close()
        s_deltas.close()
        print i - r_rng[0], pf_states, ps_states
        display_result([img], ['both'])
        cv2.imwrite("net_fail" + str(r_rng[i]) + ".jpg", img)
        i += 1


def get_rollout_paths(r_rng, options):
    s_paths, d_paths = [], []
    for rollout_num in r_rng:
        state_path = options.rollouts_dir + 'rollout'+str(rollout_num) +'/states.txt'
        deltas_path = options.rollouts_dir + 'rollout'+str(rollout_num) +'/net_deltas.txt'
        s_paths.append(state_path)
        d_paths.append(deltas_path)
    return s_paths, d_paths

def get_supervised_paths(r_rng, options):
    s_paths, d_paths = [], []
    for rollout_num in r_rng:
        state_path = options.supervised_dir + 'supervised'+str(rollout_num) +'/states.txt'
        deltas_path = options.supervised_dir + 'supervised'+str(rollout_num) +'/deltas.txt'
        s_paths.append(state_path)
        d_paths.append(deltas_path)
    return s_paths, d_paths

def assign_path(image_path, rollout_num, options):
    if image_path == options.supervised_dir:
        return options.supervised_dir + 'supervised'+str(rollout_num) +'/supervised'+str(rollout_num) + '_frame_' + str(0) + '.jpg'
    elif image_path == options.rollouts_dir:
        return options.rollouts_dir + 'rollout'+str(rollout_num) +'/rollout'+str(rollout_num) + '_frame_' + str(0) + '.jpg'

    return None


options = AMTOptions()
net = net6.NetSix()
options.tf_net = net
# options.tf_net = net6.NetSix()
options.tf_net_path = '/media/1tb/Izzy/nets/net6_07-01-2016_16h24m28s.ckpt'
sess = net.load(var_path=options.tf_net_path)
# rollout_num = 63
best_ang = [344, 357, 341, 322, 374, 319, 315, 349, 342, 373]
worst_ang = [323, 356, 331, 317, 351, 336, 367, 345, 324, 366]

best_fore = [317, 335, 329, 343, 337, 348, 322, 370, 323, 351]
worst_fore = [325, 369, 332, 326, 339, 346, 324, 364, 359, 353]

worst_test = [441, 439, 448, 440, 453]

trajectories = [250, 251] + list(range(253, 280)) + [252]

rollout_traj = [267, 265, 274, 266, 279]
supervised_traj = worst_test

# compare_supervisor_net_rol(options, net, sess, worst_test, (0,-1))
# compare_supervisor_net(options, net, sess, (425, 455), (0,-1))
# compare_supervisor_net(options, net, sess, (315, 415), (0,-1))

# trajectory_distribution(options, list(range(340,369)),(0,-1))

sup_states, sup_deltas = get_supervised_paths(supervised_traj, options)
net_states, net_deltas = get_rollout_paths(rollout_traj, options)
compare_policies(options.supervised_dir, supervised_traj, (0,-1), sup_states, sup_deltas, net_states, net_deltas, options)

# sup_states, sup_deltas = get_supervised_paths((123,143), options)
# net_states, net_deltas = get_rollout_paths((1820,1840), options)
# compare_policies(options.supervised_dir, (123,143), (0,-1), sup_states, sup_deltas, net_states, net_deltas, options)

# sess.close()
