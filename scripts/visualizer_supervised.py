#visualizer

import cv2, sys
sys.path.append('/home/annal/Izzy/vision_amt/')
import time
from options import AMTOptions
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
    state_true = np.array(state) + np.array(delta)
    state_start = state
    # print "result", state_to_pixel(state_start)[0], state_to_pixel(state_next)[0]
    return (state_to_pixel(state_start)[0], state_to_pixel(state_next)[0]), state_to_pixel(state_true)[0]

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
    return np.array(delta) * 4

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


def draw_result(img, state, color = (0,0,255), thick = 5):
     
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

def get_names(name):
    if len(name) != 0:
        folname = '/' + name + '_rollouts/'
        imname = '/' + name + '_'
    return folname, imname

def trajectory_distribution(options, rollout_lst, f_rng, name = '', test = ''):
    '''
    Draws all of the trajectories from a set onto a blank black background
    '''
    initial_image = np.zeros((420, 420, 3))
    folname, imname = get_names(name)
    folname, imname = get_test(folname, imname, test)
    for rollout_num in rollout_lst:
        state_path = options.rollouts_dir + folname + '/rollout'+str(rollout_num) +'/states.txt'
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

def get_dep_test(folname, imname, test):
    if len(test) != 0:
        folname = folname + test + '/'
        imname = imname
    print test, folname
    return folname, imname

def path_defining(folname, rollout_num, data_form):
    start_path = options.supervised_dir + folname + '/' + data_form + str(rollout_num) + '/'
    full_path = start_path + imname + 'supervised' + str(rollout_num) + '_frame_0.jpg'

def trajectory_supervised(options, rollout_lst, f_rng, names = [''], sup=False):
    '''
    Draws the trajectory of deltas on initial states from sources of several names
    '''
    init_imgs = {}
    nums = 0
    tot = len(names)
    for name in names:
        nums+=1
        folname, imname = get_names(name)
        start_path = options.supervised_dir + folname + '/supervised'
        for rollout_num in rollout_lst:
            if rollout_num in init_imgs:
                initial_image = init_imgs[rollout_num]
            else:
                full_path = start_path + str(rollout_num) + '/' + imname + 'supervised' + str(rollout_num) + '_frame_0.jpg'
                # initial_image = cv2.imread('doubleplotbackground.jpg')
                initial_image = cv2.imread(full_path)
                initial_image_white = np.zeros((420, 420, 3))
                init_imgs[rollout_num] = initial_image
                print full_path
            state_path = start_path + str(rollout_num) + '/states.txt'
            states = open(state_path, 'r')
            first = True
            last = None
            i = 0
            for stv in states:
                if first:
                    last = bound_state(string_state(stv))
                    # print last
                    first = False
                    continue

                state = bound_state(string_state(stv))
                # print type(state), last, state
                sup_change = states_to_line(last, state)


                state_pts = state_to_pixel(state)
                if i > f_rng[0] and (i < f_rng[1] or f_rng[1] == -1):
                    # draw_result(initial_image, sup_change, color = ((nums)*255.0/tot,0,255 - (nums-1)*255.0/tot), thick = 1 )
                    draw_result(initial_image, sup_change, color = (31,139,246), thick = 2 )
                    draw_result(initial_image_white, sup_change, color = (31,139,246), thick = 2 )

                if i > f_rng[1] and f_rng[1] != -1:
                    break
                i += 1
                last = state
    for r_n in init_imgs.keys():
        # print r_n
        display_result([init_imgs[r_n]])

    # display_result([initial_image], ['trajectory distribution'])
    for i in range(420):
        for j in range(420):
            # print i,j
            if np.sum(initial_image_white[i,j]) == 0:
                initial_image_white[i,j] = np.array([255,255,255])
    cv2.imwrite("test_dist_sup_white.png", initial_image_white)
    cv2.imwrite("test_dist_sup.png", initial_image)

def trajectory_rollouts(options, rollout_lst, f_rng, names = [''], tests = ['']):
    '''
    Draws the trajectory of deltas on initial states from sources of several names
    '''
    init_imgs = {}
    nums = 0
    tot = len(names) * len(tests)
    for name in names:
        for test in tests:
            nums+=1
            folname, imname = get_names(name)
            folname, imname = get_test(folname, imname, test)
            start_path = options.rollouts_dir + folname + '/rollout'
            for rollout_num in rollout_lst:
                if rollout_num in init_imgs:
                    initial_image = init_imgs[rollout_num]
                else:
                    full_path = start_path + str(rollout_num) + '/' + imname + 'rollout' + str(rollout_num) + '_frame_0.jpg'
                    initial_image = cv2.imread('doubleplotbackground.jpg')
                    initial_image_white = np.zeros((420, 420, 3))
                    init_imgs[rollout_num] = initial_image
                    print full_path
                state_path = start_path + str(rollout_num) + '/states.txt'
                states = open(state_path, 'r')
                first = True
                last = None
                i = 0
                for stv in states:
                    if first:
                        last = bound_state(string_state(stv))
                        # print last
                        first = False
                        continue

                    state = bound_state(string_state(stv))
                    # print type(state), last, state
                    sup_change = states_to_line(last, state)


                    state_pts = state_to_pixel(state)
                    if i > f_rng[0] and (i < f_rng[1] or f_rng[1] == -1):
                        # draw_result(initial_image, sup_change, color = ((nums)*255.0/tot,0,255 - (nums-1)*255.0/tot), thick = 2 )
                        draw_result(initial_image, sup_change, color = (191,191,0), thick = 2)
                        draw_result(initial_image_white, sup_change, color = (191,191,0), thick = 2 )
                    if i > f_rng[1] and f_rng[1] != -1:
                        break
                    i += 1
                    last = state
    for r_n in init_imgs.keys():
        # print r_n
        display_result([init_imgs[r_n]])
    cv2.imwrite("test_dist_dag.png", initial_image)
    for i in range(420):
        for j in range(420):
            # print i,j
            if np.sum(initial_image_white[i,j]) == 0:
                initial_image_white[i,j] = np.array([255,255,255])
    cv2.imwrite("test_dist_dag_white.png", initial_image_white)


    # display_result([initial_image], ['trajectory distribution'])
    cv2.imwrite("test_dist.jpg", initial_image)

def get_supervised_states_deltas(folname, imname, rollout_num):
    image_path = options.supervised_dir + folname + '/supervised'+str(rollout_num) +'/' + imname + 'supervised'+str(rollout_num) +'_frame_'
    state_path = options.supervised_dir + folname + '/supervised'+str(rollout_num) +'/states.txt'
    states = open(state_path, 'r')
    deltas_path = options.supervised_dir + folname + '/supervised'+str(rollout_num) +'/deltas.txt'
    deltas = open(deltas_path, 'r')
    return states, deltas, image_path

def get_first_images(image_path):
    sup_policy = cv2.imread(image_path + str(0) + '.jpg')
    net_policy = cv2.imread(image_path + str(0) + '.jpg')
    combined_policy = cv2.imread(image_path + str(0) + '.jpg')
    return sup_policy, net_policy, combined_policy

def draw_all(state, f_rng, i, net_policy, sup_policy, combined_policy, img, net_change, sup_change):
    state_pts = state_to_pixel(state)
    # print i, " ", state_pts[0], " ",state
    if i > f_rng[0] and (i < f_rng[1] or f_rng[1] == -1):
        draw_result(net_policy, net_change, color = (0,255,0))
        draw_result(sup_policy, sup_change, color = (0,0,255))
        draw_result(combined_policy, net_change, color = (0,255,0))
        draw_result(combined_policy, sup_change, color = (0,0,255))
        # draw_result(img, state_pts, color = (255,0,0))
        # draw_result(img, net_change, color = (0,255,0))
        # draw_result(img, sup_change, color = (0,0,255))
    # display_result([img])

def compare_supervisor_net_rol(options, net, sess, rollout_lst, f_rng = (0,-1), name=''):
    '''
    Takes in a set of rollouts for a supervisor, and outputs the net values at each state,
    then displays the result
    '''
    folname, imname = get_names(name)
    state_distances = []
    for rollout_num in rollout_lst:
        states, deltas, image_path = get_supervised_states_deltas(folname, imname, rollout_num)
        sup_policy, net_policy, combined_policy = get_first_images(image_path)

        i = 0
        traj_sup_distances = []
        traj_net_distances = []
        traj_state_distances = []

        for stv, delta in zip(states, deltas):
            state = stv
            img = cv2.imread(image_path + str(i) + '.jpg')
            state = bound_state(string_state(stv))
            delta = convert_delta(delta)
            sup_change, true_sup = command_to_line(delta, state)

            net_delta = feed_to_net(sess, net, img)
            net_change, true_net = command_to_line(net_delta, state)
            sup_dist = pixelstoMeters(np.linalg.norm(np.array(sup_change[0]) - np.array(true_sup)))
            net_dist = pixelstoMeters(np.linalg.norm(np.array(net_change[0]) - np.array(true_net)))
            state_dist = pixelstoMeters(np.linalg.norm(np.array(true_sup) - np.array(true_net)))

            traj_sup_distances.append(sup_dist)
            traj_net_distances.append(net_dist)
            traj_state_distances.append(state_dist)
            # print normalize_supervisor(delta)
            # print normalize_supervisor(net_delta)
            supervisor_value, supervisor_norm = normalize_value(delta)
            net_value, net_norm = normalize_value(net_delta)
            draw_all(state, f_rng, i, net_policy, sup_policy, combined_policy, img, net_change, sup_change)

            if i > f_rng[1] and f_rng[1] != -1:
                break
            i += 1
        state_distances.append(traj_state_distances)
        print "state distance error: ", np.sum(traj_state_distances), np.sum(traj_state_distances)/len(traj_state_distances)
        cv2.imwrite("worst_test_ang" + str(rollout_num) + ".jpg", combined_policy)
        states.close()
        deltas.close()
        print "rollout number: ", rollout_num
        # print "error rate: ", rollout_num
        display_result([net_policy,sup_policy, combined_policy], ['net', 'supervisor', 'both'])
        # plt.plot(distances)
        # plt.show(block=False)
        # plt.close()
        # print rollout_num - r_rng[0]

def normalize_value(delta):
    supervisor_value = [0] * 2
    supervisor_value[0] = delta[0]
    supervisor_value[1] = delta[2]
    supervisor_norm = normalize_supervisor(supervisor_value)
    supervisor_value = np.array(supervisor_value)
    return supervisor_value, supervisor_norm

def compare_supervisor_net(options, net, sess, r_rng, f_rng = (0,-1), name='', exclude=[]):
    '''
    gathers error rates on the comparison of the supervisor and the net
    '''
    exclude = set(exclude)

    frame_differences = [np.array((0.0, 0.0)) for i in range(100)]
    frame_totals = [0.0 for i in range(100)]
    ang_rates = []
    fore_rates = []
    rates = []
    costs = []
    frame_counts = []
    all_errs = []
    sup_distances = []
    net_distances = []
    folname, imname = get_names(name)
    for rollout_num in r_rng:
        traj_sup_distances = []
        traj_net_distances = []
        if rollout_num in exclude:
            print "excluded, ", rollout_num
            continue
        states, deltas, image_path = get_supervised_states_deltas(folname, imname, rollout_num)
        sup_policy, net_policy, combined_policy = get_first_images(image_path)

        error_costs, error_ang, error_fore, i = 0.0, 0.0, 0.0, 0
        window = []
        distances = []
        for stv, delta in zip(states, deltas):
            state = stv
            pth =image_path + str(i) + '.jpg'
            img = cv2.imread(pth)
            state, delta = bound_state(string_state(stv)), convert_delta(delta)
            sup_change, true_sup = command_to_line(delta, state)
            net_delta = feed_to_net(sess, net, img)
            net_change, true_net = command_to_line(net_delta, state)

            sup_dist = pixelstoMeters(np.linalg.norm(np.array(sup_change[0]) - np.array(true_sup)))
            net_dist = pixelstoMeters(np.linalg.norm(np.array(net_change[0]) - np.array(true_net)))
            traj_sup_distances.append(sup_dist)
            traj_net_distances.append(net_dist)


            # print normalize_supervisor(net_delta)
            supervisor_value, supervisor_norm = normalize_value(delta)
            net_value, net_norm = normalize_value(net_delta)
            
            err_val = np.linalg.norm(net_norm - supervisor_norm)
            all_errs.append((pth, err_val, net_change, sup_change))
            error_costs += err_val
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
            draw_all(state, f_rng, i, net_policy, sup_policy, combined_policy, img, net_change, sup_change)
            if i > f_rng[1] and f_rng[1] != -1:
                break
            i += 1
        frame_counts.append(i)
        sup_distances.append(traj_sup_distances)
        net_distances.append(traj_net_distances)
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
    sorting_fn = lambda x: x[1]
    all_errs.sort(key = sorting_fn)
    ang_rates.sort()
    fore_rates.sort()
    for val in all_errs[-100:]:
        bad_frame = cv2.imread(val[0])
        draw_result(bad_frame, val[2], color = (0,0,255))
        draw_result(bad_frame, val[3], color = (0,255,0))
        display_result([bad_frame], ['bad_frame'])

    print "total number of frames (on a rollout)"
    print frame_counts
    print np.sum(frame_counts)
    print "total trajectory costs"
    print (indices + r_rng[0]).tolist()
    print "angle trajectory costs"
    print (rate_indices + r_rng[0]).tolist()
    print "forward trajectory costs"
    print (fore_indices + r_rng[0]).tolist()
    plt.plot(ang_rates)
    plt.plot(fore_rates)
    # plt.plot(rates)
    fig = plt.figure(1)
    fig.canvas.set_window_title('traj_sorted')
    plt.show()
    plt.close()
    fig = plt.figure(1)
    fig.canvas.set_window_title('norm_rates')
    plt.plot(rates)
    plt.show()
    for i in range(100):
        frame_differences[i] = -frame_differences[i]/(r_rng[0]-r_rng[1])
    fig = plt.figure(1)
    fig.canvas.set_window_title('frames')
    plt.plot(frame_differences)
    plt.show()
    for i in range(100):
        frame_totals[i] = -frame_totals[i]/(r_rng[0]-r_rng[1])
    fig = plt.figure(1)
    fig.canvas.set_window_title('frames_norm')
    plt.plot(frame_totals)
    plt.show()


def compare_policies(path_image, r_rng, f_rng, paths_first_states, paths_first_deltas, paths_second_states, paths_second_deltas, options):
    '''
    Displays the policy of the first and second policies on the first state of the first set of data
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
                draw_result(img, f_change, color = (0,191,191))
                draw_result(img, s_change, color = (246,139,31))
            j += 1
        f_states.close()
        f_deltas.close()
        s_states.close()
        s_deltas.close()
        print i - r_rng[0], pf_states, ps_states
        display_result([img], ['both'])
        cv2.imwrite("net_fail" + str(r_rng[i]) + ".jpg", img)
        i += 1

def traj_num(line):
    return int(line[line.find('_supervised') + len('_supervised'):line.find('_frame_')])

def compare_supervisor_deltas(options, net, sess, r_rng, exclude=[], f_rng = (0,-1), name='', o_rng = (0,10)):
    '''
    gathers error rates on the comparison of the supervisor and retroactive feedback of the same form
    '''
    frame_num = lambda x: int(x[x.find('_frame_') + len('_frame_'):x.find('.jpg')])
    exclude = set(exclude)
    frame_differences = [np.array((0.0, 0.0)) for i in range(100)]
    frame_totals = [0.0 for i in range(100)]
    ang_rates = []
    fore_rates = []
    rates = []
    costs = []
    sup_distances, net_distances, state_distances = [], [], []
    folname, imname = get_names(name) 
    other_deltas_path = options.supervised_dir + folname + 'retroactive_feedback' +str(o_rng[0]) +'_'+str(o_rng[1]) + '.txt'
    other_deltas = open(other_deltas_path, 'r')
    print other_deltas_path
    for rollout_num in r_rng:
        if rollout_num in exclude:
            continue
        image_path = options.supervised_dir + folname + 'supervised'+str(rollout_num) +'/' + imname + 'supervised'+str(rollout_num) +'_frame_'
        state_path = options.supervised_dir + folname + '/supervised'+str(rollout_num) +'/states.txt'
        states = open(state_path, 'r')
        deltas_path = options.supervised_dir + folname + 'supervised'+str(rollout_num) +'/deltas.txt'
        deltas = open(deltas_path, 'r')



        sup_policy = cv2.imread(image_path + str(0) + '.jpg')
        other_policy = cv2.imread(image_path + str(0) + '.jpg')
        # combined_policy = cv2.imread(image_path + str(0) + '.jpg')
        # combined_policy = cv2.imread('subtracted-Dave_supervised1_frame_0.jpg')
        combined_policy = cv2.imread('subtract-Sherdil_supervised2_frame_0.jpg')

        error_costs = 0.0
        error_ang = 0.0
        error_fore = 0.0
        i = 0
        window = []
        distances = []
        other_delta = other_deltas.next()
        stv = states.next()
        delta = deltas.next()
        print "before: ", frame_num(delta), frame_num(other_delta)
        while frame_num(delta) < frame_num(other_delta):
            try:
                stv = states.next()
                delta = deltas.next()
                print "advanced"
            except StopIteration as e:
                print "after 1: ", frame_num(delta), frame_num(other_delta)
                break
        while frame_num(delta) > frame_num(other_delta):
            try:
                other_delta = other_deltas.next()
            except StopIteration as e:
                print "after 2: ", frame_num(delta), frame_num(other_delta)
                break
        traj_sup_distances, traj_net_distances, traj_state_distances = [], [], []
        first = True
        for stv, delta, other_delta in zip(states, deltas, other_deltas):
            if traj_num(other_delta) > traj_num(delta):
                break
            print other_delta, delta
            state = stv

            delta = convert_delta(delta)
            state = bound_state(string_state(stv))
            sup_change, true_sup = command_to_line(delta, state)
            ## added for line interpolation
            if first:
                last = bound_state(string_state(stv))
                # print last
                first = False
            state = bound_state(string_state(stv))
            sup_change = states_to_line(last, state)
            ##

            img = cv2.imread(image_path + str(i) + '.jpg')
            other_delta = convert_delta(other_delta)
            # sup_change = command_to_line(delta, state)
            # other_change = command_to_line(other_delta, state)
            # other_delta[2]  = .004
            other_change, true_other = command_to_line(other_delta, state)

            sup_dist = pixelstoMeters(np.linalg.norm(np.array(sup_change[0]) - np.array(true_sup)))
            other_dist = pixelstoMeters(np.linalg.norm(np.array(other_change[0]) - np.array(true_other)))
            state_dist = pixelstoMeters(np.linalg.norm(np.array(true_sup) - np.array(true_other)))

            traj_sup_distances.append(sup_dist)
            traj_net_distances.append(other_dist)
            traj_state_distances.append(state_dist)

            supervisor_value = [0] * 2
            supervisor_value[0] = delta[0]
            supervisor_value[1] = delta[2]
            supervisor_norm = normalize_supervisor(supervisor_value)
            other_value = [0] * 2
            other_value[0] = other_delta[0]
            other_value[1] = other_delta[2]
            other_norm = normalize_supervisor(other_value)
            other_value = np.array(other_value)
            supervisor_value = np.array(supervisor_value)
            error_costs += np.linalg.norm(other_norm - supervisor_norm)
            error_ang += np.abs(supervisor_value[0] - other_value[0])
            error_fore += np.abs(supervisor_value[1] - other_value[1])

            frame_differences[i] += np.array((np.abs(supervisor_value[0] - other_value[0]), np.abs(supervisor_value[1]-other_value[1])))
            frame_totals[i] += np.linalg.norm(other_norm-supervisor_norm)
            state_pts = state_to_pixel(state)
            if i > f_rng[0] and (i < f_rng[1] or f_rng[1] == -1):
                draw_result(other_policy, other_change, color = (0,255,0))
                draw_result(sup_policy, sup_change, color = (0,0,255))
                draw_result(combined_policy, other_change, color = (211,211,20), thick=2)
                draw_result(combined_policy, sup_change, color = (0,0,0), thick=4)
            #     draw_result(img, state_pts, color = (255,0,0))
            #     draw_result(img, other_change, color = (0,255,0))
            #     draw_result(img, sup_change, color = (0,0,255))
            # display_result([img])
            if i > f_rng[1] and f_rng[1] != -1:
                break
            i += 1
            ## Added for line interpolation
            last = state
            ##
        sup_distances.append(traj_sup_distances)
        net_distances.append(traj_net_distances)
        state_distances.append(traj_state_distances)
        display_result([other_policy,sup_policy, combined_policy], ['other', 'supervisor', 'both'])
        cv2.imwrite("supervised_feedback" + str(rollout_num) + ".png", combined_policy)
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
    print indices.tolist()
    print "angle trajectory costs"
    print rate_indices.tolist()
    print "forward trajectory costs"
    print fore_indices.tolist()
    plt.plot(ang_rates)
    plt.plot(fore_rates)
    # plt.plot(rates)
    fig = plt.figure(1)
    fig.canvas.set_window_title('traj_sorted')
    plt.show()
    plt.close()
    fig = plt.figure(1)
    fig.canvas.set_window_title('norm_rates')
    plt.plot(rates)
    plt.show()
    for i in range(100):
        frame_differences[i] = -frame_differences[i]/(r_rng[0]-r_rng[1])
    fig = plt.figure(1)
    fig.canvas.set_window_title('frames')
    plt.plot(frame_differences)
    plt.show()
    for i in range(100):
        frame_totals[i] = -frame_totals[i]/(r_rng[0]-r_rng[1])
    fig = plt.figure(1)
    fig.canvas.set_window_title('frames_norm')
    plt.plot(frame_totals)
    plt.show()
    tot = np.max(list(r_rng)) - np.min(list(r_rng))
    print "trajectory costs: ", [np.sum(traj_dists) for traj_dists in state_distances]
    print "trajectory average costs: ", np.sum([np.sum(traj_dists) for traj_dists in state_distances])/len(state_distances)
    print "trajectory average costs per step: ", np.sum([np.sum(traj_dists)/len(traj_dists) for traj_dists in state_distances])/len(state_distances)
    print "best average error: ", np.min(rates)
    print "worst average error: ", np.max(rates)
    print "normalized average error: ", np.sum(rates)/tot, np.sum(rates)
    print "foreward average error: ", np.sum(fore_rates)/tot
    print "angle average error: ", np.sum(ang_rates)/tot

def get_rollout_paths(r_rng, options, name=''):
    s_paths, d_paths = [], []
    folname, imname = get_names(name) 
    for rollout_num in r_rng:
        state_path = options.rollouts_dir + folname + 'rollout'+str(rollout_num) +'/states.txt'
        deltas_path = options.rollouts_dir + folname + 'rollout'+str(rollout_num) +'/net_deltas.txt'
        s_paths.append(state_path)
        d_paths.append(deltas_path)
    return s_paths, d_paths

def get_supervised_paths(r_rng, options, name=''):
    s_paths, d_paths = [], []
    folname, imname = get_names(name) 
    for rollout_num in r_rng:
        state_path = options.supervised_dir + folname + 'supervised'+str(rollout_num) +'/states.txt'
        deltas_path = options.supervised_dir + folname + 'supervised'+str(rollout_num) +'/deltas.txt'
        s_paths.append(state_path)
        d_paths.append(deltas_path)
    return s_paths, d_paths

def get_test(folname, imname, test):
    if len(test) != 0:
        folname = folname + test + '/'
        imname = imname + test + '_'
    print test, folname
    return folname, imname

def assign_path(image_path, rollout_num, options, name='', test=''):
    folname, imname = get_names(name)
    folname, imname = get_test(folname, imname, test)
    if image_path == options.supervised_dir:
        return options.supervised_dir + folname + 'supervised'+str(rollout_num) +'/supervised'+str(rollout_num) + '_frame_' + str(0) + '.jpg'
    elif image_path == options.rollouts_dir:
        return options.rollouts_dir + folname + 'rollout'+str(rollout_num) +'/rollout'+str(rollout_num) + '_frame_' + str(0) + '.jpg'

    return None

def evaluate_supervisor_net(options, net, sess, training=False):
    '''
    gathers error rates on the comparison of the supervisor and the net
    '''
    data = inputdata.AMTData(options.train_file, options.test_file,channels=3)
    losses = [0,0,0]
    if training:
        full_batch = data.all_train_batch(500)
    else:
        full_batch = data.all_test_batch(500)
    j = 0
    with sess.as_default():
        while full_batch is not None:
            full_ims, full_labels = full_batch
            full_dict = { net.x: full_ims, net.y_: full_labels }
            print "Evaluating... finished ", j
            num_used = len(full_ims)
            j += len(full_ims)
            if training:
                full_batch = data.all_train_batch(500)
            else:
                full_batch = data.all_test_batch(500)
            rot_loss = net.rot.eval(feed_dict=full_dict) # regression
            fore_loss = net.fore.eval(feed_dict=full_dict) # regression
            norm_loss = rot_loss/.02666667 + fore_loss/.006
            losses = [losses[0] + rot_loss*num_used , losses[1] + fore_loss*num_used, losses[2]+norm_loss*num_used]
        losses = np.array(losses)/j
    outf = open(AMTOptions.amt_dir + 'testing_outputs.txt', 'a+')
    finals = open(AMTOptions.amt_dir + 'testing_values.txt', 'a+')
    outf.write('Is training loss: ' + str(training) + ', values: ' + str(losses) + '\n')
    finals.write(str(training) + '\t' + str(losses) + '\n')
    outf.close()
    finals.close()

if __name__ == '__main__':
    options = AMTOptions()
    net = net6.NetSix()
    options.tf_net = net
    # options.tf_net = net6.NetSix()
    persons = {}
    # order: sup_20, sup_40, sup_60, DAg_20, DAg_40
    persons['Jonathan'] = ['/media/1tb/Izzy/nets/net6_07-14-2016_16h10m54s.ckpt', '/media/1tb/Izzy/nets/net6_07-15-2016_11h48m14s.ckpt', '/media/1tb/Izzy/nets/net6_07-15-2016_11h57m33s.ckpt', '/media/1tb/Izzy/nets/net6_07-14-2016_17h46m20s.ckpt', '/media/1tb/Izzy/nets/net6_07-15-2016_10h44m06s.ckpt']
    persons['Jacky'] = ['/media/1tb/Izzy/nets/net6_07-18-2016_13h43m29s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_15h47m45s.ckpt', '/media/1tb/Izzy/nets/net6_07-25-2016_10h45m25s.ckpt', '/media/1tb/Izzy/nets/net6_07-18-2016_14h32m27s.ckpt', '/media/1tb/Izzy/nets/net6_07-25-2016_14h14m23s.ckpt']
    persons['Aimee'] = ['/media/1tb/Izzy/nets/net6_07-19-2016_16h47m52s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_14h57m06s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_15h06m14s.ckpt', '/media/1tb/Izzy/nets/net6_07-19-2016_17h27m25s.ckpt', '/media/1tb/Izzy/nets/net6_07-19-2016_17h59m22s.ckpt']
    persons['Chris'] = ['/media/1tb/Izzy/nets/net6_07-20-2016_10h28m56s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_10h38m04s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_13h04m31s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_11h21m08s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_11h51m23s.ckpt']
    persons['Dave'] = ['/media/1tb/Izzy/nets/net6_07-20-2016_11h08m10s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_15h37m33s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_15h50m37s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_11h53m16s.ckpt', '/media/1tb/Izzy/nets/net6_07-20-2016_15h26m59s.ckpt']
    persons['Lauren'] = ['/media/1tb/Izzy/nets/net6_07-21-2016_12h36m29s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_13h56m55s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_14h06m39s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_13h17m00s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_14h16m19s.ckpt']
    persons['Johan'] = ['/media/1tb/Izzy/nets/net6_07-21-2016_15h30m43s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_13h42m56s.ckpt', '/media/1tb/Izzy/nets/net6_07-22-2016_13h52m02s.ckpt', '/media/1tb/Izzy/nets/net6_07-21-2016_16h17m35s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_11h51m53s.ckpt']
    persons['Sherdil'] = ['/media/1tb/Izzy/nets/net6_07-25-2016_12h38m34s.ckpt', '/media/1tb/Izzy/nets/net6_07-25-2016_13h40m49s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_12h40m57s.ckpt', '/media/1tb/Izzy/nets/net6_07-25-2016_13h12m20s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_13h29m25s.ckpt']
    persons['Richard'] = ['/media/1tb/Izzy/nets/net6_07-26-2016_16h12m46s.ckpt', '/media/1tb/Izzy/nets/net6_07-27-2016_15h02m33s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_17h45m12s.ckpt', '/media/1tb/Izzy/nets/net6_07-26-2016_16h42m18s.ckpt', '/media/1tb/Izzy/nets/net6_07-27-2016_12h26m57s.ckpt']
    persons['Sona'] = ['/media/1tb/Izzy/nets/net6_07-27-2016_18h22m44s.ckpt', '/media/1tb/Izzy/nets/net6_07-28-2016_16h16m38s.ckpt', '/media/1tb/Izzy/nets/net6_07-28-2016_16h31m52s.ckpt', '/media/1tb/Izzy/nets/net6_07-28-2016_11h14m28s.ckpt', '/media/1tb/Izzy/nets/net6_07-28-2016_16h07m15s.ckpt']

    # rollout_num = 63
    # best_ang = [344, 357, 341, 322, 374, 319, 315, 349, 342, 373]
    # worst_ang = [323, 356, 331, 317, 351, 336, 367, 345, 324, 366]

    # best_fore = [317, 335, 329, 343, 337, 348, 322, 370, 323, 351]
    # worst_fore = [325, 369, 332, 326, 339, 346, 324, 364, 359, 353]

    # worst_test = [441, 439, 448, 440, 453]

    # trajectories = [250, 251] + list(range(253, 280)) + [252]

    # rollout_traj = [267, 265, 274, 266, 279]
    # supervised_traj = 
    # supervised_traj = worst_test 0,2,8,9, 18 21 23 28 47 55 58 73 75 78 79 83 86 96 111 121
    # [60, 53, 49, 51, 67, 20, 48, 107, 105, 89, 97, 70, 119, 57, 24, 32, 3, 108, 17, 100, 104, 81, 59, 114, 120]
    # [60, 51, 48, 89, 119, 32, 17, 81, 120]
     #    [ 69  28  48  85  53  37  72  49  62  15  75  80  60 124  97  99  52  33
     #      103 122  29  18 105  71  51  23  82  20  74  13  54  67  58  73  14  87
     #      63  93   5  39  86   9  90  77  61  24 111  92 121 106 109  46  84  42
     #      40  31   7 112   0 119 100  35  91  98  36  76  57   8  22  25 108   3
     #      78  45  50  12  43  30  96  38  19 118 113   6  11   1  44 123 110 107
     #      56  70  89  83  47  10  41 116  65 120  94  21  88  16  55  27  68 114
     #      101  34  59  26  66 115   4  64 104  32 102  79   2  81  17  95 117]
     # 69 15 103 20 63 24 40
    # exclude = [2, 4, 17, 26, 32, 34, 35, 44, 64, 65, 66, 68, 79, 81, 94, 95, 102, 104, 107, 115, 117]
    # exclude = [96, 100, 81, 9, 112, 17, 53, 22, 27, 29, 95]
    exclude = [68, 101, 71, 73, 74, 107, 109, 80, 82, 115, 105, 120, 121, 92]
    # exclude = []
    # hards = [0, 31, 1, 3, 26, 19, 89, 91, 8, 96, 9, 49, 27, 25, 42, 68, 53, 51, 84, 81, 103]
    # hards = [19, 110, 0, 70, 10, 47, 49, 29, 106, 25, 82, 88, 27, 42, 89, 31, 81, 84, 53, 103]
    # hards = [113, 83, 104, 12, 56, 71, 90, 66, 63, 67, 57, 28, 54, 59, 111, 55, 69, 74, 39, 5, 50, 75, 101, 97, 20, 48, 16, 13, 65, 87, 79, 99, 32, 34, 44, 38, 85, 102, 112, 95, 73, 22, 76, 17, 24, 4, 37, 107, 92, 23, 61, 77, 78, 30, 14, 72, 105, 46, 64, 7, 35, 93, 47, 60, 109, 94, 11, 58, 62, 45, 18, 110, 80, 43, 98, 40, 41, 106, 21, 86, 88, 6, 52, 108, 33, 15, 29, 36, 82, 100, 2, 70, 10, 0, 31, 1, 3, 26, 19, 89, 91, 8, 96, 9, 49, 27, 25, 42, 68, 53, 51, 84, 81, 103]


    options.tf_net_path = sys.argv[1]
    # options.tf_net_path = '/media/1tb/Izzy/nets/net6_09-03-2016_15h07m52s.ckpt'
    sess = net.load(var_path=options.tf_net_path)
    # compare_supervisor_net_rol(options, net, sess, exclude, (0,-1), name='Caleb')

    evaluate_supervisor_net(options, net, sess, training=True)
    evaluate_supervisor_net(options, net, sess, training=False)
    # compare_supervisor_net(options, net, sess, range(65, 125), (0,-1), name='Caleb', exclude=exclude)
    # compare_supervisor_net(options, net, sess, range(425, 455), (0,-1))
    # compare_supervisor_net(options, net, sess, range(315, 415), (0,-1))

    # compare_supervisor_deltas(options, net, sess, range(0,5), f_rng = (0,-1), name='Jonathan', o_rng=(0,4))


    # trajectory_rollouts(options, list(range(0,30)),(0,-1), names=['Jonathan', 'Aimee'], tests=['sup_20_test','sup_40_test','sup_60_test'])
    # trajectory_rollouts(options, list(range(0,30)),(0,-1), names=['Johan', 'Richard', 'Dave', 'Sona', "Lauren", 'Jacky'], tests=['sup_60_test'])
    # trajectory_supervised(options, list(range(25,26)),(0,-1), names=['Jonathan', 'Aimee', 'Chris', 'Sherdil', 'Johan', 'Richard', 'Dave', 'Sona', 'Jacky', 'ChJon'])
    # trajectory_rollouts(options, list(range(5,6)),(0,-1), names=['Jonathan', 'Aimee', 'Chris', 'Sherdil', 'Johan', 'Richard', 'Dave', 'Sona', 'Jacky', 'ChJon'])
    # trajectory_supervised(options, list(range(0,60)),(0,-1), names=['Jonathan', "Lauren", "Jacky"])

    # sup_states, sup_deltas = get_supervised_paths(supervised_traj, options)
    # net_states, net_deltas = get_rollout_paths(rollout_traj, options)
    # compare_policies(options.supervised_dir, supervised_traj, (0,-1), sup_states, sup_deltas, net_states, net_deltas, options)

    # sup_states, sup_deltas = get_supervised_paths((123,143), options)
    # net_states, net_deltas = get_rollout_paths((1820,1840), options)
    # compare_policies(options.supervised_dir, (123,143), (0,-1), sup_states, sup_deltas, net_states, net_deltas, options)

    sess.close()
