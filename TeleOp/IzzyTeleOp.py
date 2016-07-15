from gripper.TurnTableControl import *
from gripper.PyControl import *
from gripper.xboxController import *
from options import AMTOptions
from pipeline.bincam import BinaryCamera
import sys, os, time, cv2, argparse
import tty, termios
from Net.tensor import inputdata
from scripts.objects import singulationImg
from scripts import overlay, click_centers, detectSingulatedCam


sys.path[0] = sys.path[0] + '/../../../GPIS/src/grasp_selection/control/DexControls'
        
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState
# from DexRobotTurntable import DexRobotTurntable
# from TurntableState import TurntableState

# ROTATE_UPPER_BOUND = 3.82
# ROTATE_LOWER_BOUND = 3.06954

# GRIP_UPPER_BOUND = .06
# GRIP_LOWER_BOUND = .0023

# TABLE_LOWER_BOUND = .002
# TABLE_UPPER_BOUND = 7.0 


def getch():
    """
        Pause the program until key press
        Return key press character
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def teleop(c, izzy, t, bc, name, template):
    target_state_i = get_state(izzy.getState())
    # target_state_t = get_state(t.getState())
    print "Start"
    i = 0
    frames = []
    deltas_lst = []
    states = []
    avg_angd = 0.0
    max_angd = 0.0
    max_ford = 0.0
    avg_ford = 0.0
    past_deltas = []
    length_pass = 3
    try:
        # for i in range(4):
        #     bc.vc.grab()
        while i < 100:
            start = time.time()
            controls = c.getUpdates()
            deltas, stop = controls2deltas(controls)

            print stop
            if stop:
                break

            for _ in range(4):
                bc.vc.grab()

            # cv2.imshow("camera", frame)
            # cv2.waitKey(30)

            if not all(d == 0.0 for d in deltas):
                print i
                i += 1

                frame = bc.read_frame()
                disp_frame = inputdata.im2tensor(frame, channels = 3)
                # vis_frame = detectSingulatedCam.highlight(disp_frame * 255)
                frames.append(frame)
                cv2.imshow("camera", disp_frame)
                cv2.waitKey(30)
                deltas_lst.append(deltas)
                # print "Current: ", target_state_i, target_state_t
                # true_cur = izzy.getState().state
                print "Current: ", target_state_i#, "true current: ", true_cur
                # target_state_i = true_cur
                # avg_angd += target_state_i[0] - true_cur[0]
                # max_angd = max(max_angd, target_state_i[0] - true_cur[0])
                # max_ford = max(max_ford, target_state_i[2] - true_cur[2])
                # avg_ford += target_state_i[2] - true_cur[2]

                states.append(target_state_i[:])
                # new_izzy, new_t = apply_deltas(deltas, target_state_i, target_state_t)

                past_deltas.append((deltas[0], deltas[1]))
                if len(past_deltas) > length_pass:
                    past_deltas.pop(0)
                deltas[0] = np.sum([past_delta[0] for past_delta in past_deltas])/len(past_deltas)
                deltas[1] = np.sum([past_delta[1] for past_delta in past_deltas])/len(past_deltas)

                new_izzy = apply_deltas(deltas, target_state_i)
                print "Deltas: ", deltas
                # target_state_i, target_state_t = new_izzy, new_t 
                target_state_i = new_izzy 
                # print "Teleop: ", new_izzy, new_t
                print "Teleop: ", new_izzy
                izzy._zeke._queueState(ZekeState(new_izzy))
                # t.gotoState(TurntableState(new_t), .25, .25)

            offset = max(0, .3 - (time.time() - start))
            print "offset time: ", offset
            time.sleep(offset)                
            print "total time: ", time.time() - start
            #if i == 501:
            #    break

    except KeyboardInterrupt:
        pass
    stop = False
    return_to_start(izzy)
    # print avg_angd/100.0, max_angd, avg_ford/100.0, max_ford
    return prompt_save(frames, deltas_lst, states, name, template)


def return_to_start(izzy):
    target_state_i = get_state(izzy.getState())
    current_state = target_state_i
    destination = np.array([3.5857, 0.0017, 0.0117, 1.1239, 0.0002, 0.0])
    while np.linalg.norm(current_state - destination) > .001:
        print np.linalg.norm(current_state - destination)
        print safety(destination - np.array(current_state))
        current_state = current_state + safety(destination - np.array(current_state))
        izzy._zeke._queueState(ZekeState(current_state))
        time.sleep(.1)
        print current_state
    time.sleep(.25)

def safety(delta):
    delta[0] = np.sign(delta[0]) * min(abs(delta[0]), .02)
    delta[2] = np.sign(delta[2]) * min(abs(delta[2]), .007)
    return delta


def prompt_save(frames, deltas, states, name, template):
    num_rollouts = len(rollout_dirs())
    print "Save if you were successful. Save this one? (y/n): "
    char = getch()
    if char == 'y':
        # print len(deltas)
        # print len(frames)
        # print states
        return save_recording(frames, deltas, states, name, template)
    elif char == 'n':
        return False
    prompt_save(frames, deltas, states, name, template)

def rollout_dirs():
        return list(os.walk(AMTOptions.supervised_dir))[0][1]

def next_rollout(start_path):
    """
    :return: the String name of the next new potential rollout
            (i.e. do not overwrite another rollout)
    """
    i = 0
    prefix = start_path + 'supervised'
    path = prefix + str(i) + "/"
    while os.path.exists(path):
        i += 1
        path = prefix + str(i) + "/"
    return 'supervised' + str(i)

def color(frame): 
    color_frame = cv2.resize(frame.copy(), (250, 250))
    cv2.imwrite('get_jp.jpg',color_frame)
    color_frame= cv2.imread('get_jp.jpg')
    return color_frame

def lst2str(lst):
    """
        returns a space separated string of all elements. A space
        also precedes the first element.
    """
    s = ""
    for el in lst:
        s += " " + str(el)
    return s

def save_recording(frames, deltas, states, name, template):
    if name:
        start_path = AMTOptions.supervised_dir + name + "_rollouts/"
    else:
        start_path = AMTOptions.supervised_dir
    try:
        os.makedirs(start_path)
    except OSError as e:
        pass

    rollout_name = next_rollout(start_path)
    rollout_path =  start_path + rollout_name +'/'
    try:
        os.makedirs(rollout_path)
    except OSError as e:
        pass
    print "Saving rollout to " + rollout_path + "..."
    deltas_file = open(rollout_path + 'deltas.txt', 'a+')
    states_file = open(rollout_path + 'states.txt', 'a+')

    # template = cv2.imread("/home/annal/Izzy/vision_amt/scripts/objects/template.png")
    np.save(rollout_path + "template.npy", template)


    i = 0
    for frame, delta, state in zip(frames, deltas, states):
        if name is not None:
            filename = name + '_' + rollout_name + '_frame_' + str(i) + '.jpg'
        else:
            filename = rollout_name + '_frame_' + str(i) + '.jpg'
        deltas_file.write(filename + lst2str(delta) + '\n')
        states_file.write(filename + lst2str(state) + '\n')
        cv2.imwrite(rollout_path + filename, frame)
        cv2.imwrite(AMTOptions.colors_dir + filename, color(frame))
        i += 1
    return True

def get_state(state):
    if isinstance(state, ZekeState): #or isinstance(state, TurntableState):
        return state.state
    return state

def controls2deltas(controls):
    deltas = [0.0] * 4
    stop = False
    if controls == None:
        return None, True
    # deltas[0] = controls[0] / 5300#300.0
    # deltas[1] = controls[2] / 30000#9000#1000.0
    deltas[0] = controls[0] / 1500.0
    deltas[1] = controls[2] / 20000.0
    deltas[2] = controls[4] / 8000.0
    deltas[3] = controls[5] / 800.0
    if abs(deltas[0]) < 8e-8:
        deltas[0] = 0.0
    if abs(deltas[1]) < 8e-4:#8e-4: #2e-2:
        deltas[1] = 0.0
    if abs(deltas[2]) < 5e-3:
        deltas[2] = 0.0
    if abs(deltas[3]) < 2e-2:
        deltas[3] = 0.0 
    deltas[2] = 0.0
    deltas[3] = 0.0
    if deltas[1] < 0:
        deltas[1] = 0.0
    print "angles: ", deltas[0]
    print "forwards: ", deltas[1]
    return deltas, stop

def controls2classes(controls):
    stop = False
    if controls == None:
        return None, True
    deltas = [0.0] * 4
    # deltas[0] = controls[0] / 5300#300.0
    # deltas[1] = controls[2] / 30000#9000#1000.0
    deltas[0] = controls[0] / 2000.0
    deltas[1] = controls[2] / 20000.0
    deltas[2] = controls[4] / 8000.0
    deltas[3] = controls[5] / 800.0
    if abs(deltas[0]) < 8e-8:
        deltas[0] = 0.0
    if abs(deltas[1]) < 8e-4:#8e-4: #2e-2:
        deltas[1] = 0.0
    if abs(deltas[2]) < 5e-3:
        deltas[2] = 0.0
    if abs(deltas[3]) < 2e-2:
        deltas[3] = 0.0 
    deltas[2] = 0.0
    deltas[3] = 0.0

    # if controls[4] != 0.0:
    #     stop = True

    angles = np.array(AMTOptions.CLASS_ANGLES)
    deltas[0] = angles[np.argmin(np.abs(angles - deltas[0]))]

    forwards = np.array(AMTOptions.CLASS_FORWARD)
    deltas[1] = forwards[np.argmin(np.abs(forwards - deltas[1]))]

    return deltas, stop

    

# def apply_deltas(delta_state,t_i,t_t):
def apply_deltas(delta_state,t_i):
    """
        Get current states and apply given deltas
        Handle max and min states as well
    """
    
    t_i[0] += delta_state[0]
    # t_i[1] = 0.00952 # for zeke
    t_i[1] = 0.0017
    t_i[2] += delta_state[1]
    # t_i[3] = 4.211 # for zeke
    t_i[3] = 1.2668
    t_i[4] = 0.0004#0.054# 0.0544 #delta_state[2]
    # t_t[0] += delta_state[3]
    t_i[0] = min(AMTOptions.ROTATE_UPPER_BOUND, t_i[0])
    t_i[0] = max(AMTOptions.ROTATE_LOWER_BOUND, t_i[0])
    t_i[2] = min(AMTOptions.EXTENSION_UPPER_BOUND, t_i[2])
    t_i[2] = max(AMTOptions.EXTENSION_LOWER_BOUND, t_i[2])
    t_i[4] = min(AMTOptions.GRIP_UPPER_BOUND, t_i[4])
    t_i[4] = max(AMTOptions.GRIP_LOWER_BOUND, t_i[4])
    # t_t[0] = min(TABLE_UPPER_BOUND, t_t[0])
    # t_t[0] = max(TABLE_LOWER_BOUND, t_t[0])

    return t_i#, t_t

def display_template(bc, template=None):
    if template is None:
        template = cv2.imread("/home/annal/Izzy/vision_amt/scripts/objects/template.png")

    template[:,:,1] = template[:,:,2]
    template[:,:,0] = np.zeros((420, 420))
    # template[:,:,2] = np.zeros((420, 420))
    # template = cv2.resize(template, (250, 250))
    while 1:
        frame = bc.read_frame()
        frame = inputdata.im2tensor(frame, channels = 3)
        final = np.abs(-frame + template/255.0)
        cv2.imshow('camera', final)
        a = cv2.waitKey(30)
        if a == 27:
            cv2.destroyAllWindows()
            break
        elif a == ord(' '):
            return 'next', template
        time.sleep(.005)
    return 'display', template

def record_template(bc, template=None):
    # records the process of placing blocks of the template
    if template is None:
        template = cv2.imread("/home/annal/Izzy/vision_amt/scripts/objects/template.png")

    template[:,:,1] = template[:,:,2]
    template[:,:,0] = np.zeros((420, 420))
    # template[:,:,2] = np.zeros((420, 420))
    # template = cv2.resize(template, (250, 250))
    frames = []
    while 1:
        frame = bc.read_frame()
        frame = inputdata.im2tensor(frame, channels = 3)
        final = np.abs(-frame + template/255.0)
        frames.append(final * 255)
        cv2.imshow('camera', final)
        a = cv2.waitKey(30)
        if a == 27:
            cv2.destroyAllWindows()
            break
        elif a == ord(' '):
            return 'next'
        time.sleep(.1)
    rollout_path = AMTOptions.supervised_dir+ 'supervised_template/'
    os.makedirs(rollout_path)
    i = 0
    for frame in frames:
        name = 'template_' + str(i) + '.jpg'
        cv2.imwrite(rollout_path + name, frame)
        i += 1
    print "saved to: ", rollout_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--supervised", help="Run for supervised mode",
    #                 action="store_true")
    # parser.add_argument("-d", "--DAgger", help="Runs for dagger mode",
    #                 action="store_true")
    parser.add_argument("-n", "--name", type=str,
                        help="run experiment, enter your name to save in folder")
    parser.add_argument("-c", "--count", type=int,
                        help="number of rollouts you want to run")
    parser.add_argument("-e", "--experiment", help="Runs in limited experiment mode, requires other flags",
                    action="store_true")
    parser.add_argument("-s", "--start", type=int,
                        help="If you would like to start at a particular iteration number")
    args = parser.parse_args()
    if args.name:
        person = args.name 
    else:
        person = None
    if args.count:
        count = args.count
    else:
        count = 0
    limited = False
    if args.experiment:
        limited = True
    if args.start:
        i_at = args.start
    else:
        i_at = 0

    print count <= 20

    print "person, ", person, "entered: ", (count == 20) 

    options = AMTOptions()
    bc = BinaryCamera('./meta.txt')
    bc.open()
    izzy = DexRobotZeke()
    izzy._zeke.steady(False)
    # t = DexRobotTurntable()
    t = None

    if count == 20 and person is not None:
        template_file = open(options.templates_dir + '/training_first_20_paths.txt', 'r')
    elif count == 40 and person is not None:
        template_file = open(options.templates_dir + '/training_last_40_paths.txt', 'r')
    elif count == 60 and person is not None:
        template_file = open(options.templates_dir + '/training_paths.txt', 'r')
    else:
        template_file = open(options.templates_dir + '/template_paths.txt', 'r')
    for i in range(i_at):
        template_file.next()

 
    c = XboxController([options.scales[0],155,options.scales[1],155,options.scales[2],options.scales[3]])
    return_to_start(izzy)
    last = None
    while True:
        if limited:
            print "Waiting for keypress ('n' -> next, 'p' -> previous, 'q' -> quit)"
        else:
            print "Waiting for keypress ('q' -> quit, 'r' -> rollout, 'u' -> update weights, 't' -> test, 'd' -> demonstrate, 'c' -> compile train/test sets, 'p' -> run on previous template, 'l' -> run templates saved in 'last_templates'): "
        char = getch()
        if limited:
            print "You are at iteration: ", i_at
            if char == 'q':
                print "Quitting..."
                break
            elif char == 'n':
                i_at += 1
                try:
                    name = template_file.next()
                    name = name[:name.find('\n')]
                except StopIteration:
                    print 'Completed all saved templates'
                    continue
                print 'Using template: ' + name
                template = np.load(name)
                last = template
                result = display_template(bc, template)[0]
                print result
                print "Rolling out..."
                if result == 'next':
                    continue
                teleop(c, izzy, t, bc, person, template)
                print "Done rolling out."
            elif char == 'p':
                print "Displaying template"
                if last is not None:
                    display_template(bc, last)
                else:
                    display_template(bc)
                print "Rolling out"
                teleop(c, izzy, t, bc, person, last)
        else:
            if char == 'q':
                print "Quitting..."
                break
            elif char == 'r':
                print "Generating template"
                singulationImg.generate_template()
                print "Displaying template"
                next, template = display_template(bc)
                if next == 'next':
                    continue
                print "Rolling out"
                teleop(c, izzy, t, bc, person, template)
            elif char == 'p':
                print "Displaying template"
                if last is not None:
                    display_template(bc, last)
                else:
                    display_template(bc)
                print "Rolling out"
                teleop(c, izzy, t, bc, person, last)
            elif char == 't':
                print "Iterating through training set"
                i = 0
                success = True
                while True:
                    try:
                        if success:
                            name = template_file.next()
                            name = name[:name.find('\n')]
                    except StopIteration:
                        print "successfully completed all saved templates"
                        break
                    print 'Using template: ' + name
                    template = np.load(name)
                    last = template
                    result = display_template(bc, template)
                    print "Rolling out..."
                    success = teleop(c, izzy, t, bc, person, template)
                    print "Done rolling out."

            elif char == 'l':
                try:
                    name = template_file.next()
                except StopIteration:
                    print 'Completed all saved templates'
                    continue
                
                print 'Using template: ' + name
                template = np.load(name[:name.find('\n')])
                result = display_template(bc, template)
                print "Rolling out..."
                if result != 'next':
                    teleop(c, izzy, t, bc, person, template)
                print "Done rolling out."
    print "Done"
   
