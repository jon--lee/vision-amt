import cv2
import numpy as np
from pipeline.bincam import BinaryCamera
import time
from options import AMTOptions
import numpy as np
from PIL import Image
import sys, argparse
sys.path.append('/home/annal/Izzy/vision_amt/scripts/objects/')
from helper import pasteOn
import tty, termios

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


img = np.zeros((512,512,3), np.uint8)
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
tx, ty = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode, img, tx, ty

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        tx, ty = ix, iy
        # print "hello world"

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # print "held down"
            tx, ty = x,y
            # print tx, ty
            # if mode == True:
            #     cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            # else:
            #     cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ix, iy = -1, -1

        # print drawing
        # if mode == True:
        #     cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        # else:
        #     cv2.circle(img,(x,y),5,(0,0,255),-1)

def pixelstoMeters(val):
    return 0.5461/420*val

def metersToPixels(val):
    return 420/0.5461*val

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
    base_ext = .41 # The extention
    ang_offset = np.pi + .42#0.26760734641 # the offset of horizontal from the angle
    by, bx = 210,250#metersToPixels(.3), metersToPixels(.2)
    ###

    grip_ang = (float(state[0]) - ang_offset)
    grip_ext = (float(state[2]) + base_ext)
    # print grip_ang, grip_ext

    grip_post = np.array([metersToPixels(grip_ext*np.sin(grip_ang)),metersToPixels(grip_ext*np.cos(grip_ang))])
    # print grip_ang
    grip_L = np.array([by*np.sin(grip_ang),by]) #the start position of the gripper

    translation = np.array([bx, by + 420])
    rotation = np.array([[-1, 0], [0, -1]])
    grip_B = np.dot(rotation, -translation)
    grip_post = np.dot(rotation, -translation + grip_post)
    grip_L = np.dot(rotation, -translation + grip_L )
    grip_post = (int(grip_post[0]), int(grip_post[1]))
    grip_L = (int(grip_L[0]), int(grip_L[1]))

    print grip_post, grip_L, grip_B

    return grip_post, grip_L, grip_B


def state_to_value(state):
    L = 420 #size in pixels of the viewscreen
    base_ext = .41 # The extention
    ang_offset = np.pi + .42#0.26760734641 # the offset of horizontal from the angle
    by, bx = 210,250#metersToPixels(.3), metersToPixels(.2)

    grip_ang = (float(state[0]) - ang_offset)
    grip_ext = (float(state[2]) + base_ext)
    # print grip_ang, grip_ext

    grip_post = np.array([metersToPixels(grip_ext*np.sin(grip_ang)),metersToPixels(grip_ext*np.cos(grip_ang))])

    translation = np.array([bx, by + 420])
    rotation = np.array([[-1, 0], [0, -1]])
    grip_post = np.dot(rotation, -translation + grip_post)
    grip_post = (int(grip_post[0]), int(grip_post[1]))


    return grip_post


def convert_locations(state, ix, iy, tx, ty):
    rotation = ix - tx
    extension = iy - ty 
    delta = []
    appear_delta = []

    delta.append(np.sign(rotation) * min(np.abs(pixelstoMeters(rotation))/3, .0266))
    delta.append(np.sign(extension) * min(np.abs(pixelstoMeters(extension))/3, .006))
    appear_delta.append(np.sign(rotation) * min(np.abs(pixelstoMeters(rotation)), .0266 * 4))
    appear_delta.append(np.sign(extension) * min(np.abs(pixelstoMeters(extension)), .006 * 4))
    n_state = list(state)
    n_state[0] += appear_delta[0]
    n_state[2] += appear_delta[1]
    return n_state, delta 


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

def string_state(state):
    split = state.split(' ')
    # print state
    labels = np.array( [ float(x) for x in split[1:] ] )
    # print labels
    return labels

def convert_base_pixel(loc):
    by, bx = 210,250
    L = 420
    translation = np.array([bx, by + L])
    rotation = np.array([[-1, 0], [0, -1]])
    return np.dot(rotation, -translation + loc)

def make_pixels(loc):
    return (int(loc[0]), int(loc[1]))

def base_frame(rot, arm, state, dist=.03):
    c, s = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[c,s], [-s,c]])
    trans = np.dot(rot_mat, np.array([0, metersToPixels(dist+float(state[2]))]))

    
    newarm = arm.rotate(rot * 180/np.pi, expand = True)
    center = np.array((newarm.size[0]/2, newarm.size[1]/2))
    paste = convert_base_pixel(trans + center)

    return paste, newarm

def base_finger(rot, finger, state, offset, dist=.38):
    c, s = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[c,s], [-s,c]])
    trans = np.dot(rot_mat, offset + np.array([0, metersToPixels(dist+float(state[2]))]))
    # trans = trans + offset

    
    newfinger = finger.rotate(rot * 180/np.pi, expand = True)
    # center = np.array((newarm.size[0]/2, newarm.size[1]/2))
    f_center = np.array((newfinger.size[0]/2, newfinger.size[1]/2))
    paste = convert_base_pixel(trans+f_center)

    return paste, newfinger

def save_data(feedback, write_path):
    write_file = open(write_path, 'a+')
    for line in feedback:
        # print line
        write_file.write(str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + '\n')
    write_file.close()

options = AMTOptions()
def draw_rollouts(r_lst, name):

    if name is not None:
        write_path = AMTOptions.rollouts_dir + name + "_rollouts/retroactive_feedback" + str(r_lst[0]) + '_' + str(r_lst[-1]) +'.txt'
    else:
        write_path = AMTOptions.rollouts_dir + "retroactive_feedback" + str(r_lst[0]) + '_' + str(r_lst[-1]) +'.txt'

    if name is not None:
        start_path = AMTOptions.rollouts_dir + name + "_rollouts/"
    else:
        start_path = AMTOptions.rollouts_dir
    write_file = open(write_path, 'w')
    write_file.close()
    cv2.namedWindow('image')
    arm = Image.open("scripts/Arm_lbl_prv.png").rotate(90)
    finger_L = Image.open("scripts/Gripper_t_lbl.png").rotate(90)
    finger_R = Image.open("scripts/Gripper_lbl.png").rotate(90)
    # arm = Image.open("scripts/objects/back.png")
    # pasteOn(arm, full_arm, arm.size[0]/2, 0)
    # pasteOn(finger_L, full_arm, arm.size[0]/2, 100)
    # pasteOn(finger_R, full_arm, arm.size[0]/2, 100)
    global img, drawing, ix, iy, tx, ty
    r_i = 0
    while r_i < len(r_lst):
        rollout_num = r_lst[r_i]
        feedback = []
        rollout_name = 'rollout' + str(rollout_num)
        image_path = start_path + rollout_name +'/' + name + '_' + 'rollout'+str(rollout_num) +'_frame_'
        colors_path = name + '_' + rollout_name + '_frame_'
        state_path = start_path + rollout_name +'/states.txt'
        states = open(state_path, 'r')

        img = np.zeros((420,420,3), np.uint8)
        cv2.setMouseCallback('image',draw_circle)

        lastx, lasty = tx, ty
        for s in range(90):
            start = time.time()
            image_frame = image_path + str(s) + '.jpg'
            img = cv2.imread(image_frame)
            rows,cols,colour = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
            img = cv2.warpAffine(img,M,(cols,rows))
            start = time.time()
            while True:
                cv2.imshow('image',img)
                k = cv2.waitKey(30) & 0xFF
                if k == ord('m'):
                    mode = not mode
                elif k == 27:
                    break
                if time.time() - start > .05:
                    break
        fst = True
        # ix, iy = 
        for i in range(90):
            state = convert_state(states.next())
            image_frame = image_path + str(i) + '.jpg'
            # print image_frame
            img = cv2.imread(image_frame)
            rows,cols,colour = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
            img = cv2.warpAffine(img,M,(cols,rows))
            if drawing:
                # start, end, base = state_to_pixel(state)
                # cv2.line(img,start,end,(0,255,0),2)
                # start,end, base = state_to_pixel(convert_locations(state, ix, iy, tx, ty))
                o_stv = state_to_value(state)
                ix, iy = o_stv[0], o_stv[1]
                newstate, delta = convert_locations(state, ix, iy, tx, ty)
                stv = state_to_value(newstate)
                feedback.append([colors_path + str(i) + '.jpg'] + delta)
                # cv2.circle(img,paste,10,(0,0,255),4)
                cv2.circle(img, stv, 10, (255,0,0), 4)
                cv2.circle(img, o_stv, 10, (0,0,255), 4)

                rot = (newstate[0]- np.pi - .42)
                paste, newarm = base_frame(rot, arm, newstate)

                paste_L, f_L = base_finger(rot, finger_L, newstate, np.array([16,0]))
                paste_R, f_R = base_finger(rot, finger_R, newstate, np.array([-16,0]))

                pil_img = Image.fromarray(img)
                pasteOn(pil_img, newarm, paste[0], paste[1])
                print paste_R, ix, iy, tx, ty, delta
                pasteOn(pil_img, f_R, paste_R[0], paste_R[1])
                pasteOn(pil_img, f_L, paste_L[0], paste_L[1])

                img = np.array(pil_img)
                # cv2.rectangle(img,(ix,iy),(tx,ty),(0,255,0),-1)
            
            start = time.time()
            while True:
                if drawing:
                    if tx != lastx and ty != lasty:
                        # cv2.rectangle(img,(ix,iy),(tx, ty),(0,255,0),-1)
                        lastx, lasty = tx,ty
                cv2.imshow('image',img)
                k = cv2.waitKey(30) & 0xFF
                if k == ord('m'):
                    mode = not mode
                elif k == 27:
                    break
                elif k == ord(' '):
                    fst = False
                    break
                if time.time() - start > .35 and not fst:
                    if i == 89:
                        blue = np.ones((420, 420)).astype('uint8') * 20
                        img[:,:,0] += blue
                        cv2.imshow('image',img)
                        k = cv2.waitKey(30) & 0xFF
                    break

            print "next frame: ", i
            # if name is not None:
            #     target_path = AMTOptions.rollouts_dir + name + "_rollouts/feedback_img_" + str(i) + ".jpg"
            #     cv2.imwrite(target_path, img)
        print "You are at iteration: ", r_i
        print "do you want to continue? enter (y/n)"
        char = getch()
        if char == 'y':
            print "would you like to save your last run? enter (y/n)"
            char = getch()
            if char == 'y':
                save_data(feedback, write_path)
                r_i += 1
            else:
                continue
        if char == 'n':
            break
    # print feedback
    


    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str,
                        help="run experiment, will prompt for name")
    parser.add_argument("-f", "--first", type=int,
                        help="enter the starting value of rollout")
    parser.add_argument("-l", "--last", type=int,
                        help="enter the last value of the rollout")
    args = parser.parse_args()
    if args.name is not None:
        person = args.name 
    else:
        person = None
    if args.first is not None:
        first = args.first
    else:
        print "please enter a first value with -f"
        sys.exit()
    if args.last is not None:
        last = args.last
    else:
        print "please enter a last value with -l (not inclusive)"
        sys.exit()
    draw_rollouts([i for i in range(first, last)], person)

# print max_distance(centers())