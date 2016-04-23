"""
Class variables are typical constants (do not change externally)
Instance variables are options (ok to change case by case externally)

This is modeled after leveldb's options
"""
import os
class Options():
    translations = [0.0, 0.0, 0.0, 0.0]
    scales = [40.0, 120.0, 90.0, 100.0]
    drift = 20.0
    OFFSET_X = 190
    OFFSET_Y = 105
    WIDTH = 260
    HEIGHT = 260
    
    #root_dir = '/Users/JonathanLee/Desktop/sandbox/vision/'
    root_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
    data_dir = root_dir + 'data/'
    datasets_dir = data_dir + 'datasets/'
    frames_dir = data_dir + 'record_frames/'
    videos_dir = data_dir + 'record_videos/'
    hdf_dir = root_dir + 'Net/hdf/'
    tf_dir = root_dir + 'Net/tensor/'
    images_dir = data_dir + 'images/'
    amt_dir = data_dir + 'amt/'
    record_amt_dir = amt_dir + "record_amt/"
    nets_dir = '/media/1tb/Izzy/nets/'

    def __init__(self):
        self.test = False
        self.deploy = False
        self.learn = False
        self.model_path = ""        # path to network architecture prototxt
        self.weights_path = ""      # path to weights (should match model)
        self.show = False           # whether to show a preview window from bincam
        self.record = False         # whether to record frames with bincam
        self.scales = Options.scales
        self.translations = Options.translations
        self.drift = Options.drift
        self.tf_net = None          # tensorflow model
        self.tf_net_path = ""       # path to tensorflow model's weights


class AMTOptions(Options):
    
    train_file = Options.amt_dir + "train.txt"
    test_file = Options.amt_dir + "test.txt"
    deltas_file = Options.amt_dir + "deltas.txt"
    rollouts_file = Options.amt_dir + "rollouts.txt"

    rollouts_dir = Options.amt_dir + "rollouts/"
    binaries_dir = Options.amt_dir + "binaries/"
    originals_dir = Options.amt_dir + "frames/"
    grayscales_dir = Options.amt_dir + "grayscales/"
    colors_dir = Options.amt_dir + "colors/"
    policies_dir = Options.amt_dir + "policies/"

    ROTATE_UPPER_BOUND = 3.82
    ROTATE_LOWER_BOUND = 3.06954
    
    GRIP_UPPER_BOUND = .06
    GRIP_LOWER_BOUND = .0023
    
    TABLE_LOWER_BOUND = .002
    TABLE_UPPER_BOUND = 7.0    
