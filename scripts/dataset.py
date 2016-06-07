import datetime
import os
import cv2

class Dataset():

    def __init__(self, name, path, options):
        self.name = name
        self.options = options
        self.path = options.datasets_dir + self.name + "/"
        self.writer = open(self.path + "controls.txt", 'a')
        self.state_writer = open(self.path + "states.txt", 'a')
        self.i = Dataset.next_datum_index(self.path)
        self.staged = []

    def stage(self, frame, controls, state=None):
        self.staged.append((frame, controls, state))

    def commit(self):
        print "Committing " + len(self.staged) + " staged elements..."
        for stage in self.staged:
            frame, controls, state = stage
            self.put(frame, controls, state=state)
        self.staged = []
        print "Done writing to dataset"

    def put(self, frame, controls, state=None):
        filename = "img_" + str(self.i) + ".jpg"
        filepath = self.path + filename
        if os.path.isfile(filepath):
            self.i += 1
            self.put(frame, controls)
        else:
            controls_str = Dataset.controls2str(controls)
            self.writer.write(filename + controls_str + '\n')
            if state:
                state_str = Dataset.controls2str(state)            
                self.state_writer.write(filename + state_str + '\n')
            cv2.imwrite(self.path + filename, frame)
            self.i += 1

    def get(self, filename):
        """ return cv2 image with filename and list of controls """
        filepath = self.path + filename
        if not os.path.isfile(filepath):
            raise Exception("File not found at " + filepath)
        return cv2.imread(filepath)
        


    @staticmethod
    def next_datum_index(path, i=0):
        filepath = path + "img_" + str(i) + ".jpg"
        while os.path.isfile(filepath):
            i += 1
            filepath = path + "img_" + str(i) + ".jpg"
        return i

    @staticmethod
    def create_ds(options, prefix=""):
        name = prefix + "_" + datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss") + "/"
        path = options.datasets_dir + name
        os.makedirs(path)
        ds = Dataset(name, path, options)
        
        with open('./meta.txt', 'r') as f_in:
            lines = f_in.readlines()
            with open(ds.path + "meta.txt", 'a+') as f_out:
                f_out.writelines(lines)

        about = open(ds.path + "about.txt", 'a+')
        about.write(datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S'))
        about.write("\n\nnet: " + options.model_path + "\nweights: " + options.weights_path)

        return ds

    @staticmethod
    def get_ds(options, name):
        path = options.datasets_dir + name
        ds = Dataset(name, path, options)
        if not os.path.exists(ds.path):
            raise Exception("No dataset found at " + path)
        return ds

    @staticmethod
    def controls2str(controls):
        """ Returns space separated controls with space preceding all controls """
        controls_str = ""
        for c in controls:
            controls_str += " " + str(c)
        return controls_str

