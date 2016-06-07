from options import AMTOptions
import os



if __name__ == '__main__':
    turkers_path = AMTOptions.data_dir + "amt/turkers/"
    lines = []
    for filename in os.listdir(turkers_path):
        if '_labels.txt' in filename:
            turker_file = open(turkers_path + filename, 'r')
            print filename
            for line in turker_file:
                if not 'rollout145' in line:
                    lines.append(line)

    
    turkers_filepath = AMTOptions.data_dir + 'amt/turkers.txt'
    turkers_file = open(turkers_filepath, 'w')
    for line in lines:
        turkers_file.write(line)   
