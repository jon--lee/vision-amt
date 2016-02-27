from options import AMTOptions

def merge_rollouts(lower, upper):
    merge_file = open(AMTOptions.data_dir + 'amt/net_deltas_mrg.txt', 'w')
    for r in range(lower, upper+1):
        rollout_file = open(get_rollout_file_name(r), 'r')
        for line in rollout_file:
            merge_file.write(line)
        rollout_file.close()

    merge_file.close()


def get_rollout_file_name(rollout):
    directory = AMTOptions.data_dir + 'amt/rollouts/rollout' + str(rollout) + '/'
    return directory + 'net_deltas.txt'

if __name__ == '__main__':
    merge_rollouts(100, 140)
