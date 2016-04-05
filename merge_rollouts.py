from options import AMTOptions

def merge_rollouts(lower, upper):
    merge_file = open(AMTOptions.data_dir + 'amt/net_deltas_mrg.txt', 'w')
    for r in range(lower, upper+1):
        rollout_file = open(get_rollout_file_name(r), 'r')
        for line in rollout_file:
            merge_file.write(line)
        rollout_file.close()

    merge_file.close()

def merge_rollouts_deltas():
    merge_file = open(AMTOptions.data_dir + 'amt/expert_net.txt', 'w')
    deltas_file = open(AMTOptions.data_dir + 'amt/labels_amt_me_1108_1127.txt', 'r')

    expert_at = deltas_file.readline()
    r = find_rollout_no(expert_at)
    exp_frame = find_frame_no(expert_at)
    while expert_at:
        rollout_file = open(get_rollout_file_name(r), 'r')
        # i is a dummy variable
        for i in range(exp_frame):
            merge_file.write(rollout_file.readline())
        lastr = r
        while lastr == r:
            merge_file.write(expert_at)
            expert_at = deltas_file.readline()
            if not expert_at:
                return
            r = find_rollout_no(expert_at)
            exp_frame = find_frame_no(expert_at)
        rollout_file.close()
    merge_file.close()

def find_rollout_no(expert_at):
    print expert_at[expert_at.find('rollout') + 7:expert_at.find('_')]
    return int(expert_at[expert_at.find('rollout') + 7:expert_at.find('_')])

def find_frame_no(expert_at):
    return int(expert_at[expert_at.find('frame_') + 6:expert_at.find('.jpg')])


def get_rollout_file_name(rollout):
    directory = AMTOptions.data_dir + 'amt/rollouts/rollout' + str(rollout) + '/'
    return directory + 'net_deltas.txt'

if __name__ == '__main__':
    # merges the current deltas.txt (filled in after running a rollout) with the rollout deltas
    merge_rollouts_deltas()
