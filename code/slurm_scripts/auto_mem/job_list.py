from itertools import product
from os import path

import os
import subprocess

mlp_size_list = ['-mlp_size ' + str(mlp_size) for mlp_size in [1000]]
mlp_depth_list = ['-mlp_depth ' + str(mlp_depth) for mlp_depth in [1]]
dropout_list = ['-dropout_rate ' + str(dropout_rate) for dropout_rate in [0.5]]
num_cell_list = ['-num_cells ' + str(num_cells) for num_cells in [5, 10, 20, 50]]
model_loc_list = ['-model_loc /share/data/speech/shtoshni/resources']
max_segment_list = ['-max_segment_len ' + str(max_segment_len) for max_segment_len in [512]]
mem_type_list = ['-mem_type ' + str(mem_type) for mem_type in ['lru', 'fixed_mem']]
over_loss_wt_list = ['-over_loss_wt ' + str(over_loss_wt) for over_loss_wt in [0.01, 0.1, 1.0]]
lr_list = ['-init_lr ' + str(lr) for lr in [5e-4]]
seed = ['-seed 1']

JOB_NAME = 'auto_mem'

out_dir = path.join(os.getcwd(), 'outputs/mem_type')
if not path.isdir(out_dir):
    os.makedirs(out_dir)

# Write commands to output file
out_file = path.join(out_dir, 'commands.txt')
base_dir = "/share/data/speech/shtoshni/research/litbank_coref"
with open(out_file, 'w') as out_f:
    for option_comb in product(mlp_depth_list, mlp_size_list, dropout_list, num_cell_list,
                               model_loc_list, max_segment_list, mem_type_list,
                               over_loss_wt_list, lr_list, seed):
        base = '{}/code/slurm_scripts/auto_mem/run.sh'.format(base_dir)
        base += ' -base_model_dir {}/models '.format(base_dir)
        base += ' -base_data_dir {}/data '.format(base_dir)
        cur_command = base
        for value in option_comb:
            cur_command += value + ' '

        cur_command = cur_command.strip()
        out_f.write(cur_command + '\n')

subprocess.call(
    "cd {}; python ~/slurm_batch.py {} -J {}".format(out_dir, out_file, JOB_NAME), shell=True)
