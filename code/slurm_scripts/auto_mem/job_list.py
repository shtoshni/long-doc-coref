from itertools import product
from os import path

import os
import subprocess

mem_type_list = ['-mem_type ' + mem_type for mem_type in ['lru', 'fixed_mem', 'unbounded']]
cross_val_split_list = ['-cross_val_split ' + str(cross_val_split) for cross_val_split in range(1)]
mlp_size_list = ['-mlp_size ' + str(mlp_size) for mlp_size in [1024]]
mlp_depth_list = ['-mlp_depth ' + str(mlp_depth) for mlp_depth in [1]]
dropout_list = ['-dropout_rate ' + str(dropout_rate) for dropout_rate in [0.5]]
model_loc_list = ['-model_loc /share/data/speech/shtoshni/resources']
max_segment_list = ['-max_segment_len ' + str(max_segment_len) for max_segment_len in [512]]
lr_list = ['-init_lr ' + str(lr) for lr in [5e-4]]
seed = ['-seed 1']


num_cell_list = ['-num_cells ' + str(num_cells) for num_cells in [5, 10, 20, 50]]
over_loss_wt_list = ['-over_loss_wt ' + str(over_loss_wt) for over_loss_wt in [0.1, 1.0]]

JOB_NAME = 'auto_mem'

out_dir = path.join(os.getcwd(), 'outputs/mem_type')
if not path.isdir(out_dir):
    os.makedirs(out_dir)

# Write commands to output file
out_file = path.join(out_dir, 'commands.txt')
base_dir = "/share/data/speech/shtoshni/research/litbank_coref"

common_options = [mlp_size_list, mlp_depth_list, dropout_list, model_loc_list,
                  max_segment_list, lr_list, seed, cross_val_split_list]
fixed_mem_options = [num_cell_list, over_loss_wt_list]
with open(out_file, 'w') as out_f:
    for mem_type in mem_type_list:
        for option_comb in product(*common_options):
            base = '{}/code/slurm_scripts/auto_mem/run.sh'.format(base_dir)
            base += ' -base_model_dir {}/models '.format(base_dir)
            base += ' -base_data_dir {}/data '.format(base_dir)
            cur_command = base
            for value in option_comb:
                cur_command += (value + " ")

            cur_command += mem_type + ' '
            # cur_command = cur_command.strip()

            if 'unbounded' in mem_type:
                out_f.write(cur_command + '\n')
            else:
                for fixed_mem_comb in product(*fixed_mem_options):
                    cur_base_command = str(cur_command)
                    for value in fixed_mem_comb:
                        cur_base_command += value + ' '
                    out_f.write(cur_base_command + '\n')

subprocess.call(
    "cd {}; python ~/slurm_batch.py {} -J {}".format(out_dir, out_file, JOB_NAME), shell=True)
