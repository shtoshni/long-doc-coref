from itertools import product
from os import path

import os
import subprocess


mem_size_list = [''] + ['-mem_size ' + str(mem_size) for mem_size in [1000]]
mlp_size_list = ['-mlp_size ' + str(mlp_size) for mlp_size in [1000, 2000, 3000]]
dropout_list = ['-dropout_rate ' + str(dropout_rate) for dropout_rate in [0.3]]
num_cell_list = ['-num_cells ' + str(num_cells) for num_cells in [10, 20, 50]]
model_loc_list = ['-model_loc /share/data/speech/shtoshni/resources']
max_segment_list = ['-max_segment_len ' + str(max_segment_len) for max_segment_len in [512]]
ment_emb_list = ['-ment_emb ' + str(ment_emb) for ment_emb in ['endpoint']]
entity_rep_list = ['-entity_rep ' + str(entity_rep) for entity_rep in ['avg']]
lr_list = ['-init_lr ' + str(lr) for lr in [5e-4]]
seed = ['-seed 0']

JOB_NAME = 'auto_mem'

out_dir = path.join(os.getcwd(), 'outputs/mem_size')
if not path.isdir(out_dir):
    os.makedirs(out_dir)

# Write commands to output file
out_file = path.join(out_dir, 'commands.txt')
base_dir = "/share/data/speech/shtoshni/research/litbank_coref"
with open(out_file, 'w') as out_f:
    for option_comb in product(mem_size_list, mlp_size_list, dropout_list, num_cell_list,
                               model_loc_list, max_segment_list, lr_list,
                               ment_emb_list, entity_rep_list, seed):
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
