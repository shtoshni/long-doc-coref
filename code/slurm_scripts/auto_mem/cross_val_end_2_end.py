from itertools import product
from os import path
import os
import subprocess

mem_type_list = ['-mem_type ' + mem_type for mem_type in ['unbounded', 'fixed_mem', 'lru']]
doc_enc_list = ['-doc_enc ' + mem_type for mem_type in ['overlap']]
mlp_size_list = ['-mlp_size ' + str(mlp_size) for mlp_size in [3000]]
mlp_depth_list = ['-mlp_depth ' + str(mlp_depth) for mlp_depth in [1]]
coref_mlp_depth_list = ['-coref_mlp_depth ' + str(mlp_depth) for mlp_depth in [1]]

## CHANGE THIS!!
dropout_list = ['-dropout_rate ' + str(dropout_rate) for dropout_rate in [0.3]]
ignore_wt_list = ['-ignore_wt ' + str(ignore_wt) for ignore_wt in [0.1, 0.2, 1.0]]

model_loc_list = ['-pretrained_bert_dir /share/data/speech/shtoshni/resources']
max_segment_list = ['-max_segment_len ' + str(max_segment_len) for max_segment_len in [512]]
lr_list = ['-init_lr ' + str(lr) for lr in [5e-4]]
seed = ['-seed 10']
cross_val_split_list = ['-cross_val_split ' + str(cross_val_split) for cross_val_split in range(10)]

num_cell_list = ['-num_cells {} '.format(x) for (x) in [5, 10, 25, 50]]
train_span_model_list = ['-train_span_model -top_span_ratio 0.3 ']
over_loss_wt_list = ['-over_loss_wt ' + str(over_loss_wt) for over_loss_wt in [1.0]]
new_ent_wt_list = ['-new_ent_wt ' + str(new_ent_wt) for new_ent_wt in [1.0]]


JOB_NAME = 'cross_val_litbank_base'

out_dir = path.join(os.getcwd(), 'outputs/cross_val_litbank_base')
if not path.isdir(out_dir):
    os.makedirs(out_dir)

# Write commands to output file
out_file = path.join(out_dir, 'commands.txt')
base_dir = "/share/data/speech/shtoshni/research/litbank_coref"

common_options = [doc_enc_list, mlp_size_list, mlp_depth_list, coref_mlp_depth_list,
                  dropout_list, model_loc_list, train_span_model_list,
                  max_segment_list, lr_list, seed, cross_val_split_list,
                  over_loss_wt_list, ignore_wt_list, new_ent_wt_list]
# print(common_options)
# print(mem_type_list)
fixed_mem_options = [num_cell_list]
with open(out_file, 'w') as out_f:
    for mem_type in mem_type_list:
        # print(mem_type)
        for option_comb in product(*common_options):
            # print(option_comb)
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

            # print(cur_command)

subprocess.call(
    "cd {}; python ~/slurm_batch.py {} -J {}".format(out_dir, out_file, JOB_NAME), shell=True)
