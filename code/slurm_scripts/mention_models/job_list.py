from itertools import product
from os import path

import os
import subprocess

cross_val_split_list = ['-cross_val_split ' + str(cross_val_split) for cross_val_split in range(10)]
mlp_size_list = ['-mlp_size ' + str(mlp_size) for mlp_size in [1000, 3000]]
mlp_depth_list = ['-mlp_depth ' + str(mlp_depth) for mlp_depth in [1]]
doc_enc_list = ['-doc_enc ' + mem_type for mem_type in ['overlap']]
dropout_list = ['-dropout_rate ' + str(dropout_rate) for dropout_rate in [0.3, 0.5]]
pretrained_bert_list = ['-pretrained_bert_dir /share/data/speech/shtoshni/resources']
model_size_list = ['-model_size ' + model_size for model_size in ['base', 'large']]
max_span_width_list = ['-max_span_width ' + str(max_span_width) for max_span_width in [20, 30]]
max_segment_list = ['-max_segment_len ' + str(max_segment_len) for max_segment_len in [512]]
max_epochs_list = ['-max_epochs 35']
lr_list = ['-init_lr ' + str(lr) for lr in [5e-4]]
seed = ['-seed 0']

JOB_NAME = 'litbank_ment'

out_dir = path.join(os.getcwd(), 'outputs/litbank')
if not path.isdir(out_dir):
    os.makedirs(out_dir)

# Write commands to output file
out_file = path.join(out_dir, 'commands.txt')
base_dir = "/share/data/speech/shtoshni/research/litbank_coref"

common_options = [cross_val_split_list, mlp_size_list, mlp_depth_list, doc_enc_list,
                  dropout_list, pretrained_bert_list, model_size_list,
                  max_span_width_list, max_segment_list, max_epochs_list, lr_list, seed]
with open(out_file, 'w') as out_f:
    for option_comb in product(*common_options):
        base = '{}/code/slurm_scripts/mention_models/run.sh'.format(base_dir)
        base += ' -base_model_dir {}/models '.format(base_dir)
        base += ' -base_data_dir {}/data '.format(base_dir)
        cur_command = base
        for value in option_comb:
            cur_command += (value + " ")

        cur_command = cur_command.strip()
        out_f.write(cur_command + '\n')

subprocess.call(
    "cd {}; python ~/slurm_batch.py {} -J {}".format(out_dir, out_file, JOB_NAME), shell=True)
