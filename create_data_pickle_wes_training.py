"""
Get input matrix and label for each chr (many candidate) and save them with pickle
"""

import sys 
import os
import pickle
import multiprocessing as mp
import time

from tool.get_input_matrix_label_wes import FrequencyInputMatrix
from tool.get_reference_string import ReferenceString

from config import config_param
        
_path_data = './data/candidate_sam/{}' #this contain the high confidence result in window of file sam created by a specific lab
_save_dir = './data/pickle/{}_ws{}_pickle/'

window_size = int(sys.argv[1])
lab_name = sys.argv[2]

save_dir = _save_dir.format(lab_name, window_size)
path_data = _path_data.format(lab_name)

get_matrix = FrequencyInputMatrix(window_size=window_size, training=True)
get_ref = ReferenceString(config_param.reference_file)
all_ref = get_ref.get_22_chromosome() #Grch38


def run(list_chr):
    for chr_dir in list_chr:
        chr_name = os.path.basename(chr_dir)
        ref = all_ref[chr_name]

        path_save = os.path.join(save_dir, chr_name)
        os.makedirs(path_save, exist_ok=True)
        
        input_matrix, output_label = get_matrix(chr_dir, ref)
        print(input_matrix.shape, output_label.shape)

        # save pickle
        with open(os.path.join(path_save, 'input_matrix.pickle'), 'wb') as fo:
            pickle.dump(input_matrix, fo)
        with open(os.path.join(path_save, 'output_label.pickle'), 'wb') as fo:
            pickle.dump(output_label, fo)


def main():
    list_chr = []
    for i in range(22):
        list_chr.append(os.path.join(path_data, 'chr'+str(i+1)))
    
    run(list_chr)
    

main()


# nohup python3 create_data_pickle_wes_training.py 16 &