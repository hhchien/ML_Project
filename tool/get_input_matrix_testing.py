
from tool.alignment_read_v2 import Alignment
from config import config_param
from tool.get_reference_string import ReferenceString
import numpy as np 
from tool.get_data_in_sam import get_data_samfile, compute_length, split_cigar, find_md_field, split_md_field


class FrequencyInputMatrix:
    def __init__(self, window_size):
        self.window_size = window_size
        self.alignment = Alignment(self.window_size)
        self.type_label = config_param.type_label
        self.all_ref = ReferenceString(config_param.reference_file).get_22_chromosome()

    def __call__(self, sam_tumor_file, candidate, ref):
        sub_ref, sub_reads, sub_quals = self.alignment(sam_tumor_file, candidate, ref)
        if len(sub_reads) > 0:
            list_matrix = self.frequency_matrix(sub_ref, sub_reads, sub_quals)

            return np.array(list_matrix)

    def frequency_matrix(self, sub_ref, sub_reads, sub_quals):
        """
        Sub_refs using for tumor and normal sub read, which contain 'N' in sequence
        """
        # matrix ref
        characters = ['-', 'A', 'C', 'G', 'T']
        _ref = sub_ref[0]
        matrix_ref_return = np.zeros((len(characters), len(_ref)))
        
        for i in range(len(_ref)):
            if _ref[i] in characters:
                matrix_ref_return[characters.index(_ref[i])][i]+=1

        # matrix reads
        matrix_read_return =  np.zeros((len(characters), len(_ref)))
        for read in sub_reads:
            for j in range(len(read)):
                if read[j] in characters:
                    matrix_read_return[characters.index(read[j])][j]+=1 

        matrix_qual_return =  np.zeros((len(characters), len(sub_reads[0])))
        for i in range(len(sub_reads)):
            read = sub_reads[i]
            for j in range(len(read)):
                if read[j] in characters and sub_quals[i][j] != '*':
                    matrix_qual_return[characters.index(read[j])][j]+= self.compute_qual(sub_quals[i][j])

        return self.create_input_matrix(matrix_ref_return, matrix_read_return, matrix_qual_return)

    def create_input_matrix(self, matrix_a, matrix_b, matrix_c):
        matrix_3d_return = []
        shape1, shape2 = matrix_a.shape
        assert matrix_a.shape == matrix_b.shape
        for i in range(len(matrix_a)):
            matrix_3d_return.append(np.dstack([matrix_a[i], matrix_b[i], matrix_c[i]]))

        return np.array(matrix_3d_return).reshape((shape1, shape2, 3)).astype(np.float32)

    def compute_qual(self, _string):
        """
        probability that the base read from the sequencer is faulty (tinhs xác suất base đọc được từ máy giải trình tự bị lỗi)
        """
        result = 0
        for char in _string:
            q = ord(char)
            result += np.power(10, -1 * (q-33)/10)
        return result/len(_string)  