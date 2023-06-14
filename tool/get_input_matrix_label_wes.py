"""
Using for wes data, 50 gen using for testing, not training
"""

import numpy as np
import os
import random

from tool.alignment_read_v2 import Alignment
from tool.get_candidate_in_sam import Candidate
from tool.get_data_in_sam import compute_length, get_data_samfile
from config import config_param
from tool.get_reference_string import ReferenceString


class FrequencyInputMatrix:
    def __init__(self, window_size, training):
        self.window_size = window_size
        self.alignment = Alignment(self.window_size)
        self.get_candidate = Candidate()
        self.type_label = config_param.type_label
        self.all_ref = ReferenceString(config_param.reference_file).get_22_chromosome()
        self.gen_infor = config_param.gen_info_50
        self.fortrain = training

    def __call__(self, chr_dir, ref):
        #generates input matrices and labels for each candidate position
        list_matrix = []    #chứa ma trận đầu vào
        list_label = []     #chứa nhãn

        list_candidate_sam_file = [os.path.join(chr_dir, f) for f in os.listdir(chr_dir)]   #các file sam của các candidate tạo thành 1 list
        chromo = int(os.path.basename(chr_dir)[3:]) #số thứ tự của nst hiện tại

        for candidate_sam_file in list_candidate_sam_file:
            # =========================create positive data=================================================
            candidate = int(os.path.basename(candidate_sam_file).split('_')[0]) #lấy vị trí của candidate theo truth đang xét
            if self.check_candidate_50gen(chromo, candidate, self.fortrain): #kiểm tra xem có trong 50gen để test không
                label = self.type_label[os.path.basename(candidate_sam_file).split('_')[1].split('.')[0]]   #label theo truth

                list_candidate = self.get_candidate(candidate_sam_file) #list chứa tất cả các vị trí đã tìm thấy đột biến trong tất cả các đoạn đọc trong 1 file sam của 1 candidate theo truth
                if candidate in list_candidate:
                    sub_ref, sub_reads, sub_quals = self.alignment(candidate_sam_file, candidate, ref)
                    if len(sub_reads) > 0:
                        list_matrix.append(self.frequency_quality_matrix(sub_ref, sub_reads, sub_quals))
                        list_label.append(self.one_hot_vector(len(self.type_label), label))

                # =========================create negative data==================================================
                candidate = int(os.path.basename(candidate_sam_file).split('_')[0]) #can be deleted
                nu_candidate_ref = ref[candidate-1]
                label = self.type_label['non-somatic'] #set all đột biến found in sam file to be non-somatic

                # for training, chose 1 negative candidate, which random
                if self.fortrain:
                    negative_candidate = self.get_random_negative_candidate(candidate_sam_file, candidate)
                    
                    if negative_candidate != 0:
                        sub_ref, sub_reads, sub_quals = self.alignment(candidate_sam_file, negative_candidate, ref)
                        if len(sub_reads) > 0:
                            list_matrix.append(self.frequency_quality_matrix(sub_ref, sub_reads, sub_quals))
                            list_label.append(self.one_hot_vector(len(self.type_label), label))

                # for testing, get 2 random negatives
                else:
                    negative_candidates = self.get_candidate(candidate_sam_file)
                    if candidate in negative_candidates:
                        negative_candidates.remove(candidate)
                    if len(negative_candidates) > 0:
                        if len(negative_candidates) == 1:
                            sub_ref, sub_reads, sub_quals = self.alignment(candidate_sam_file, negative_candidates[0], ref)
                            if len(sub_reads) > 0:
                                list_matrix.append(self.frequency_quality_matrix(sub_ref, sub_reads, sub_quals))
                                list_label.append(self.one_hot_vector(len(self.type_label), label))
                        else:
                            random.shuffle(negative_candidates)
                            for negative_candidate in negative_candidates[:2]:
                                sub_ref, sub_reads, sub_quals = self.alignment(candidate_sam_file, negative_candidate, ref)
                                if len(sub_reads) > 0:
                                    list_matrix.append(self.frequency_quality_matrix(sub_ref, sub_reads, sub_quals))
                                    list_label.append(self.one_hot_vector(len(self.type_label), label))

        assert len(list_matrix)==len(list_label)
        return np.array(list_matrix), np.array(list_label)

    def check_candidate_50gen(self, chromo, candidate, is_training):
        """Using data 50 gen wes for testing, not training
        """
        if is_training:
            for gen in self.gen_infor:
                if self.gen_infor[gen][0] == chromo:
                    if self.gen_infor[gen][1] <= candidate <= self.gen_infor[gen][2]:
                        return False
            return True

        else:
            for gen in self.gen_infor:
                if self.gen_infor[gen][0] == chromo:
                    if self.gen_infor[gen][1] <= candidate <= self.gen_infor[gen][2]:
                        return True
            return False

    def get_random_negative_candidate(self, candidate_sam_file, positive_candidate):
        """
        Get random candidate to non-somatic positive
        """
        list_candidate = self.get_candidate(candidate_sam_file)
        if positive_candidate in list_candidate:
            list_candidate.remove(positive_candidate)

        if len(list_candidate) == 0:
            return 0
        if len(list_candidate) == 1:
            return list_candidate[0]
        
        list_index = np.arange(len(list_candidate))
        random.shuffle(list_index)
        return list_candidate[list_index[0]]
        
    def one_hot_vector(self, length, position):
        """
        One hot vector for label output model
        """
        vector = np.zeros(length)
        vector[position]=1
        return vector

    def frequency_quality_matrix(self, sub_ref, sub_reads, sub_quals):
        """
        Sub_refs using for tumor and normal sub read, which contain 'N' in sequence
        """

        # matrix ref
        characters = ['-', 'A', 'C', 'G', 'T']
        _ref = sub_ref[0]
        matrix_ref_return = np.zeros((len(characters), len(_ref)))
        #day la 3 ma tran dau vao: ma tran tham chieu (vi tri cua cac base tham chieu), ma tran tần suất xuất hiện của các base trên từng vị trí, ma trận chất lượng chứa chất lượng các cơ sở thuộc các đoạn đọc trên cùng từng vị trí
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
        """
        Merge matrix reference, matrix tumor, matrix quality to one matrix 3 dimentions
        """
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