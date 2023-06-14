"""
Alignment reads and reference for one candidate
"""

import numpy as np 
import os 
import re

from tool.get_data_in_sam import compute_length, get_data_samfile, split_cigar


class Alignment:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, sam_file, candidate, reference_string):
        # tumor file 
        qname_t, flag_t, rname_t, pos_t, mapq_t, cigar_t, rnext_t, pnext_t, tlen_t, seq_t, qual_t, optional_field_t = get_data_samfile(sam_file)

        list_index_read = self.get_list_read(pos_t, cigar_t, candidate)
        qname = []
        pos = []
        cigar = []
        seq = []
        qual = []

        for _index in list_index_read:
            qname.append(qname_t[_index])
            pos.append(pos_t[_index])
            cigar.append(cigar_t[_index])
            seq.append(seq_t[_index]) 
            qual.append(qual_t[_index])

        # alignment v1
        alignment_tumor_read = [] # save ref and list of read, each seq contains A,C,T,G and -
        position_tumor_read = []  # save index of ref (0) and each read to align read position
        list_changes = [] # change for origin ref

        for i in range(len(pos)):
            if self.check_cigar(cigar[i]) == 0:    # only M
                alignment_tumor_read.append(seq[i])
                position_tumor_read.append(pos[i])
            
            else:  # I/S or D/H
                new_seq, changes = self.alignment_v1(seq[i], pos[i], cigar[i], qname[i])
                alignment_tumor_read.append(new_seq) #read
                list_changes.extend(changes) #list-changes has insert in every reads in 1 samfile
                # check if D/H in first cigar --> position decrease
                if self.check_cigar(cigar[i]) == 2:
                    cigar_inf = split_cigar(cigar[i])
                    position_tumor_read.append(pos[i]-cigar_inf[0][1])
                else:
                    position_tumor_read.append(pos[i])
        
        assert len(alignment_tumor_read) == len(position_tumor_read) == len(qname)
        sub_changes, references_changes = self.process_list_changes(list_changes)

        # alignment v2
        new_ref, new_alignment_reads, position_new = self.alignment_v2(alignment_tumor_read, position_tumor_read, (qname, cigar), sub_changes, references_changes, reference_string)
        
        # find new position for candidate after alignment
        new_candidate = self.find_new_candidate(references_changes, candidate)
        new_quality = self.get_new_quality(new_alignment_reads, qual)
        sub_ref, sub_reads, sub_quals = self.get_sub_ref_reads(new_ref, new_alignment_reads, position_new, new_candidate, new_quality)

        return sub_ref, sub_reads, sub_quals

    def get_sub_ref_reads(self, ref, seq, pos, candidate, qual):
        """Get sub ref and sub reads for input frequency matrix"""
        assert len(seq) == len(pos)
        ref = [ref[candidate-self.window_size-1:candidate+self.window_size]]

        reads = []
        quals = []
        for _index in range(len(seq)):
            if candidate-self.window_size >= pos[_index] and candidate+self.window_size <= pos[_index] + len(seq[_index])-1:
                index = candidate-pos[_index]
                reads.append(seq[_index][index-self.window_size:index+self.window_size+1])
                index = candidate-pos[_index]
                quals.append(qual[_index][index-self.window_size:index+self.window_size+1])

        return ref, reads, quals

    def get_new_quality(self, reads, quals):
        # get quality of read when read changed (maybe added '-')
        assert len(quals) == len(reads)

        new_quals = []
        for i in range(len(reads)):
            qual = quals[i]
            _qual = ''
            _j = 0
            for j in range(len(reads[i])):
                if reads[i][j] != '-':
                    _qual += qual[_j]
                    _j+=1
                else:
                    _qual += '*'
                    
            assert len(_qual) == len(reads[i])
            new_quals.append(_qual)

        return new_quals

    def find_new_candidate(self, changes, candidate):
        # find new candidate when position of read changed
        changes = self.sort_list(changes)
        add_position = 0
        for (old, new, name) in changes:
            if old <= candidate:
                add_position += new-old

        return candidate+add_position

    def alignment_v1(self, _seq, _pos, _cigar, _qname):
        """
        Alignment 1st time: browse on each read (both tumor and normal)
         - If CIGAR contains only M, append only the seq and pos of that read to alignment_reads and position_reads
         - If CIGAR contains D (Delete: present in ref but no read), that read is appended with "-" in the appropriate position, then append read after alignment and pos of that read to alignment_reads and position_reads ( Even if there is a change, the pos will not change at this step)
         - If CIGAR contains I, S (read has but ref does not), then there is a change on ref, there is no change on read, so still append seq and pos of that read to alignment_reads and position_reads, simultaneously append (index_old, index_new, qname) of that read to list_changes
        """
        current_change = []
        index = int(_pos)
        new_seq = _seq
        cigar_inf = split_cigar(_cigar)
        
        count_IS = 0
        for op, value in cigar_inf:
            if op in ['S','I']:
                current_change.append((index-count_IS, index+value-count_IS, _qname))
                count_IS += value
            elif op in ['D', 'H']:
                new_seq = self.add_undefine_in_str(new_seq, int(_pos), index, value) 
            
            index += value

        return new_seq, current_change

    def alignment_v2(self, alignment_reads, position_reads, read_info, sub_changes, references_changes, ref_string):
        """
        - Update position of reads following the change of Ref
        - Update Ref following list_changes
        - Update reads following sub_changes (sub_changes are the changes of reads with change that index_old overlaps and index_new < index_new_max)
        - UPdate reads following the change of Ref (when pos_read < index_old < pos_read + len_read – 1)
        """
        qname, cigar = read_info
        references_changes = self.sort_list(references_changes)
        
        # Update position of read
        add_position = np.zeros(len(position_reads))
        for (old, new, name) in references_changes:
            for i in range(len(position_reads)):
                if qname[i] not in name and old <= position_reads[i]:
                    add_position[i] += new-old

        position_new = []
        for i in range(len(position_reads)):
            position_new.append(int(position_reads[i]+add_position[i]))
        
        # Update references for all insert changes
        new_ref = ref_string
        remember = 0 
        for i in range(len(references_changes)):
            old, new, _ = references_changes[i]
            new_ref = self.add_undefine_in_str(new_ref, 1, int(old)+remember, int(new)-int(old)) #change ref according to read
            remember += int(new)-int(old)
            
        # change read with sub_changes
        # update reads theo sub changes (nhung read nao co insert nhung chua du do dai nhu trong ref thi them '-' den khi du)
        _alignment_reads = []
        for i in range(len(alignment_reads)): # voi moi read ktra xem no co sub changes k
            list_find_ind = self.find_read(qname[i], sub_changes)
            if len(list_find_ind)>0: # neu co sub changes
                remember_middle = 0 # nho so luong '-' da them trong read nay (alignment_reads[i])
                middle_alignment_read = alignment_reads[i]
                for j in range(len(list_find_ind)):
                    index_old, index_new, _ = sub_changes[list_find_ind[j]]
                    if index_old < int(position_reads[i]) + compute_length(cigar[i])-1:
                        middle_alignment_read = self.add_undefine_in_str(middle_alignment_read, int(position_reads[i]), int(index_old)+remember_middle, int(index_new)-int(index_old))
                        remember_middle += int(index_new)-int(index_old)
                _alignment_reads.append(middle_alignment_read)
            else:
                _alignment_reads.append(alignment_reads[i])
            
        assert len(qname) == len(_alignment_reads)

        # update nhung reads con lai k nam trong sub changes (sub changes chi moi luu lai nhung reads co insert nhung chua co do dai lon nhat)
        new_alignment_reads = []
        for i in range(len(_alignment_reads)): # vong for ngoai la lam voi moi read ne
            remember = 0 # van la nho so luong '-' da them trong read nay
            middle_read = _alignment_reads[i]
            for (old, new, name) in references_changes: # con vong for trong la lam voi moi ref change ne 
                remember_2 = 0 # nay kieu la nho do dai doan insert  cua nhung read chi co insert k co M
                if self.check_cigar(cigar[i]) == 1:
                    cigar_inf = split_cigar(cigar[i])
                    remember_middle = 0 
                    for op, value in cigar_inf:
                        if op in ['S','I'] and remember_middle+int(position_reads[i]) < old: # kiem tra xem dang truoc cai change cua ref co doan doc k co M k de luu lai remember_2, chac z
                            remember_2 += value
                        remember_middle+=value

                remember_3 = 0 
                list_find_ind = self.find_read(qname[i], sub_changes)
                if len(list_find_ind)>0:
                    remember_middle = 0
                    for j in range(len(list_find_ind)):
                        index_old, index_new, _ = sub_changes[list_find_ind[j]]
                        if index_old +remember_middle < old: 
                            remember_3 += int(index_new)-int(index_old) 
                        remember_middle += int(index_new)-int(index_old)

                if qname[i] not in name and position_reads[i] < old+remember+remember_2+remember_3 < position_reads[i] + compute_length(cigar[i]) + remember + remember_3 - 1: # kiem tra xem ref change nay co nam giua vi tri bat dau voi vi tri ket thuc cua read k (neu no nam giua thi read nay phai update theo no)
                    middle_read = self.add_undefine_in_str(middle_read, int(position_reads[i]), int(old)+remember+remember_2+remember_3, int(new)-int(old))
                    remember += new - old
    
            new_alignment_reads.append(middle_read)
            
        return new_ref, new_alignment_reads, position_new    

    def get_list_read(self, list_pos, list_cigar, candidate):
        """Get list index of read, which contain candidate and allow window_size"""
        list_index = []
        for i in range(len(list_pos)):
            _pos = list_pos[i]
            _cigar = list_cigar[i]
            if _pos <= candidate-self.window_size and candidate+self.window_size <= _pos + compute_length(_cigar)-1:
                list_index.append(i)
        
        return list_index

    def check_cigar(self, cigar_string):
        """ 
        Just care D and S/I, not care M --> return 1
        if CIGAR only contain M --> return 0
        """
        process_txt =  re.sub(r"[0-9]", "", cigar_string)

        if process_txt[0] in ['D', 'H']:    # if cigar has D or H in first --> position of read decrease
            return 2
        
        if 'M' not in process_txt:
            return 1
        
        return 0

    def sort_list(self, list_triplet):
        """ 
        Sort list changes with index_old
        """
        for i in range(len(list_triplet)):
            minimum = i
            
            for j in range(i + 1, len(list_triplet)):
                # Select the smallest value
                if list_triplet[j][0] < list_triplet[minimum][0]:
                    minimum = j

            # Place it at the front of the 
            # sorted end of the array
            list_triplet[minimum], list_triplet[i] = list_triplet[i], list_triplet[minimum]
            
        return list_triplet


    def add_undefine_in_str(self, _seq, _pos, position_change, value):
        """
        Add '-' to string (Ref or read) with relative position
        """
        add_str = '-' * value
        index = position_change - _pos
        if index == 0:
            return add_str + _seq
        else:
            return _seq[:index] + add_str + _seq[index:]

    def process_same_old_changes(self, _list):
        sub_elements = []
        max_index_new = 0
        for i in range(len(_list)):
            max_index_new = max(max_index_new, _list[i][1])
            
        for i in range(len(_list)):
            if _list[i][1] < max_index_new:
                sub_elements.append((_list[i][1], max_index_new, _list[i][2]))

        
        list_qname = []
        for i in range(len(_list)):
            list_qname.append(_list[i][2])
            
        return sub_elements, (_list[i][0], max_index_new, list_qname)


    def process_list_changes(self, list_changes):
        """
        Handle cases with the same index_old, get the widest range (ie the largest index_new), 
        return the largest area, are also offset regions so that the smallest region is equal to the largest region
        """
        check = [0]*len(list_changes)
        
        sub_elements = []   # save list of (index_old, index_new, seq)
        references_changes = []
        for i in range(len(list_changes)):
            if check[i] == 0:
                check[i] = 1
                middle = [list_changes[i]]
                for j in range(i+1, len(list_changes)):
                    if list_changes[i][0] == list_changes[j][0]:
                        middle.append(list_changes[j])
                        check[j] = 1
            
                if len(middle) > 1:
                    x, y = self.process_same_old_changes(middle)
                    sub_elements.extend(x)
                    references_changes.append(y)

                else:
                    references_changes.append((list_changes[i][0], list_changes[i][1], [list_changes[i][2]]))
        return sub_elements, references_changes

    def find_read(self, _sub, _list):
        """
        Find read in a list changes to edit this read
        Only take change that is mentioned in qname
        """
        list_find = [] # beacause maybe has > 1 change (S/I) ind 1 read
        for i in range(len(_list)):
            _,_,name = _list[i]
            if _sub == name:
                list_find.append(i)  
        return list_find         
            