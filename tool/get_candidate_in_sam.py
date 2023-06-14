"""
Get list candidate in sam file
"""

import numpy as np
import re 
from tool.get_data_in_sam import get_data_samfile, compute_length, split_cigar, find_md_field, split_md_field


def compute_qual(_string):
    """
    probability that the base read from the sequencer is faulty (tinhs xác suất base đọc được từ máy giải trình tự bị lỗi)
    """
    result = 0
    for char in _string:
        q = ord(char)
        result += np.power(10, -1 * (q-33)/10)
    return result/len(_string)  


class Candidate:
    def __init__(self, chr_split=0):
        self.chr_split = chr_split
        pass

    def __call__(self, sam_tumor_file):
        qname, flag, rname, pos, mapq, cigar, rnext, pnext, tlen, seq, qual, optional_field = get_data_samfile(sam_tumor_file)
    
        if(self.chr_split): 
            list_candidate = [[] for i in range(23)]
        else: 
            list_candidate = [] #list chứa tất cả các vị trí đã tìm thấy đột biến trong tất cả các đoạn đọc   
        if len(qname) > 0:
            # using a list to save which position is used  
            position_start = int(pos[0])
            position_end = max(int(pos[i])+compute_length(cigar[i])-1 for i in range(len(pos)))
            visited = np.zeros(position_end - position_start + 1)

            for i in range(len(cigar)): #means every candidate in a sam file
                # using Optional field MD for mismatch 
                # using CIGAR for insert, delete 
                if self.chr_split:
                    chr_num = re.sub("[a-zA-Z]", "", rname[i])
                    if not chr_num.isdigit(): continue
                    else: idx = int(chr_num)
                
                cigar_infor = split_cigar(cigar[i])
                count = 0
                count_m = 0
                count_d = 0
                for op, value in cigar_infor:
                    if op =='D' or op =='H':
                        if not visited[int(pos[i])+count-position_start] and op =='D': 
                            if(self.chr_split):
                                list_candidate[idx].append(int(pos[i])+count-1)
                            else:
                                list_candidate.append(int(pos[i])+count-1)
                            visited[int(pos[i])+count-position_start] = 1
                                
                        count -= value
                        count_d += value
                    elif op == 'I' or op =='S':
                        if not visited[int(pos[i])+count+count_d-position_start] and op == 'I':
                            if(self.chr_split):
                                list_candidate[idx].append(int(pos[i])+count+count_d-1)
                            else:    
                                list_candidate.append(int(pos[i])+count+count_d-1)
                            visited[int(pos[i])+count+count_d-position_start] = 1
                            
                    else:   # M 
                        _option_field = optional_field[i]
                        md_field = find_md_field(_option_field)
                        split_data = split_md_field(md_field, count_m, value)
                        count_m += value 
                        count_middle = 0
                        for each_data in split_data:
                            if each_data.isdigit():
                                count_middle += int(each_data)
                            else:
                                if not visited[int(pos[i])+count_middle+count-position_start]:
                                    if(self.chr_split):
                                        list_candidate[idx].append(int(pos[i])+count_middle+count)
                                    else:
                                        list_candidate.append(int(pos[i])+count_middle+count)
                                    visited[int(pos[i])+count_middle+count-position_start] = 1

                                count_middle +=1
                    count += value
        
        return list_candidate
