import re
import numpy as np


def get_data_samfile(sam_file):
    """
    field_data = ['qname', 'flag', 'rname', 'pos', 'mapq', 'cigar', 'rnext', 'pnext', 'tlen', 'seq', 'qual', 'optional_filed']
    """

    f = open(sam_file, 'r')
    all_line = f.read().split('\n')[:-1]

    total_field_data = 12
    qname = []
    flag = []
    rname = []
    pos = []
    mapq = []
    cigar = []
    rnext = []
    pnext = []
    tlen = []
    seq = []
    qual = []
    optional_filed = []

    for line in all_line:
        split_data = line.split('\t')
        if len(split_data) > 10 and split_data[5] != '*': # cigar==* --> reject 
            qname.append(split_data[0])
            flag.append(split_data[1])
            rname.append(split_data[2])
            pos.append(int(split_data[3]))
            mapq.append(split_data[4])
            cigar.append(split_data[5])
            rnext.append(split_data[6])
            pnext.append(split_data[7])
            tlen.append(split_data[8])
            seq.append(split_data[9])
            qual.append(split_data[10])
            optional_filed.append('\t'.join(split_data[total_field_data-1:]))

    return qname, flag, rname, pos, mapq, cigar, rnext, pnext, tlen, seq, qual, optional_filed


def compute_length(cigar_string):
    """
    Compute length of read string with CIGAR
    """
    length = 0

    process_txt =  re.sub(r"[a-zA-Z]", " ", cigar_string)
    process_txt = process_txt.split(' ')
    for i in process_txt[:-1]:
        length += int(i)

    return length


def split_cigar(cigar_string):
    """
    Split CIGAR to (num, char), eg. 33M2D11S --> [('M', 33), ('D', 2), ('S', 11)]
    """
    process_txt =  re.sub(r"[a-zA-Z]", " ", cigar_string)
    nums = process_txt.split(' ')
    nums = [i for i in nums if i != '']
    
    process_txt =  re.sub(r"[0-9]", " ", cigar_string)
    chars = process_txt.split(' ')
    chars = [i for i in chars if i != '']
    
    result = []
    for i in range(len(nums)):
        result.append((chars[i], int(nums[i])))
    
    return result


def find_md_field(optional_data):
    """
    Find MD field in option fields
    """
    split_data = optional_data.split('\t')
    for i in split_data:
        if 'MD:Z' in i:
            return i 


def split_md_field(md_field, start, length):
    """
    Split MD field to find the mismatch position
    """
    md_field = md_field[5:]
    middle_split = []
    txt = ''
    i = 0
    while i < len(md_field):
        if md_field[i].isdigit():
            txt += md_field[i]
        elif md_field[i] =='^':
            if txt != '':
                middle_split.append(txt)
                txt = ''
            txt += '^'
            for j in range(i+1, len(md_field)):
                if not md_field[j].isdigit():
                    txt+=md_field[j]
                else:
                    i=j-1
                    middle_split.append(txt)
                    txt = ''
                    break
        else:
            if txt != '':
                middle_split.append(txt)
                txt = ''
            middle_split.append(md_field[i])
        
        if i == len(md_field) - 1 and txt != '':
            middle_split.append(txt)
        i+=1
    
    result_split = []
    count = 0
    total = 0
    for each_data in middle_split:
        if each_data[0] != '^':
            if each_data.isdigit():
                count += int(each_data)
                
                if count > start and total < length:
                    if count-start <= length:
                        middle = str(count-start)
                        total += count-start
                        start = count
                    else:
                        middle = str(length-total)
                        total = length
                    result_split.append(middle)
                
            elif each_data.isalpha():
                count+=1
                if count > start and total < length:
                    result_split.append(each_data)
                    total += 1
                    start+=1

    return result_split


def compute_qual(_string):
    """
    probability that the base read from the sequencer is faulty 
    """
    result = 0
    for char in _string:
        q = ord(char)
        result += np.power(10, -1 * (q-33)/10)
    return result/len(_string)  
