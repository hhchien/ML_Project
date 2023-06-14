""" 
Get list candidate in vcf and create list file sam, which contain reads with cancidate in between
"""

import os
import sys
from tool.get_candidate_to_sam import CandidateSam
from tool.process_label import get_infor_label

# for wes data
_bam_file = './data/bam/{}.bwa.dedup.bam' #path to BAM raw data files of six laboratories

truth_vcf_1 = './reference/vcf/high-confidence_sINDEL_in_HC_regions_v1.2.vcf' #high confidence files contain proved somatic variants 
truth_vcf_2 = './reference/vcf/high-confidence_sSNV_in_HC_regions_v1.2.vcf'   #that practically detected by six laboratories. Can be
                                                                              #consider as ground truth to label training data.

_candidate_sam_folder = './data/candidate_sam/{}'

lab_name = sys.argv[1]
bam_file = _bam_file.format(lab_name) # eg. WES_IL_T_2
candidate_sam_folder = _candidate_sam_folder.format(lab_name)
get_save_candidate = CandidateSam(bam_file_path = bam_file, window_size=160, path_save=candidate_sam_folder)

# run with chr1 - chr22
for i in range(22):
    print('Processing chr{} ...'.format(i+1))
    num_chr = i+1
    positions, refs, alts  = get_infor_label(truth_vcf_1, num_chr)
    get_save_candidate(num_chr, positions, refs, alts)

    positions, refs, alts  = get_infor_label(truth_vcf_2, num_chr)
    get_save_candidate(num_chr, positions, refs, alts)

print('Done!')