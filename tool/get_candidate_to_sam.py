"""
Create SAM file with list position somatic in VCF and BAM file
"""

import os

class CandidateSam:
    ''' 
    using cmd: samtools view -h _path_bam_file '_candidate:_start-_end' > _candidate_number.sam
    '''
    def __init__(self, bam_file_path, window_size, path_save):
        self.bam_file = bam_file_path
        self.window_size = window_size
        self.path_save = path_save
        self.command = 'samtools view -h {} \'chr{}:{}-{}\' > {}'  # (bam_file_path, chromo, candidate-window_size, candidate+window_size, sam_file_path)
    
    def __call__(self,chromosome, list_candidate, list_ref, list_alt):
        os.makedirs(os.path.join(self.path_save, 'chr'+str(chromosome)), exist_ok=True)

        # find reads contain each candidate
        for i in range(len(list_candidate)):
            candidate = list_candidate[i]
            ref = list_ref[i]
            alt = list_alt[i]
            if len(ref) == len(alt) == 1:
                file_save_path = os.path.join(self.path_save, 'chr'+str(chromosome), '{}_snv.sam'.format(candidate))
            else:
                file_save_path = os.path.join(self.path_save, 'chr'+str(chromosome), '{}_indel.sam'.format(candidate))
                
            os.system(self.command.format(self.bam_file, chromosome, candidate-self.window_size, candidate+self.window_size, file_save_path))
    