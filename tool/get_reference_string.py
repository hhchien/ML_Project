import Bio
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import GC


# print("biopython version: ", Bio.__version__)

class ReferenceString:
    def __init__(self, reference_fasta_file):
        self.file_fasta = reference_fasta_file

        self.list_concern_chr = []
        for i in range(1, 23):
            self.list_concern_chr.append('chr'+str(i))

    def get_22_chromosome(self):
        reference_sequence = {}

        for seq_record in SeqIO.parse(self.file_fasta, "fasta"):
            if seq_record.id in self.list_concern_chr:
                reference_sequence[seq_record.id] = seq_record.seq

        return reference_sequence
    
    def statistic_all_gen(self):
        # FASTA parsing example
        count_gens = 0
        list_chromosome = []

        for seq_record in SeqIO.parse(self.file_fasta, "fasta"):
            count_gens += 1
            list_chromosome.append(seq_record.id)

        # chr1 - chr22
        print("Total {} gens".format(count_gens))
        print("List chromosomes:", list_chromosome)


    



    

