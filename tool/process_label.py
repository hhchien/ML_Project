def check_label(position, _ref, _alt, truth_data):
    """
    Check position is non-somatic, snv or indel
    """
    positions, refs, alts = truth_data
    for index in range(len(positions)):
        if positions[index] == position:
            if _ref == refs[index] and _alt == alts[index]:
                return True
    return False


def get_infor_label(vcf_file, chromosome):
    """
    Get ground truth for each chromosome from vcf file
    """
    f = open(vcf_file, 'r')
    lines = f.read().split('\n')[:-1]
    
    find = False
    positions = []
    refs = []
    alts = []
    
    for line in lines:
        if line[0] != '#':
            line_split = line.split('\t')
            
            if find and line_split[0] != 'chr'+str(chromosome):
                break
            elif line_split[0] == 'chr'+str(chromosome):
                find = True
                positions.append(int(line_split[1]))
                refs.append(line_split[3])
                alts.append(line_split[4])
    
    return positions, refs, alts 

