"""Give SAM file, return CSV file result"""
import statistics
import torch 
import torch.nn as nn
import os
import csv
import time
import sys
from tool.get_candidate_in_sam import Candidate
from tool.get_reference_string import ReferenceString
from model.dataloader import CustomDataset
import numpy as np 
from config import config_param
from tool.get_input_matrix_testing import FrequencyInputMatrix


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
## CUDA for PyTorch
torch.backends.cudnn.benchmark = True

class RunPredict:
    def __init__(self, sam_file, high_confidence = 0):
        self.sam_file = sam_file
        self.high_confidence = high_confidence
        self.confidence_threshold = 0.9
        self.model_path = './save_model/best_classifier_type_somatic_model_full_wes_ws16.pth'
        self.reference_file = './reference/GRCh38.d1.vd1.fa'
        self.window_size = 16
        self.type_label = config_param.type_label
        self.output_folder = './output'
        os.makedirs(self.output_folder, exist_ok=True)
        self.gen_info_50 = {
            'ABL1': [9, 130713043, 130887675],
            'AKT1': [14, 104769349, 104795748],
            'ALK': [2, 29190992, 29921589],
            'APC': [5, 112707498, 112846239],
            'ATM': [11, 108222484, 108369102],
            'BRAF': [7, 140713328, 140924929],
            'BRCA2': [13, 32315508, 32400268],
            'CDH1': [16, 68737292, 68835537],
            'CDKN2A': [9, 21967752, 21995324],
            'CSF1R': [5, 150053295, 150113365],
            'CTNNB1': [3, 41199422, 41240445],
            'EGFR': [7, 55019017, 55211628],
            'ERBB2': [17, 39688094, 39728660],
            'ERBB4': [2, 211375717, 212538802],
            'EZH2': [7, 148807374, 148884344],
            'FBXW7': [4, 152320544, 152536873],
            'FGFR1': [8, 38411143, 38468635],
            'FGFR2': [10, 121478330, 121598458],
            'FGFR3': [4, 1793293, 1808872],
            'FLT3': [13, 28003274, 28100587],
            'GNA11': [19, 3094362, 3123999],
            'GNAQ': [9, 77716097, 78031811],
            'GNAS': [20, 58839681, 58911192],
            'HNF1A': [12, 120977683, 121002512],
            'HRAS': [11, 532242, 535576],
            'IDH1': [2, 208236227, 208255071],
            'IDH2': [15, 90083045, 90102468],
            'JAK2': [9, 4984390, 5129948],
            'JAK3': [19, 17824782, 17848071],
            'KDR': [4, 55078481, 55125595],
            'KIT': [4, 54657957, 54740715],
            'KRAS': [12, 25205246, 25250929],
            'MET': [7, 116672196, 116798386],
            'MLH1': [3, 36993487, 37050846],
            'MPL': [1, 43336875, 43354466],
            'NOTCH1': [9, 136494433, 136546048],
            'NPM1': [5, 171387116, 171410900],
            'NRAS': [1, 114704469, 114716771],
            'PDGFRA': [4, 54229127, 54298245],
            'PIK3CA': [3, 179148114, 179240093],
            'PTEN': [10, 87863625, 87971930],
            'PTPN11': [12, 112418915, 112509918],
            'RB1': [13, 48303751, 48481890],
            'RET': [10, 43077069, 43130351],
            'SMAD4': [18, 51030213, 51085042],
            'SMARCB1': [22, 23786966, 23838009],
            'SMO': [7, 129188633, 129213548],
            'SRC': [20, 37344690, 37406050],
            'STK11': [19, 1205778, 1228431],
            'TP53': [17, 7668421, 7687490],
            'VHL': [3, 10141778, 10153667]
        }

    def __call__(self):
        start = time.time()
        # create file csv 
        # gen_name = os.path.basename(self.sam_file).split('.')[0]
        output_filename = self.sam_file.split('/')[-1].replace(".sam", "_out")
        if self.high_confidence: 
            output_filename = "high_confidence_{}".format(output_filename)
        writer = self.create_csv_file(output_filename)

        # get dataloader, predict and save
        dataloader = self.create_dataloader()
        print("Done creating dataloader!")
        
        if dataloader is not None:
            # load model
            model = torch.load(self.model_path)
            # model.to(device)
            print("Evaluating...")
            model.eval()

            predict = []
            candidates = []

            for local_batch, candidate in dataloader:
                # Transfer to GPU
                local_batch = local_batch.to(device)
                outputs, _ = model(local_batch)
                predict.extend(outputs.data.cpu().numpy())
                candidates.extend(candidate.data.numpy())

            assert len(predict) == len(candidates)
            for i in range(len(predict)):
                softmax = nn.Softmax()
                softmax_output = softmax(torch.from_numpy(predict[i])).cpu().detach().numpy()
                statistic = np.max(softmax_output)
                type_soma = self.get_key(np.argmax(softmax_output), self.type_label)
                if np.argmax(softmax_output) != 0:
                    if (self.high_confidence and statistic >= self.confidence_threshold) or not self.high_confidence:
                        self.write_csv(writer, (candidates[i], type_soma, statistic))

        print('Process with {}s', time.time()-start)

    def create_dataloader(self):
        get_candidate = Candidate(chr_split=1)
        list_candidate = get_candidate(self.sam_file)
        print(list_candidate)
        # get chromosome and gen name
        # gen_name = os.path.basename(self.sam_file).split('.')[0]
        # num_chr = self.gen_info_50[gen_name][0]
        get_ref = ReferenceString(self.reference_file)
        all_ref = get_ref.get_22_chromosome()
        # ref_string = all_ref['chr{}'.format(num_chr)]
        print("Done getting reference!")
        print(len(all_ref))
        get_matrix = FrequencyInputMatrix(self.window_size)
        matrix = []
        candidates = []

        print("Generating input matrices...")
        for i in range(1,23):
            ref_string = all_ref['chr{}'.format(i)]
            list_candidate_chr = list_candidate[i]
            for candidate in list_candidate_chr:
                _matrix = get_matrix(self.sam_file, candidate, ref_string)
                if _matrix is not None:
                    matrix.append(_matrix)
                    candidates.append(candidate)

        if len(matrix) > 0:
            matrix = np.array(matrix)
            input_matrix = self.norm_matrix(matrix)

            # convert to tensor torch (Chanels, Heigh, Width)
            input_matrix =  torch.permute(torch.from_numpy(input_matrix), (0, 3, 1, 2))
            candidates = np.array(candidates)
            # Create Dataloader 
            params = {'batch_size': 32,
                    'shuffle': True,
                    'num_workers': 3}

            dataset = CustomDataset(input_matrix, candidates)
            data_generator = torch.utils.data.DataLoader(dataset, **params)

            return data_generator
        else:
            return None

    def norm_matrix(self, matrix):
        for i in range(len(matrix)):
            if np.max(matrix[i, :, :, 1]) !=0:
                matrix[i, :, :, 1] /= np.max(matrix[i, :, :, 1])
            if np.max(matrix[i, :, :, 2]) !=0:
                matrix[i, :, :, 2] /= np.max(matrix[i, :, :, 2])
        return matrix

    def create_csv_file(self, gen_name):
        f = open(os.path.join(self.output_folder,'{}.csv'.format(gen_name)), 'w')
        writer = csv.writer(f)
        writer.writerow(['candidate_position', 'type_somatic', 'statiscal'])
        return writer

    def write_csv(self, writer, data):
        candidate, type_soma, statistical = data
        writer.writerow([candidate, type_soma, statistical]) 

    # function to return key for any value
    def get_key(self, val, my_dict):
        for key, value in my_dict.items():
            if val == value:
                return key
        
if __name__=="__main__":
    file_path = sys.argv[1]
    if len(sys.argv) > 2:
        run = RunPredict(file_path, high_confidence=('-h' in sys.argv))
    else:
        run = RunPredict(file_path)
    run()


        


