## Required environment
- Python 3.9
- pip 22.0.4

## Install required modules
```$ pip install -r requirements.txt```

## Preprocessing data and training
- Visit the following URL
https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files
- Download the reference files which is too large for us to push (github allows only < 100MB file)
GRCh38.d1.vd1.fa.tar.gz
- Compress and save GRCh38.d1.vd1.fa in ./reference

- Change the paths from files (Only when any alternation is made)

- For training, input data supposes to be BAM files which contain aligned gene sequencing data provided by the laboratories and (a) VCF file(s) where high confidence variants are recorded. However, those files are huge for us to attach in this repository. Thus, manually download required data is needed. They are published publicly on:
    - BAM files: https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/seqc/Somatic_Mutation_WG/data/WES/
    - VCF files: https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/seqc/Somatic_Mutation_WG/release/latest/

- With those above data files, run: ```$ python3 create_sam_input_training.py <lab_name>``` to create SAM file for each candidate (a data point), where ```<lab_name>``` is the lab to collect data. E.g.: WES_IL_T_2,...

- For a easier use without requirement of external data, we have already attached SAM files for candidates from six laboratories on 'data/candidate_sam'. If following this way, please start from next step.

- After obtaining SAM files, run: ```$ python3 create_data_pickle_wes_training.py <ws> <lab_name> ``` to create input matrix and label for each candidate to feed into the model, where ```<ws>``` is the window size (e.g. 10, 16, 20, ...) and ```<lab_name>``` is the lab from which data collected above.

- To create known-label test data, run: ```$ python3 create_data_pickle_wes_test.py <ws> <lab_name> ```, where two arguments are the same as previous step.

- Training the model and testing it on the known-label data created above, run: ```$ python3 training_wes.py <ws>```, where ```<ws>``` is the window size of the input matrices.

## Note
If running on some specific laboratories, please remember to change the list_lab on training_wes.py
