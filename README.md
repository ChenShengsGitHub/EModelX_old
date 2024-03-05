# EModelX
EModelX is a method for automatic cryo-EM protein complex structure modeling.

## Environment
`conda env create -f EModelX.yml`  
install requirement for AlphaFold:  
For AlphaFold: https://github.com/deepmind/alphafold    

## Minimal Example: Modeling for new EM maps
`python run_com_modeling.py --protocol=temp_flex --EM_map=./inputs/emd_32336.map.gz --fasta=./inputs/7w72.fasta --template_dir=./inputs/templates --output_dir=./outputs`  
, where you can replace `--protocol` within [seq_free,temp_free,temp_flex,temp_rigid]
, and `--EM_map` with your target EM map  
, and `--fasta` with your target fasta
, and `--template_dir`: the dir of template folder, needed when --protocol in [temp_flex,temp_rigid], path format for different chain please reference to ./inputs/templates  
, and `--output_dir`: the pdb output dir

## Web Server
https://bio-web1.nscc-gz.cn/app/EModelX
![EModelX](outputs/figure1.png)
