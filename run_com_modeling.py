import argparse
import warnings
import torch
from static_config import static_config
import numpy as np
import os
from modules.AutoEM import Solver
from unet3d.emmodel import ResUNet3D4EM


def main(dynamic_config):
    torch.manual_seed(static_config.seed)
    warnings.filterwarnings('ignore')
    np.set_printoptions(threshold=np.inf,suppress=True,precision=2)
    
    
    emid = dynamic_config.EM_map.split('/')[-1].split('emd_')[1].split('.map.gz')[0]
    resol = 'Unknown'
    pdbid = dynamic_config.EM_map.split('/')[-1].split('.fasta')[0]
    print('###\t{}\t{}\t{}\t###'.format(emid, pdbid,resol))

    if not os.path.exists(dynamic_config.EM_map):
        print('no em map! skip {}\t{}\t{}'.format(emid, pdbid,resol))
        return
    
    
    AutoEM_solver = Solver(emid, pdbid, resol, dynamic_config)
    
    BB_model = ResUNet3D4EM().to('cuda')
    CA_model = ResUNet3D4EM().to('cuda')
    AA_model = ResUNet3D4EM().to('cuda')
    BB_model.load_state_dict(torch.load(static_config.best_BB_model))
    CA_model.load_state_dict(torch.load(static_config.best_CA_model))
    AA_model.load_state_dict(torch.load(static_config.best_AA_model))
    BB_model.eval()
    CA_model.eval()
    AA_model.eval()

    AutoEM_solver.nnProcess(BB_model,CA_model,AA_model)
    
    AutoEM_solver.dynamic_config=dynamic_config
    run_result=AutoEM_solver.highConfFragAlign()
    if run_result != 'success':
        print(run_result)
        return

    AutoEM_solver.dynamic_config=dynamic_config
    run_result=AutoEM_solver.compModeling()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=str, default='temp_free', help='choose among seq_free,temp_free,temp_flex,temp_rigid')
    parser.add_argument('--EM_map', type=str, default='./inputs/emd_32336.map.gz', help='path of EM map, necessity')
    parser.add_argument('--fasta', type=str, default='./inputs/7w72.fasta', help='path of fasta file, needed when --protocol in [temp_free,temp_flex,temp_rigid]')
    parser.add_argument('--template_dir', type=str, default='./inputs/templates', help='dir of template folder, needed when --protocol in [temp_flex,temp_rigid], path format for different chain please reference to ./inputs/templates')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='dir of output pdbs')



    parser.add_argument('--CA_score_thrh', type=float, default=0.35, help='set as default is ok')
    parser.add_argument('--frags_len', type=int, default=150, help='set as default is ok')
    parser.add_argument('--MCS_n_hop', type=int, default=6, help='set as default is ok')
    parser.add_argument('--neigh_mat_thrh', type=float, default=0.7, help='set as default is ok')
    parser.add_argument('--mul_proc_num', type=int, default=30, help='set as default is ok')
    parser.add_argument('--MCS_score_thrh', type=float, default=2, help='set as default is ok')
    parser.add_argument('--gap_len', type=int, default=3, help='set as default is ok')
    parser.add_argument('--MCS_struct_len', type=int, default=5, help='set as default is ok')
    
    
    
    dynamic_config = parser.parse_args()

    main(dynamic_config)

