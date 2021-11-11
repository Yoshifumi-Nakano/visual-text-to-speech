import argparse
import pathlib
from pathlib import Path
import re
import sys

from tqdm import tqdm

from convert_label import read_lab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lab',type=str,help='path to lab files. this program searchs for .lab files in specified directory and subdirectories')
    parser.add_argument('output',type=str,help='path to output accent and tg file')
    parser.add_argument('speaker',type=str,help='speaker name')
    parser.add_argument('--with_accent',action='store_true',help='whether you want accent labels or not')

    args = parser.parse_args()
    lab_files = pathlib.Path(args.lab).glob('**/*.lab')
    print("lab_files",lab_files)

    # create output directory
    
    tg_dir = (Path(args.output) / 'TextGrid'/args.speaker)
    ac_dir = (Path(args.output)/ 'accent')
    if not tg_dir.exists():
        tg_dir.mkdir(parents=True)
    if not ac_dir.exists():
        ac_dir.mkdir()

    # iter through lab files
    for lab_file in tqdm(lab_files):
        if args.with_accent:
            accent = []
            with open(lab_file) as f:
                lines = f.readlines()
            lab, accent = pp_symbols(lines)
            with open(ac_dir/ lab_file.with_suffix('.accent').name,mode='w') as f:
                f.writelines([''.join(accent)])
        
        label = read_lab(str(lab_file))
        textgridFilePath = tg_dir/lab_file.with_suffix('.TextGrid').name
        label.to_textgrid(textgridFilePath)

        

    
