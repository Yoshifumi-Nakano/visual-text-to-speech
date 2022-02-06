import argparse
import pathlib
from pathlib import Path
from tqdm import tqdm
from convert_label import read_lab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lab',type=str,help='path to lab files. this program searchs for .lab files in specified directory and subdirectories')
    parser.add_argument('output',type=str,help='path to output accent and tg file')
    parser.add_argument('speaker',type=str,help='speaker name')

    args = parser.parse_args()
    lab_files = pathlib.Path(args.lab).glob('**/*.lab')

    # create output directory
    tg_dir = (Path(args.output) / 'TextGrid'/args.speaker)
    if not tg_dir.exists():
        tg_dir.mkdir(parents=True)

    # iter through lab files
    for lab_file in tqdm(lab_files):
        label = read_lab(str(lab_file))
        textgridFilePath = tg_dir/lab_file.with_suffix('.TextGrid').name
        label.to_textgrid(textgridFilePath)

        

    
