import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent/'src'))
from bdb27.pipeline import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Brake & Burst: Combine to NFL translation')
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--data-dir')
    args=parser.parse_args()
    run(args.data_dir,args.synthetic,report_template=(Path(__file__).parent/'REPORT_TEMPLATE.md').read_text())
