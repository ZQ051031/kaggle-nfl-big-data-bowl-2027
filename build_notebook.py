"""Generate portable notebook cells directly from source; no package import at runtime."""
import re
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).parent

def build():
    nb=nbf.v4.new_notebook()
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}}
    cells=[nbf.v4.new_markdown_cell('# Brake & Burst — Combine to NFL performance\n\nSelf-contained reproducible analysis. **Synthetic results are not NFL evidence.** Attach the licensed competition dataset in Kaggle after accepting the rules. No local package imports, authentication, downloads or external files are required. The fallback creates synthetic schema-matched data with planted signal. Code: MIT; competition data: CC BY-NC 4.0. Publish this notebook publicly and attach its URL to the writeup only after a real-data run and review.')]
    for name in ['data','synthetic','combine_features','game_features','modelling','plotting','pipeline']:
        source=(ROOT/f'src/bdb27/{name}.py').read_text()
        source=re.sub(r'^from \.\w+ import .*\n','',source,flags=re.M)
        cells += [nbf.v4.new_markdown_cell('## '+name.replace('_',' ').title()),nbf.v4.new_code_cell(source)]
    cells += [nbf.v4.new_markdown_cell('## Run: attached competition data, otherwise clearly labelled synthetic fixture'),nbf.v4.new_code_cell("from pathlib import Path\nDATA_PATH = Path('/kaggle/input/nfl-big-data-bowl-2027')\nDATA_PATH = DATA_PATH if DATA_PATH.exists() else None\nREPORT_TEMPLATE = "+repr((ROOT/'REPORT_TEMPLATE.md').read_text())+"\nsummary = run(data_dir=DATA_PATH, synthetic=DATA_PATH is None, report_template=REPORT_TEMPLATE)"),nbf.v4.new_code_cell("from IPython.display import display, Markdown, Image\ndisplay(Markdown(Path('REPORT.md').read_text()))\nfor path in sorted(Path('figures').glob('*.png')):\n    display(Image(filename=str(path)))\ndisplay(pd.read_csv('outputs/cv_metrics.csv').groupby(['validation','family','variant','task']).mean(numeric_only=True))")]
    nb.cells=cells
    nbf.write(nb,ROOT/'analysis.ipynb')

if __name__=='__main__': build()
