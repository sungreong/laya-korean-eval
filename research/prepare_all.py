import runpy
from pathlib import Path
root=Path(__file__).resolve().parent
runpy.run_path(str(root/'prepare_source.py'),run_name='__main__')
runpy.run_path(str(root/'prepare_eval.py'),run_name='__main__')
runpy.run_path(str(root/'download_checkpoint.py'),run_name='__main__')
