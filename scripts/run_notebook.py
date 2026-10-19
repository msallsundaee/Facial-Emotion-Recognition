"""Execute the lab and retain outputs, including the last cell if execution fails."""
from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(root / '.cache' / 'matplotlib'))
os.environ.setdefault('KERAS_HOME', str(root / '.cache' / 'keras'))
os.environ.setdefault('IPYTHONDIR', str(root / '.cache' / 'ipython'))
source = root / 'notebooks' / 'week7_fer.ipynb'
destination = root / 'outputs' / 'week7_fer_executed.ipynb'
destination.parent.mkdir(parents=True, exist_ok=True)
notebook = nbformat.read(source, as_version=4)
client = NotebookClient(notebook, timeout=7200, kernel_name='python3',
                        resources={'metadata': {'path': str(root)}})
def cell_started(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Executing cell {cell_index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0]}', flush=True)


def cell_finished(**kwargs):
    nbformat.write(notebook, destination)


client.on_cell_start = cell_started
client.on_cell_executed = cell_finished
try:
    client.execute()
finally:
    nbformat.write(notebook, destination)
    print(f'Notebook outputs saved to {destination}', flush=True)
