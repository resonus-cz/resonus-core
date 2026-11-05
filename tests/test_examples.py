from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.ipynb"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_notebook_runs(path):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3").execute()
    outputs = [o for cell in nb.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    assert any("image/png" in o.get("data", {}) for o in outputs)
