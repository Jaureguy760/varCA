import subprocess
import sys
import textwrap
import pytest
from pathlib import Path

np = pytest.importorskip('numpy')
pd = pytest.importorskip('pandas')
sklearn = pytest.importorskip('sklearn')

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score


def test_metrics_script(tmp_path):
    table = textwrap.dedent("""
        1\t1\t0.9
        1\t0\t0.2
        0\t0\t0.1
        0\t1\t0.8
    """).strip()
    table_path = tmp_path / "table.tsv"
    table_path.write_text(table)
    out_path = tmp_path / "out.txt"

    subprocess.run([
        sys.executable,
        str((Path(__file__).resolve().parent.parent / 'scripts' / 'metrics.py')),
        '-o', str(out_path),
        str(table_path)
    ], check=True)

    out_vals = [float(x) for x in out_path.read_text().splitlines()]

    df = pd.read_csv(table_path, sep='\t', header=None, names=['truth','predict','probs'])
    scores = list(precision_recall_fscore_support(df['truth'], df['predict'], beta=0.5, average='binary')[:3])
    scores += list(confusion_matrix(df['truth'], df['predict']).sum(0))
    scores += [roc_auc_score(df['truth'], df['probs']), average_precision_score(df['truth'], df['probs'])]

    expected = [scores[idx] for idx in [1,0,2,4,5,6,7]]

    assert pytest.approx(out_vals) == expected

