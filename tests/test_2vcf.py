import subprocess
import sys
import textwrap
from pathlib import Path
import pytest

pytest.importorskip('pandas')
pytest.importorskip('numpy')
pytest.importorskip('pysam')


def test_2vcf_runs(tmp_path):
    prepared = textwrap.dedent("""
        CHROM\tPOS\tREF\tcall-snp~REF\tcall-snp~ALT
        1\t100\tA\tA\tG
        1\t200\tT\tT\t.
    """).strip()
    classified = textwrap.dedent("""
        CHROM\tPOS\tvarca~prob.1\tvarca~CLASS:
        1\t100\t0.9\t1
        1\t200\t0.3\t0
    """).strip()

    prep_path = tmp_path / "prep.tsv"
    prep_path.write_text(prepared)
    class_path = tmp_path / "class.tsv"
    class_path.write_text(classified)
    out_vcf = tmp_path / "out.vcf"

    subprocess.run([
        sys.executable,
        str((Path(__file__).resolve().parent.parent / 'scripts' / '2vcf.py')),
        '-o', str(out_vcf),
        str(class_path),
        str(prep_path)
    ], check=True)

    assert out_vcf.exists()
    text = out_vcf.read_text().splitlines()
    assert any(line.startswith('#CHROM') for line in text)
    assert any(not line.startswith('#') for line in text)

