#!/bin/bash

bam="$1"
[[ -z "$1" ]] && { echo "Parameter 1 is empty" 1>&2; exit 1; }
peaks="$2"
[[ -z "$2" ]] && { echo "Parameter 2 is empty" 1>&2; exit 1; }
genome="$3"
[[ -z "$3" ]] && { echo "Parameter 3 is empty" 1>&2; exit 1; }
output_dir="$4"
[[ -z "$4" ]] && { echo "Parameter 4 is empty" 1>&2; exit 1; }
gatk_dir="$7"
[[ -z "$7" ]] && { echo "Parameter 7 is empty" 1>&2; exit 1; }


gatk --java-options "-Xmx4g" SelectVariants \
  -L "$peaks" \
  -R "$genome" \
  -V "$gatk_dir/gatk.vcf.gz" \
  --select-type SNP --select-type NO_VARIATION \
  -O "$output_dir/gatk-snps.vcf.gz"

# convert vcf to table
gatk --java-options "-Xmx4g" VariantsToTable \
  -V "$output_dir/gatk-snps.vcf.gz" \
  -O "$output_dir/gatk-snp.tsv" \
  -L "$peaks" \
  -F CHROM -F POS -F REF -F ALT -F QD -F FS \
  -F SOR -F MQ -F ReadPosRankSum -GF DP -GF GT;
