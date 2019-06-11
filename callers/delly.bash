#!/bin/bash

bam="$1"
[[ -z "$1" ]] && { echo "Parameter 1 is empty" 1>&2; exit 1; }
peaks="$2"
[[ -z "$2" ]] && { echo "Parameter 2 is empty" 1>&2; exit 1; }
genome="$3"
[[ -z "$3" ]] && { echo "Parameter 3 is empty" 1>&2; exit 1; }
output_dir="$4"
[[ -z "$4" ]] && { echo "Parameter 4 is empty" 1>&2; exit 1; }


echo -e "CHROM\tPOS\tREF\tALT\tRC\tCN\tGT\tGQ" > "$output_dir/delly.tsv"
delly call -g "$genome" "$bam" -o "$output_dir"/delly.bcf && \
bcftools convert "$output_dir"/delly.bcf -Ov | \
vcftools --vcf - --recode --keep-INFO-all --bed "$peaks" -c | \
bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%RC]\t[%CN]\t[%GT]\t[%GQ]\n' - >> "$output_dir/delly.tsv"
