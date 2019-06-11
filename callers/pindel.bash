#!/bin/bash

bam="$1"
[[ -z "$1" ]] && { echo "Parameter 1 is empty" 1>&2; exit 1; }
peaks="$2"
[[ -z "$2" ]] && { echo "Parameter 2 is empty" 1>&2; exit 1; }
genome="$3"
[[ -z "$3" ]] && { echo "Parameter 3 is empty" 1>&2; exit 1; }
output_dir="$4"
[[ -z "$4" ]] && { echo "Parameter 4 is empty" 1>&2; exit 1; }
samp="$5"
[[ -z "$5" ]] && { echo "Parameter 5 is empty" 1>&2; exit 1; }
threads="$6"
[[ -z "$6" ]] && { echo "Parameter 6 is empty" 1>&2; exit 1; }


pindel -f "$genome" -i <(echo "$bam" 300 "$samp") -j "$peaks" -o "$output_dir/" -L "$output_dir/log" -T "$threads"
for file in _D _INV _SI _TD; do
	pindel2vcf -p "$output_dir/$file" -r "$genome" -R x -d 00000000 -v "$output_dir/$file.raw.vcf"
	bcftools norm -d all "$output_dir/$file.raw.vcf" > "$output_dir/$file.vcf"
	bgzip -f "$output_dir/$file.vcf" && tabix -p vcf -f "$output_dir/$file.vcf.gz"
done
echo -e "CHROM\tPOS\tREF\tALT\tHOMLEN\tSVLEN\tSVTYPE\tNTLEN\tPL\tGT\tRD\tAD" > "$output_dir/pindel.tsv"
bcftools concat "$output_dir/"{_D,_INV,_SI,_TD}.vcf.gz | \
bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/HOMLEN\t%INFO/SVTYPE\t%INFO/NTLEN\t[%PL]\t[%GT]\t[%RD]\t[%AD]\n' - >> "$output_dir/pindel.tsv"
