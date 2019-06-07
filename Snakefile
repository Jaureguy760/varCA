import os
import warnings
from snakemake.utils import min_version

##### set minimum snakemake version #####
min_version("5.5")

configfile: "config.yaml"


def read_samples():
    """Function to get names and dna fastq paths from a sample file
    specified in the configuration. Input file is expected to have 3
    columns: <unique_sample_id> <fastq1_path> <fastq2_path>. Modify
    this function as needed to provide a dictionary of sample_id keys and
    (fastq1, fastq1) values"""
    f = open(config['sample_file'], "r")
    samp_dict = {}
    for line in f:
        words = line.strip().split("\t")
        samp_dict[words[0]] = (words[1], words[2])
    return samp_dict
SAMP = read_samples()

# the user can change config['SAMP_NAMES'] here (or define it in the config
# file) to contain whichever sample names they'd like to run the pipeline on
if 'SAMP_NAMES' not in config:
    config['SAMP_NAMES'] = list(SAMP.keys())
else:
    # double check that the user isn't asking for samples they haven't provided
    user_samps = set(config['SAMP_NAMES'])
    config['SAMP_NAMES'] = list(set(SAMP.keys()).intersection(user_samps))
    if len(config['SAMP_NAMES']) != len(user_samps):
        warnings.warn("Not all of the samples requested have provided input. Proceeding with as many samples as is possible...")

# which pipeline should we run?
pipeline_type = []
if 'snp_callers' in config and config['snp_callers']:
    pipeline_type.append('snp')
if 'indel_callers' in config and config['indel_callers']:
    pipeline_type.append('indel')


rule all:
    # if you'd like to run the pipeline on only a subset of the samples,
    # you should specify them in the config['SAMP_NAMES'] variable above
    input:
        # expand(config['output_dir'] + "/peaks/{sample}/peaks.bed", sample=config['SAMP_NAMES'])
        expand(
            config['output_dir'] + "/merged_{type}/{sample}.tsv.gz", sample=config['SAMP_NAMES'],
            type=[
                i for i in ["snp", "indel"]
                if i+"_callers" in config and config[i+"_callers"]
            ]
        )

rule align:
    """Align reads using BWA-MEM. Note that we use -R to specify read group
    info for haplotype caller"""
    input:
        ref = config['genome'],
        fastq1 = lambda wildcards: SAMP[wildcards.sample][0],
        fastq2 = lambda wildcards: SAMP[wildcards.sample][1]
    output:
        config['output_dir'] + "/align/{sample}/aln.bam"
    threads: config['num_threads']
    conda: "env.yml"
    shell:
        "bwa mem -t {threads} {input.ref} {input.fastq1} {input.fastq2} "
        "-R '@RG\\tID:{wildcards.sample}\\tLB:lib1\\tPL:Illumina\\tPU:unit1\\tSM:{wildcards.sample}' | "
        "samtools view -S -b -h -F 4 -q 20 -> {output}"

rule add_mate_info:
    """Use fixmate to fill in mate coordinates and mate related flags, since
    our data is pair-ended. We need the MC tags (included because we used the
    -m flag) that it creates for markdup"""
    input:
        rules.align.output
    output:
        config['output_dir'] + "/align/{sample}/sorted.mated.bam"
    threads: config['num_threads']
    conda: "env.yml"
    shell:
        "samtools sort -n -@ {threads} {input} | "
        "samtools fixmate -m -@ {threads} -O bam - - | "
        "samtools sort -@ {threads} -o {output} -"

rule rm_dups:
    """Remove duplicates that may have occurred from PCR and index the
    resulting file."""
    input:
        rules.add_mate_info.output
    output:
        final_bam = config['output_dir'] + "/align/{sample}/rmdup.bam",
        final_bam_index = config['output_dir'] + "/align/{sample}/rmdup.bam.bai"
    threads: config['num_threads']
    conda: "env.yml"
    shell:
        "samtools markdup -@ {threads} {input} {output.final_bam} && "
        "samtools index -b -@ {threads} {output.final_bam}"

rule call_peaks:
    """Call peaks in the bam files using macs2"""
    input:
        rules.rm_dups.output.final_bam
    params:
        output_dir = lambda wildcards, output: os.path.dirname(output[0])
    output:
        config['output_dir'] + "/peaks/{sample}/{sample}_peaks.narrowPeak"
    conda: "env.yml"
    shell:
        "macs2 callpeak --nomodel --extsize 200 --slocal 1000 --qvalue 0.05 "
        "-g hs -f BAMPE -t {input} -n {wildcards.sample} --outdir {params.output_dir}"

rule bed_peaks:
    """Convert the BAMPE file to a sorted BED file"""
    input:
        rules.call_peaks.output
    output:
        config['output_dir'] + "/peaks/{sample}/peaks.bed"
    conda: "env.yml"
    shell:
        # to convert to BED, we must extract the first three columns (chr, start, stop)
        "cut -f -3 \"{input}\" | sort -k1,1V -k2,2n > \"{output}\""

rule run_caller:
    """Run any callers that are needed"""
    input:
        bam = rules.rm_dups.output.final_bam,
        peaks = rules.bed_peaks.output
    params:
        genome = config['genome'],
        temp_dir = lambda wildcards, output: os.path.dirname(output[0])+"/"+wildcards.caller
    output:
        config['output_dir'] + "/callers/{sample}/{caller}.tsv"
    conda: "env.yml"
    shell:
        "mkdir -p \"{params.temp_dir}\" && "
        "callers/{wildcards.caller}.bash {input.bam} {input.peaks} {params.genome} "
        "{output} {params.temp_dir} {wildcards.sample} {threads}"

rule prepare_all_sites:
    """prepare sites for output in the merged table"""
    input:
        rules.bed_peaks.output
    output:
        temp(config['output_dir'] + "/merged_snp/{sample}-all_sites.csv")
    shell:
        "awk -F '\\t' -v 'OFS=\\t' '{{for (i=$2;i<$3;i++) print $1\",\"i}}' "
        "{input} > {output}"

rule prepare_merge:
    """
        1) add the caller as a prefix of every column name
        2) sort the file by CHROM and POS
        3) separate chrom and pos cols by comma instead of tab
        4) replace NA with .
        5) remove the header
        (not necessarily in that order)
    """
    input:
        rules.run_caller.output
    output:
        pipe(config['output_dir'] + "/callers/{sample}/{caller}.prepared.tsv")
    conda: "env.yml"
    shell:
        "tail -n+2 {input} | sed -e 's/\\tNA\\t/\\t.\\t/g' | "
        "sort -k1,1V -k2,2n | sed -e 's/\\t\+/,/' > {output}"

rule join_all_sites:
    """
        1) add all sites to the prepared caller output using an outer join
        2) rename the column headers so we know which caller they came from
        3) get rid of the CHROM and POS cols so we can merge later
        4) rename the ref and alt columns as REF and ALT (to standardize them)
        (not necessarily in that order)
    """
    input:
        sites = rules.prepare_all_sites.output,
        caller_output = rules.prepare_merge.input,
        prepared_caller_output = rules.prepare_merge.output
    output:
        pipe(config['output_dir'] + "/merged_snp/{sample}.{caller}.tsv")
    conda: "env.yml"
    shell:
        "join -t $'\\t' -e. -a1 -a2 -o auto --nocheck-order "
        "{input.sites} {input.prepared_caller_output} | cut -f 2- | cat <("
            "head -n1 {input.caller_output} | cut -f 5- | tr '\\t' '\\n' | "
            "sed -e 's/^/{wildcards.caller}~/' | cat <("
                "echo -e \'\\t{wildcards.caller}~REF\\t{wildcards.caller}~ALT\'"
            ") - | paste -s"
        ") - > {output}"

rule merge_snp_callers:
    """merge the columns of each snp caller into a single file"""
    input:
        all_sites = rules.prepare_all_sites.output,
        caller_output = lambda wildcards: expand(
            rules.join_all_sites.output,
            caller=config['snp_callers'],
            sample=wildcards.sample
        )
    output:
        config['output_dir'] + "/merged_snp/{sample}.tsv.gz"
    conda: "env.yml"
    shell:
        "paste <(echo -e 'CHROM\tPOS'; sed -e 's/,\+/\\t/' {input.all_sites}) "
        "{input.caller_output} | gzip > {output}"

rule merge_indel_callers:
    """merge the columns of each indel caller into a single file"""
    input:
        lambda wildcards: expand(
            rules.join_all_sites.output,
            caller=config['indel_callers'],
            sample=wildcards.sample
        )
    output:
        config['output_dir'] + "/merged_indel/{sample}.tsv.gz"
    conda: "env.yml"
    shell:
        "paste {input} | sed -e 's/,\+/\\t/' | gzip > {output}"
