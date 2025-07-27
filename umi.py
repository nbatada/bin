#!/usr/bin/env python3

import pysam
import argparse
from collections import defaultdict, Counter
import os
import sys # Import sys for stderr
import numpy as np # For median calculation

def parse_umi_location(umi_preset, umi_read_arg, umi_start_arg, umi_end_arg):
    if umi_preset == "takara_tcr":
        read_with_umi = 'R2'
        read_idx = 1 # R2 is at index 1 in (R1, R2) tuple
        start = 0
        end = 12
    else: # "custom" preset or default
        read_with_umi = umi_read_arg
        read_idx = 0 if umi_read_arg.upper() == 'R1' else 1
        start = umi_start_arg
        end = umi_end_arg
    return read_idx, start, end

def umi_count_main(args):
    umi_read_idx, umi_start, umi_end = parse_umi_location(args.umi_preset, args.umi_read, args.umi_start, args.umi_end)
    
    # Extract sample name from the input R1 file
    sample_name = os.path.basename(args.f1).replace('_R1.fastq.gz', '').replace('_R1.fastq', '')

    umi_grouped_sequences = defaultdict(list)
    total_reads = 0
    reads_without_umi_extracted = 0

    with pysam.FastxFile(args.f1) as r1_file, pysam.FastxFile(args.f2) as r2_file:
        for r1_read, r2_read in zip(r1_file, r2_file):
            total_reads += 1
            
            target_read = r1_read if umi_read_idx == 0 else r2_read
            
            umi = ""
            if len(target_read.sequence) >= umi_end:
                umi = target_read.sequence[umi_start:umi_end]
            else:
                reads_without_umi_extracted += 1
                umi = "UMI_NOT_EXTRACTED" 

            # Trim the first 5bp from Read 1 for biological sequence comparison
            r1_seq_bio = r1_read.sequence[5:] if len(r1_read.sequence) >= 5 else ""
            # Trim the first 19bp from Read 2 for biological sequence comparison
            r2_seq_bio = r2_read.sequence[19:] if len(r2_read.sequence) >= 19 else ""

            if umi != "UMI_NOT_EXTRACTED":
                umi_grouped_sequences[umi].append((r1_seq_bio, r2_seq_bio))
    
    umi_counts = {umi: len(seq_list) for umi, seq_list in umi_grouped_sequences.items()}
    num_unique_umis = len(umi_counts)
    
    # Calculate median reads per UMI
    umi_counts_list = [count for umi, count in umi_counts.items() if umi != "UMI_NOT_EXTRACTED"]
    median_reads_per_umi = np.median(umi_counts_list) if umi_counts_list else 0

    # Print statistical summary as comments
    print(f"# sample_id\t{sample_name}")

    # Most important metrics (reordered)
    print(f"# reads_total\t{total_reads}") # 1) number of input reads
    print(f"# reads_left_umi_collapse\t{num_unique_umis}") # 2) number of reads after umi collapsing (equivalent to unique UMIs)
    umi_unique_percent_of_total_reads = (num_unique_umis / total_reads * 100) if total_reads > 0 else 0
    print(f"# reads_pct_left\t{umi_unique_percent_of_total_reads:.2f}") # 3) percent of total reads left after umi collapsing
    print(f"# umi_count\t{num_unique_umis}") # New: total number of unique UMIs
    
    if num_unique_umis > 0:
        avg_reads_per_umi = total_reads / num_unique_umis
        print(f"# reads_per_umi\t{avg_reads_per_umi:.2f}") # 4) average number of reads per umi
    else:
        print(f"# reads_per_umi\tN/A")
    # print(f"# umi.median_reads_per_umi\t{median_reads_per_umi:.2f}") # 5) median number of reads per umi

    # Print the UMI frequency table
    print("umi\tcount")
    for umi, count in umi_counts.items():
        if umi != "UMI_NOT_EXTRACTED":
            print(f"{umi}\t{count}")

def move_umi_and_trim_main(args):
    umi_read_idx, umi_start, umi_end = parse_umi_location(args.umi_preset, args.umi_read, args.umi_start, args.umi_end)

    # Extract sample name from the input R1 file
    sample_name = os.path.basename(args.f1).replace('_R1.fastq.gz', '').replace('_R1.fastq', '')

    output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.basename(args.f1).replace('_R1.fastq.gz', '')
    
    if args.output_prefix:
        prefix_dir = os.path.dirname(os.path.abspath(args.output_prefix))
        if not prefix_dir:
            prefix_dir = output_dir
        os.makedirs(prefix_dir, exist_ok=True)
        final_output_prefix_base = os.path.basename(args.output_prefix)
        output_r1_path = os.path.join(prefix_dir, f"{final_output_prefix_base}_R1.fastq")
        output_r2_path = os.path.join(prefix_dir, f"{final_output_prefix_base}_R2.fastq")
    elif args.collapse:
        output_r1_path = os.path.join(output_dir, f"{base_filename}_umi_collapsed_R1.fastq")
        output_r2_path = os.path.join(output_dir, f"{base_filename}_umi_collapsed_R2.fastq")
    else:
        output_r1_path = os.path.join(output_dir, f"{base_filename}_umi_trimmed_R1.fastq")
        output_r2_path = os.path.join(output_dir, f"{base_filename}_umi_trimmed_R2.fastq")

    total_input_reads_move_trim = 0
    total_output_reads_move_trim = 0

    print(f"DEBUG: Output directory: {output_dir}", file=sys.stderr)
    print(f"DEBUG: Output R1 path: {output_r1_path}", file=sys.stderr)
    print(f"DEBUG: Output R2 path: {output_r2_path}", file=sys.stderr)

    if args.collapse:
        umi_grouped_reads_for_output = defaultdict(list)
        
        with pysam.FastxFile(args.f1) as r1_in, pysam.FastxFile(args.f2) as r2_in:
            for r1_read, r2_read in zip(r1_in, r2_in):
                total_input_reads_move_trim += 1
                
                target_read = r1_read if umi_read_idx == 0 else r2_read
                umi = ""
                if len(target_read.sequence) >= umi_end:
                    umi = target_read.sequence[umi_start:umi_end]
                else:
                    umi = "UMI_NOT_EXTRACTED" 
                
                r1_read.name = f"{r1_read.name}_UMI:{umi}"
                r2_read.name = f"{r2_read.name}_UMI:{umi}"

                if args.trim:
                    r2_read.sequence = r2_read.sequence[19:] if len(r2_read.sequence) >= 19 else ""
                    r2_read.quality = r2_read.quality[19:] if len(r2_read.quality) >= 19 else ""

                if umi != "UMI_NOT_EXTRACTED":
                    umi_grouped_reads_for_output[umi].append((r1_read, r2_read))
        
        with open(output_r1_path, 'w') as r1_out, \
             open(output_r2_path, 'w') as r2_out:
            for umi, reads_list in umi_grouped_reads_for_output.items():
                if reads_list:
                    r1_read_out = reads_list[0][0]
                    r2_read_out = reads_list[0][1]
                    r1_out.write(f"@{r1_read_out.name}\n{r1_read_out.sequence}\n+\n{r1_read_out.quality}\n")
                    r2_out.write(f"@{r2_read_out.name}\n{r2_read_out.sequence}\n+\n{r2_read_out.quality}\n")
                    total_output_reads_move_trim += 1
    else:
        with pysam.FastxFile(args.f1) as r1_in, \
             pysam.FastxFile(args.f2) as r2_in, \
             open(output_r1_path, 'w') as r1_out, \
             open(output_r2_path, 'w') as r2_out:
            for r1_read, r2_read in zip(r1_in, r2_in):
                total_input_reads_move_trim += 1
                
                target_read = r1_read if umi_read_idx == 0 else r2_read
                umi = ""
                if len(target_read.sequence) >= umi_end:
                    umi = target_read.sequence[umi_start:umi_end]
                else:
                    umi = "UMI_NOT_EXTRACTED" 
                
                r1_read.name = f"{r1_read.name}_UMI:{umi}"
                r2_read.name = f"{r2_read.name}_UMI:{umi}"

                if args.trim:
                    r2_read.sequence = r2_read.sequence[19:] if len(r2_read.sequence) >= 19 else ""
                    r2_read.quality = r2_read.quality[19:] if len(r2_read.quality) >= 19 else ""

                r1_out.write(f"@{r1_read.name}\n{r1_read.sequence}\n+\n{r1_read.quality}\n")
                r2_out.write(f"@{r2_read.name}\n{r2_read.sequence}\n+\n{r2_read.quality}\n")
                total_output_reads_move_trim += 1

    # Print statistical summary as comments
    print(f"# sample_id\t{sample_name}")
    print(f"# read.total_input\t{total_input_reads_move_trim}")
    print(f"# read.total_output\t{total_output_reads_move_trim}")
    if total_input_reads_move_trim > 0:
        output_percent = (total_output_reads_move_trim / total_input_reads_move_trim) * 100
        print(f"# read.output_percent_of_input\t{output_percent:.2f}")
    else:
        print(f"# read.output_percent_of_input\tN/A")
    
    print(f"# processed_files\t{args.f1},{args.f2}")
    print(f"# output_files\t{output_r1_path},{output_r2_path}")


def inspect_umi_variations_main(args):
    umi_read_idx, umi_start, umi_end = parse_umi_location(args.umi_preset, args.umi_read, args.umi_start, args.umi_end)
    
    all_r1_sequences = []
    all_r2_sequences = []

    with pysam.FastxFile(args.f1) as r1_file, pysam.FastxFile(args.f2) as r2_file:
        for r1_read, r2_read in zip(r1_file, r2_file):
            target_read = r1_read if umi_read_idx == 0 else r2_read
            
            umi = ""
            if len(target_read.sequence) >= umi_end:
                umi = target_read.sequence[umi_start:umi_end]
            else:
                umi = "UMI_NOT_EXTRACTED" 

            if umi in args.inspect_umis:
                # Trim the first 5bp from Read 1 for biological sequence comparison
                r1_seq_bio = r1_read.sequence[5:] if len(r1_read.sequence) >= 5 else ""
                r2_seq_bio = r2_read.sequence[19:] if len(r2_read.sequence) >= 19 else "" # Trim first 19bp
                
                # Store original read name to include in FASTA header
                original_r1_name = r1_read.name.split(' ')[0] # Take only the first part of the read name
                original_r2_name = r2_read.name.split(' ')[0]

                all_r1_sequences.append((original_r1_name, umi, r1_seq_bio))
                all_r2_sequences.append((original_r2_name, umi, r2_seq_bio))

    # Print R1 sequences first
    for name, umi, seq in all_r1_sequences:
        print(f">R1_{name}_UMI:{umi}")
        print(seq)

    # Then print R2 sequences
    for name, umi, seq in all_r2_sequences:
        print(f">R2_{name}_UMI:{umi}")
        print(seq)


def main():
    print(f"umi.py version: {VERSION}", file=sys.stderr)

    parser = argparse.ArgumentParser(
        description="""UMI processing utilities for FASTQ files.
This script provides functionalities to count UMIs or process FASTQ files
by moving UMIs to read names and trimming non-biological regions.
All FASTQ inputs are assumed to be gzipped (.gz).""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_count = subparsers.add_parser(
        "umi_count", 
        help="Count unique UMIs and their occurrences.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    required_count = parser_count.add_argument_group('Required Arguments')
    required_count.add_argument("-f1", required=True, help="Path to gzipped R1 FASTQ file.")
    required_count.add_argument("-f2", required=True, help="Path to gzipped R2 FASTQ file.")
    
    optional_count = parser_count.add_argument_group('Optional Arguments')
    optional_count.add_argument("--umi-preset", default="custom", choices=["custom", "takara_tcr"],
                                     help="""Predefined UMI location preset.
'takara_tcr': Sets UMI to R2, start 0, end 12.
'custom': Requires --umi-read, --umi-start, --umi-end to be specified.
Default: custom""")
    optional_count.add_argument("--umi-read", default="R2", choices=["R1", "R2"],
                                     help="Specify which read contains the UMI (R1 or R2). Only used if --umi-preset is 'custom'. Default: R2")
    optional_count.add_argument("--umi-start", type=int, default=0,
                                     help="0-indexed start position of the UMI in the specified read. Only used if --umi-preset is 'custom'. Default: 0")
    optional_count.add_argument("--umi-end", type=int, default=12,
                                     help="0-indexed end position (exclusive) of the UMI in the specified read. Only used if --umi-preset is 'custom'. Default: 12")
    parser_count.set_defaults(func=umi_count_main)

    parser_move_trim_collapse = subparsers.add_parser(
        "move_umi_and_trim",
        help="Move UMI to read name, trim from R2, and optionally collapse reads.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    required_move_trim = parser_move_trim_collapse.add_argument_group('Required Arguments')
    required_move_trim.add_argument("-f1", required=True, help="Path to gzipped R1 FASTQ file.")
    required_move_trim.add_argument("-f2", required=True, help="Path to gzipped R2 FASTQ file.")

    optional_move_trim = parser_move_trim_collapse.add_argument_group('Optional Arguments')
    optional_move_trim.add_argument("--umi-preset", default="custom", choices=["custom", "takara_tcr"],
                                     help="""Predefined UMI location preset.
'takara_tcr': Sets UMI to R2, start 0, end 12.
'custom': Requires --umi-read, --umi-start, --umi-end to be specified.
Default: custom""")
    optional_move_trim.add_argument("--umi-read", default="R2", choices=["R1", "R2"],
                                     help="Specify which read contains the UMI (R1 or R2). Only used if --umi-preset is 'custom'. Default: R2")
    optional_move_trim.add_argument("--umi-start", type=int, default=0,
                                     help="0-indexed start position of the UMI in the specified read. Only used if --umi-preset is 'custom'. Default: 0")
    optional_move_trim.add_argument("--umi-end", type=int, default=12,
                                     help="0-indexed end position (exclusive) of the UMI in the specified read. Only used if --umi-preset is 'custom'. Default: 12")
    optional_move_trim.add_argument("--collapse", action="store_true", 
                                          help="Collapse reads by UMI (keep one read per UMI).")
    optional_move_trim.add_argument("--trim", action="store_true", default=False,
                                          help="Trim the non-biological region (first 19bp) from Read 2. Default: False (do not trim)")
    optional_move_trim.add_argument("--output-prefix", help="Prefix for output FASTQ files (e.g., 'my_sample_output'). "
                                                                   "Files will be named <prefix>_R1.fastq and <prefix>_R2.fastq. "
                                                                   "If not provided, a default prefix based on input will be used.")
    parser_move_trim_collapse.set_defaults(func=move_umi_and_trim_main)

    # New subparser for inspecting UMI variations
    parser_inspect_umi = subparsers.add_parser(
        "inspect_umi_variations",
        help="Inspect detailed sequence variations for specific UMIs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    required_inspect = parser_inspect_umi.add_argument_group('Required Arguments')
    required_inspect.add_argument("-f1", required=True, help="Path to gzipped R1 FASTQ file.")
    required_inspect.add_argument("-f2", required=True, help="Path to gzipped R2 FASTQ file.")
    required_inspect.add_argument("--inspect-umis", nargs='+', required=True, 
                                  help="Space-separated list of UMI sequences to inspect (e.g., AAAAAA BBBBBB).")

    optional_inspect = parser_inspect_umi.add_argument_group('Optional Arguments')
    optional_inspect.add_argument("--umi-preset", default="custom", choices=["custom", "takara_tcr"],
                                     help="""Predefined UMI location preset.
'takara_tcr': Sets UMI to R2, start 0, end 12.
'custom': Requires --umi-read, --umi-start, --umi-end to be specified.
Default: custom""")
    optional_inspect.add_argument("--umi-read", default="R2", choices=["R1", "R2"],
                                     help="Specify which read contains the UMI (R1 or R2). Only used if --umi-preset is 'custom'. Default: R2")
    optional_inspect.add_argument("--umi-start", type=int, default=0,
                                     help="0-indexed start position of the UMI in the specified read. Only used if --umi-preset is 'custom'. Default: 0")
    optional_inspect.add_argument("--umi-end", type=int, default=12,
                                     help="0-indexed end position (exclusive) of the UMI in the specified read. Only used if --umi-preset is 'custom'. Default: 12")
    parser_inspect_umi.set_defaults(func=inspect_umi_variations_main)


    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
