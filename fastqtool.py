#!/usr/bin/env python3
# Updated: 27-July-2025
import os
import sys
import re
import argparse
import pysam
import numpy as np
from collections import defaultdict, Counter
import random # For downsample

# --------------------------
# Utility Functions
# --------------------------
def _print_verbose(args, message):
    if args.verbose:
        sys.stderr.write(f"VERBOSE: {message}\n")

def _parse_umi_location(umi_preset, umi_read_arg, umi_start_arg, umi_end_arg):
    if umi_preset == "takara_tcr":
        read_idx = 1
        start = 0
        end = 12
    else:
        read_idx = 0 if umi_read_arg.upper() == 'R1' else 1
        start = umi_start_arg
        end = umi_end_arg
    return read_idx, start, end

# --------------------------
# Operation Handler Functions
# --------------------------
def _handle_rename_filenames(args):
    _print_verbose(args, "Starting rename_filenames operation.")

    file_extension = ".fastq"
    if not args.not_gz:
        file_extension += ".gz"
    _print_verbose(args, f"Using target file extension: {file_extension}")

    sample_map = {}
    try:
        with open(args.map_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    sys.stderr.write(f"Error: Map file line '{line.strip()}' is not in 'old_prefix new_sample_id' format.\n")
                    sys.exit(1)
                old_prefix, new_sample_id = parts
                sample_map[old_prefix] = new_sample_id
        _print_verbose(args, f"Loaded {len(sample_map)} entries from map file: {args.map_file}")
    except FileNotFoundError:
        sys.stderr.write(f"Error: Map file '{args.map_file}' not found.\n")
        sys.exit(1)

    prefix_re = None
    read_part_re = None
    if args.prefix_regex:
        try:
            prefix_re = re.compile(args.prefix_regex)
            _print_verbose(args, f"Compiled prefix regex: '{args.prefix_regex}'")
        except re.error as e:
            sys.stderr.write(f"Error: Invalid prefix_regex provided. {e}\n")
            sys.exit(1)
    if args.read_part_regex:
        try:
            read_part_re = re.compile(args.read_part_regex)
            _print_verbose(args, f"Compiled read_part regex: '{args.read_part_regex}'")
        except re.error as e:
            sys.stderr.write(f"Error: Invalid read_part_regex provided. {e}\n")
            sys.exit(1)

    try:
        all_fastq_files = [f for f in os.listdir(args.fastq_dir) if f.endswith(file_extension)]
        all_fastq_files.sort()
        _print_verbose(args, f"Found {len(all_fastq_files)} fastq files in '{args.fastq_dir}'")
    except FileNotFoundError:
        sys.stderr.write(f"Error: Fastq directory '{args.fastq_dir}' not found.\n")
        sys.exit(1)
    except NotADirectoryError:
        sys.stderr.write(f"Error: '{args.fastq_dir}' is not a directory.\n")
        sys.exit(1)

    mappings_to_perform = []
    sorted_map_prefixes = sorted(sample_map.keys(), key=len, reverse=True)

    for original_file in all_fastq_files:
        prefix_key = None
        r_part = None

        if prefix_re and read_part_re:
            prefix_match = prefix_re.search(original_file)
            read_part_match = read_part_re.search(original_file)

            if prefix_match:
                prefix_key = prefix_match.group(1)
            if read_part_match:
                r_part = read_part_match.group(1)
        else:
            for mp_prefix in sorted_map_prefixes:
                if original_file.startswith(mp_prefix):
                    prefix_key = mp_prefix
                    remaining_part = original_file[len(mp_prefix):]
                    r_match = re.search(r'(R[12])', remaining_part)
                    if r_match:
                        r_part = r_match.group(1)
                    break

        if not prefix_key:
            _print_verbose(args, f"Warning: Could not determine prefix for '{original_file}'. Skipping.")
            continue
        if not r_part:
            _print_verbose(args, f"Warning: Could not determine R1/R2 part for '{original_file}'. Skipping.")
            continue

        if prefix_key in sample_map:
            new_sample_id = sample_map[prefix_key]
            old_filepath = os.path.join(args.fastq_dir, original_file)
            new_filename = f"{new_sample_id}_{r_part}{file_extension}"
            mappings_to_perform.append((old_filepath, new_filename))
            _print_verbose(args, f"Mapped '{original_file}' to '{new_filename}'")
        else:
            _print_verbose(args, f"Warning: Inferred prefix '{prefix_key}' from '{original_file}' not found in map file. Skipping.")

    if not mappings_to_perform:
        sys.stderr.write("No valid mappings found to perform.\n")
        return

    if args.no_inspection:
        sys.stderr.write("Creating symbolic links...\n")
        for old_path, new_name in mappings_to_perform:
            try:
                os.symlink(old_path, new_name)
                sys.stderr.write(f"Linked: {old_path} -> {new_name}\n")
            except FileExistsError:
                sys.stderr.write(f"Warning: Link '{new_name}' already exists. Skipping.\n")
            except OSError as e:
                sys.stderr.write(f"Error creating link for '{old_path}' to '{new_name}': {e}\n")
    else:
        sys.stdout.write("Proposed mappings (run with --no-inspection to create symlinks):\n")
        for old_path, new_name in mappings_to_perform:
            sys.stdout.write(f"  {old_path} -> {new_name}\n")


def _handle_umi_count(args):
    _print_verbose(args, "Starting umi_count operation.")
    umi_read_idx, umi_start, umi_end = _parse_umi_location(args.umi_preset, args.umi_read, args.umi_start, args.umi_end)
    _print_verbose(args, f"UMI location: read_idx={umi_read_idx}, start={umi_start}, end={umi_end}")

    sample_name = os.path.basename(args.f1).replace('_R1.fastq.gz', '').replace('_R1.fastq', '')
    _print_verbose(args, f"Sample name: {sample_name}")

    umi_grouped_sequences = defaultdict(list)
    total_reads = 0
    reads_without_umi_extracted = 0

    try:
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

                if umi != "UMI_NOT_EXTRACTED":
                    r1_seq_bio = r1_read.sequence[5:] if len(r1_read.sequence) >= 5 else ""
                    r2_seq_bio = r2_read.sequence[19:] if len(r2_read.sequence) >= 19 else ""
                    umi_grouped_sequences[umi].append((r1_seq_bio, r2_seq_bio))
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: Input FASTQ file not found: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error reading FASTQ files: {e}\n")
        sys.exit(1)
            
    umi_counts = {umi: len(seq_list) for umi, seq_list in umi_grouped_sequences.items()}
    umi_total = len(umi_counts)
    
    umi_counts_list = [count for umi, count in umi_counts.items() if umi != "UMI_NOT_EXTRACTED"]
    median_reads_per_umi = np.median(umi_counts_list) if umi_counts_list else 0

    sys.stdout.write(f"# sample_id\t{sample_name}\n")
    sys.stdout.write(f"# reads_total\t{total_reads}\n")
    sys.stdout.write(f"# reads_left\t{umi_total}\n")
    reads_left_pct = (umi_total / total_reads * 100) if total_reads > 0 else 0
    sys.stdout.write(f"# reads_left_pct\t{reads_left_pct:.2f}\n")
    sys.stdout.write(f"# umi_total\t{umi_total}\n")
    
    if umi_total > 0:
        avg_reads_per_umi = total_reads / umi_total
        sys.stdout.write(f"# reads_per_umi\t{avg_reads_per_umi:.2f}\n")
    else:
        sys.stdout.write(f"# reads_per_umi\tN/A\n")

    sys.stdout.write("umi\tcount\tpct\n")
    
    sorted_umi_counts = sorted(umi_counts.items(), key=lambda item: item[1], reverse=True)
    for umi, count in sorted_umi_counts:
        if umi != "UMI_NOT_EXTRACTED":
            pct = (count / umi_total * 100) if umi_total > 0 else 0
            sys.stdout.write(f"{umi}\t{count}\t{pct:.2f}\n")


def _handle_collapse_umi(args):
    _print_verbose(args, "Starting collapse_umi operation.")
    umi_read_idx, umi_start, umi_end = _parse_umi_location(args.umi_preset, args.umi_read, args.umi_start, args.umi_end)
    _print_verbose(args, f"UMI location: read_idx={umi_read_idx}, start={umi_start}, end={umi_end}")

    sample_name = os.path.basename(args.f1).replace('_R1.fastq.gz', '').replace('_R1.fastq', '')
    _print_verbose(args, f"Sample name: {sample_name}")

    output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    _print_verbose(args, f"Default output directory: {output_dir}")

    base_filename = os.path.basename(args.f1).replace('_R1.fastq.gz', '').replace('_R1.fastq', '')
    
    if args.output_prefix:
        prefix_dir = os.path.dirname(os.path.abspath(args.output_prefix))
        if not prefix_dir:
            prefix_dir = output_dir
        os.makedirs(prefix_dir, exist_ok=True)
        final_output_prefix_base = os.path.basename(args.output_prefix)
        output_r1_path = os.path.join(prefix_dir, f"{final_output_prefix_base}_R1.fastq")
        output_r2_path = os.path.join(prefix_dir, f"{final_output_prefix_base}_R2.fastq")
        _print_verbose(args, f"Using custom output prefix: {args.output_prefix}")
    else:
        output_r1_path = os.path.join(output_dir, f"{base_filename}_umi_R1.fastq")
        output_r2_path = os.path.join(output_dir, f"{base_filename}_umi_R2.fastq")
        _print_verbose(args, "Output filenames for collapse mode (default).")

    total_input_reads_move_trim = 0
    total_output_reads_move_trim = 0

    _print_verbose(args, f"Output R1 path: {output_r1_path}")
    _print_verbose(args, f"Output R2 path: {output_r2_path}")

    try:
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

                if not args.do_not_trim_umi:
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
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: Input FASTQ file not found: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error during FASTQ processing: {e}\n")
        sys.exit(1)

    sys.stdout.write(f"# sample_id\t{sample_name}\n")
    sys.stdout.write(f"# read.total_input\t{total_input_reads_move_trim}\n")
    sys.stdout.write(f"# read.total_output\t{total_output_reads_move_trim}\n")
    if total_input_reads_move_trim > 0:
        output_percent = (total_output_reads_move_trim / total_input_reads_move_trim) * 100
        sys.stdout.write(f"# read.output_percent_of_input\t{output_percent:.2f}\n")
    else:
        sys.stdout.write(f"# read.output_percent_of_input\tN/A\n")
    
    sys.stdout.write(f"# processed_files\t{args.f1},{args.f2}\n")
    sys.stdout.write(f"# output_files\t{output_r1_path},{output_r2_path}\n")

def _handle_count_reads(args):
    _print_verbose(args, "Starting count_reads operation.")
    sys.stdout.write("file\tread_count\n")
    for fastq_file in args.files:
        total_reads = 0
        try:
            with pysam.FastxFile(fastq_file) as f:
                for _ in f:
                    total_reads += 1
            sys.stdout.write(f"{fastq_file}\t{total_reads}\n")
            _print_verbose(args, f"Counted {total_reads} reads in {fastq_file}")
        except FileNotFoundError:
            sys.stderr.write(f"Error: File '{fastq_file}' not found. Skipping.\n")
        except Exception as e:
            sys.stderr.write(f"Error processing '{fastq_file}': {e}\n")

def _handle_trim_adapters(args):
    _print_verbose(args, "Starting trim_adapters operation.")

    adapters = {}
    if args.preset == "truseq":
        adapters['R1'] = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
        adapters['R2'] = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"
        _print_verbose(args, "Using TruSeq adapter preset.")
    elif args.preset == "nextera":
        adapters['R1'] = "CTGTCTCTTATACACATCT"
        adapters['R2'] = "CTGTCTCTTATACACATCT"
        _print_verbose(args, "Using Nextera adapter preset.")
    elif args.adapter_r1:
        adapters['R1'] = args.adapter_r1
        if args.adapter_r2:
            adapters['R2'] = args.adapter_r2
        else:
            adapters['R2'] = args.adapter_r1 # Assume same for R2 if not specified
        _print_verbose(args, f"Using custom adapters: R1={adapters['R1']}, R2={adapters.get('R2', 'N/A')}")
    else:
        sys.stderr.write("Error: No adapter preset or custom adapter sequences provided.\n")
        sys.exit(1)

    output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    _print_verbose(args, f"Output directory: {output_dir}")

    base_filename_r1 = os.path.basename(args.f1).replace('.fastq.gz', '').replace('.fastq', '')
    base_filename_r2 = os.path.basename(args.f2).replace('.fastq.gz', '').replace('.fastq', '')

    output_r1_path = os.path.join(output_dir, f"{base_filename_r1}_trimmed.fastq")
    output_r2_path = os.path.join(output_dir, f"{base_filename_r2}_trimmed.fastq")

    _print_verbose(args, f"Output R1 path: {output_r1_path}")
    _print_verbose(args, f"Output R2 path: {output_r2_path}")

    try:
        with pysam.FastxFile(args.f1) as r1_in, \
             pysam.FastxFile(args.f2) as r2_in, \
             open(output_r1_path, 'w') as r1_out, \
             open(output_r2_path, 'w') as r2_out:
            for r1_read, r2_read in zip(r1_in, r2_in):
                # Trim R1
                trimmed_r1_seq = r1_read.sequence
                trimmed_r1_qual = r1_read.quality
                if 'R1' in adapters:
                    adapter_pos = trimmed_r1_seq.find(adapters['R1'])
                    if adapter_pos != -1:
                        trimmed_r1_seq = trimmed_r1_seq[:adapter_pos]
                        trimmed_r1_qual = trimmed_r1_qual[:adapter_pos]
                        _print_verbose(args, f"Trimmed R1: {r1_read.name} at {adapter_pos}")

                # Trim R2
                trimmed_r2_seq = r2_read.sequence
                trimmed_r2_qual = r2_read.quality
                if 'R2' in adapters:
                    adapter_pos = trimmed_r2_seq.find(adapters['R2'])
                    if adapter_pos != -1:
                        trimmed_r2_seq = trimmed_r2_seq[:adapter_pos]
                        trimmed_r2_qual = trimmed_r2_qual[:adapter_pos]
                        _print_verbose(args, f"Trimmed R2: {r2_read.name} at {adapter_pos}")

                r1_out.write(f"@{r1_read.name}\n{trimmed_r1_seq}\n+\n{trimmed_r1_qual}\n")
                r2_out.write(f"@{r2_read.name}\n{trimmed_r2_seq}\n+\n{trimmed_r2_qual}\n")
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: Input FASTQ file not found: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error during FASTQ processing: {e}\n")
        sys.exit(1)
    sys.stdout.write(f"Adapter trimming complete. Output files: {output_r1_path}, {output_r2_path}\n")

def _handle_downsample(args):
    _print_verbose(args, "Starting downsample operation.")

    if args.num_reads is None and args.percentage is None:
        sys.stderr.write("Error: Either --num-reads or --percentage must be specified for downsample.\n")
        sys.exit(1)
    if args.num_reads is not None and args.percentage is not None:
        sys.stderr.write("Error: Cannot specify both --num-reads and --percentage for downsample.\n")
        sys.exit(1)

    input_files = args.files
    output_files = []

    for i in range(0, len(input_files), 2):
        f1_path = input_files[i]
        f2_path = input_files[i+1] if (i+1) < len(input_files) else None

        base_filename_r1 = os.path.basename(f1_path).replace('.fastq.gz', '').replace('.fastq', '')
        output_r1_path = os.path.join(os.getcwd(), f"{base_filename_r1}_downsampled.fastq")
        output_files.append(output_r1_path)

        if f2_path:
            base_filename_r2 = os.path.basename(f2_path).replace('.fastq.gz', '').replace('.fastq', '')
            output_r2_path = os.path.join(os.getcwd(), f"{base_filename_r2}_downsampled.fastq")
            output_files.append(output_r2_path)
        else:
            output_r2_path = None

        _print_verbose(args, f"Processing {f1_path}" + (f" and {f2_path}" if f2_path else ""))

        try:
            total_reads = 0
            with pysam.FastxFile(f1_path) as f:
                for _ in f:
                    total_reads += 1
            _print_verbose(args, f"Total reads in {f1_path}: {total_reads}")

            target_reads = 0
            if args.num_reads is not None:
                target_reads = min(args.num_reads, total_reads)
            elif args.percentage is not None:
                target_reads = int(total_reads * (args.percentage / 100.0))
            _print_verbose(args, f"Target reads for downsampling: {target_reads}")

            if target_reads == 0:
                sys.stderr.write(f"Warning: No reads to output for {f1_path} based on downsample criteria. Skipping.\n")
                continue

            # Reservoir sampling for efficient downsampling
            sampled_reads_r1 = []
            sampled_reads_r2 = [] if f2_path else None

            with pysam.FastxFile(f1_path) as r1_in, \
                 (pysam.FastxFile(f2_path) if f2_path else None) as r2_in:
                for i, (r1_read, r2_read) in enumerate(zip(r1_in, r2_in or [None]*total_reads)):
                    if i < target_reads:
                        sampled_reads_r1.append(r1_read)
                        if f2_path:
                            sampled_reads_r2.append(r2_read)
                    else:
                        r = random.randint(0, i)
                        if r < target_reads:
                            sampled_reads_r1[r] = r1_read
                            if f2_path:
                                sampled_reads_r2[r] = r2_read
            
            with open(output_r1_path, 'w') as r1_out:
                for read in sampled_reads_r1:
                    r1_out.write(f"@{read.name}\n{read.sequence}\n+\n{read.quality}\n")
            _print_verbose(args, f"Wrote {len(sampled_reads_r1)} reads to {output_r1_path}")

            if f2_path:
                with open(output_r2_path, 'w') as r2_out:
                    for read in sampled_reads_r2:
                        r2_out.write(f"@{read.name}\n{read.sequence}\n+\n{read.quality}\n")
                _print_verbose(args, f"Wrote {len(sampled_reads_r2)} reads to {output_r2_path}")

        except FileNotFoundError as e:
            sys.stderr.write(f"Error: Input FASTQ file not found: {e}. Skipping pair.\n")
        except Exception as e:
            sys.stderr.write(f"Error during downsampling of {f1_path}: {e}. Skipping pair.\n")
    
    sys.stdout.write(f"Downsampling complete. Output files: {', '.join(output_files)}\n")


# --------------------------
# Argument Parser Setup
# --------------------------
def _setup_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "A versatile tool for manipulating fastq files.\n\n"
            "Usage:\n"
            "    fastqtool.py [GLOBAL_OPTIONS] <command> [COMMAND_SPECIFIC_OPTIONS]\n\n"
            "Global Options affect how input is read and overall behavior.\n"
            "Commands perform specific fastq manipulations.\n"
            "Use 'fastqtool.py <command> --help' for details on each."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    global_options_args = [
        (("-d", "--fastq_dir"), {"default": ".", "help": "Directory containing the original fastq files (default: current directory)."}),
        (("--not-gz",), {"action": "store_true", "help": "Do not assume .gz extension for new filenames (i.e., use .fastq instead of .fastq.gz)."}),
        (("--verbose",), {"action": "store_true", "help": "Enable verbose debug output to stderr."}),
    ]

    global_options_group = parser.add_argument_group("Global Options")
    for args, kwargs in global_options_args:
        global_options_group.add_argument(*args, **kwargs)


    subparsers = parser.add_subparsers(dest="operation", help="Available commands. Use 'fastqtool.py <command> --help' for details.")

    rename_parser = subparsers.add_parser(
        'rename_filenames',
        help='Rename or create symlinks for fastq files.',
        description='This command reads fastq files from a specified directory, extracts prefixes and read parts using regexes, and maps them to new sample IDs from a provided map file. It then either prints the proposed mappings or creates symbolic links.'
    )
    for args, kwargs in global_options_args:
        rename_parser.add_argument(*args, **kwargs)
    required_args_rename = rename_parser.add_argument_group("Required Arguments")
    required_args_rename.add_argument('-m', '--map_file', required=True, help='Path to a two-column map file (old_prefix new_sample_id).')
    optional_args_rename = rename_parser.add_argument_group("Optional Arguments")
    optional_args_rename.add_argument('--prefix_regex', help='Regular expression with a capturing group to extract the unique prefix from input filenames (e.g., "(SAM\\d+_LIB\\d+)"). If not provided, the script will attempt to infer the prefix from the map file.')
    optional_args_rename.add_argument('--read_part_regex', help='Regular expression with a capturing group to extract the read part (e.g., R1, R2) from input filenames (e.g., "(R[12])"). If not provided, the script will attempt to infer R1/R2 from the filename.')
    optional_args_rename.add_argument('--no-inspection', action="store_true", help='If set, proceed to create symbolic links without printing proposed mappings.')


    umi_common_args = [
        (("--umi-preset",), {"default": "takara_tcr", "choices": ["custom", "takara_tcr"],
                          "help": """Predefined UMI location preset.
'takara_tcr': Sets UMI to R2, start 0, end 12.
'custom': Requires --umi-read, --umi-start, --umi-end to be specified.
Default: takara_tcr"""}),
        (("--umi-read",), {"default": "R2", "choices": ["R1", "R2"],
                        "help": "Specify which read contains the UMI (R1 or R2). Only used if --umi-preset is 'custom'. Default: R2"}),
        (("--umi-start",), {"type": int, "default": 0,
                         "help": "0-indexed start position of the UMI in the specified read. Only used if --umi-preset is 'custom'. Default: 0"}),
        (("--umi-end",), {"type": int, "default": 12,
                       "help": "0-indexed end position (exclusive) of the UMI in the specified read. Only used if --umi-preset is 'custom'. Default: 12"}),
    ]

    umi_count_parser = subparsers.add_parser(
        "umi_count",
        help="Count unique UMIs and their occurrences.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    required_args_count = umi_count_parser.add_argument_group('Required Arguments')
    required_args_count.add_argument("-f1", required=True, help="Path to gzipped R1 FASTQ file.")
    required_args_count.add_argument("-f2", required=True, help="Path to gzipped R2 FASTQ file.")
    optional_args_count = umi_count_parser.add_argument_group('Optional Arguments')
    for args, kwargs in umi_common_args:
        optional_args_count.add_argument(*args, **kwargs)
    umi_count_parser.set_defaults(func=_handle_umi_count)

    collapse_umi_parser = subparsers.add_parser(
        "collapse_umi",
        help="Extract UMI, move to read name, trim R2, and collapse reads.",
        description="This command processes paired-end FASTQ files to extract UMIs, move them to read names, trim non-biological regions from Read 2 by default, and collapse reads with identical UMIs and biological sequences.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    required_args_collapse_umi = collapse_umi_parser.add_argument_group('Required Arguments')
    required_args_collapse_umi.add_argument("-f1", required=True, help="Path to gzipped R1 FASTQ file.")
    required_args_collapse_umi.add_argument("-f2", required=True, help="Path to gzipped R2 FASTQ file.")
    optional_args_collapse_umi = collapse_umi_parser.add_argument_group('Optional Arguments')
    for args, kwargs in umi_common_args:
        optional_args_collapse_umi.add_argument(*args, **kwargs)
    optional_args_collapse_umi.add_argument("--do-not-trim-umi", action="store_true", help="If set, do NOT trim the non-biological region (first 19bp) from Read 2. By default, trimming is performed.")
    optional_args_collapse_umi.add_argument("--output-prefix", help="Prefix for output FASTQ files (e.g., 'my_sample_output'). Files will be named <prefix>_R1.fastq and <prefix>_R2.fastq. If not provided, a default prefix based on input will be used.")
    collapse_umi_parser.set_defaults(func=_handle_collapse_umi)

    # Subparser for 'count_reads'
    count_reads_parser = subparsers.add_parser(
        'count_reads',
        help='Count reads in one or more FASTQ files.',
        description='This command counts the total number of reads in the provided FASTQ files.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    required_args_count_reads = count_reads_parser.add_argument_group('Required Arguments')
    required_args_count_reads.add_argument('files', nargs='+', help='One or more FASTQ files (e.g., file1.fastq.gz file2.fastq.gz).')
    count_reads_parser.set_defaults(func=_handle_count_reads)

    # Subparser for 'trim_adapters'
    trim_adapters_parser = subparsers.add_parser(
        'trim_adapters',
        help='Trim adapter sequences from FASTQ files.',
        description='This command trims common Illumina adapter sequences or custom sequences from paired-end FASTQ files.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    required_args_trim_adapters = trim_adapters_parser.add_argument_group('Required Arguments')
    required_args_trim_adapters.add_argument("-f1", required=True, help="Path to gzipped R1 FASTQ file.")
    required_args_trim_adapters.add_argument("-f2", required=True, help="Path to gzipped R2 FASTQ file.")
    
    adapter_group = trim_adapters_parser.add_mutually_exclusive_group(required=True)
    adapter_group.add_argument("--preset", choices=["truseq", "nextera"],
                               help="Use predefined adapter sequences for 'truseq' or 'nextera' kits.")
    adapter_group.add_argument("--adapter-r1", help="Custom adapter sequence for Read 1.")
    optional_args_trim_adapters = trim_adapters_parser.add_argument_group('Optional Arguments')
    optional_args_trim_adapters.add_argument("--adapter-r2", help="Custom adapter sequence for Read 2 (defaults to --adapter-r1 if not provided with --adapter-r1).")
    trim_adapters_parser.set_defaults(func=_handle_trim_adapters)

    # Subparser for 'downsample'
    downsample_parser = subparsers.add_parser(
        'downsample',
        help='Downsample FASTQ files to a specified number or percentage of reads.',
        description='This command creates a subset of FASTQ files by randomly selecting a specified number or percentage of reads.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    required_args_downsample = downsample_parser.add_argument_group('Required Arguments')
    required_args_downsample.add_argument('files', nargs='+', help='One or more FASTQ files (e.g., R1.fastq.gz R2.fastq.gz). Paired-end files should be provided as pairs (e.g., R1.fq R2.fq R1_sample2.fq R2_sample2.fq).')
    
    downsample_group = downsample_parser.add_mutually_exclusive_group(required=True)
    downsample_group.add_argument("--num-reads", type=int, help="Number of reads to downsample to.")
    downsample_group.add_argument("--percentage", type=float, help="Percentage of reads to keep (e.g., 10.0 for 10%%).")
    downsample_parser.set_defaults(func=_handle_downsample)


    return parser

# --------------------------
# Dispatch Table
# --------------------------
OPERATION_HANDLERS = {
    "rename_filenames": _handle_rename_filenames,
    "umi_count": _handle_umi_count,
    "collapse_umi": _handle_collapse_umi,
    "count_reads": _handle_count_reads,
    "trim_adapters": _handle_trim_adapters,
    "downsample": _handle_downsample,
}

# --------------------------
# Main Function
# --------------------------
def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()

    if not args.operation:
        parser.print_help()
        sys.exit(0)

    handler = OPERATION_HANDLERS.get(args.operation)
    if handler is None:
        sys.stderr.write(f"Error: Unknown command '{args.operation}'.\n")
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except Exception as e:
        _print_verbose(args, f"An unexpected error occurred: {e}")
        sys.stderr.write(f"An error occurred during '{args.operation}' command execution: {e}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
