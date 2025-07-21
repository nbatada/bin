#!/usr/bin/env python3
import argparse
import gzip
import sys
import re

def _open_file(filepath, mode='r'):
    if filepath == '-':
        if 'r' in mode:
            return sys.stdin
        else:
            return sys.stdout
    elif filepath.endswith('.gz'):
        return gzip.open(filepath, mode + 't')
    else:
        return open(filepath, mode)

def unwrap_fasta(input_file, output_file):
    current_header = ""
    current_sequence = []
    for line in input_file:
        line = line.strip()
        if line.startswith('>'):
            if current_header:
                output_file.write(f"{current_header}\t{''.join(current_sequence)}\n")
            current_header = line
            current_sequence = []
        else:
            current_sequence.append(line)
    if current_header:
        output_file.write(f"{current_header}\t{''.join(current_sequence)}\n")

def wrap_fastq(input_file, output_file):
    for line in input_file:
        line = line.strip()
        if line:
            parts = line.split('\t', 1)
            if len(parts) == 2:
                header, sequence = parts
                output_file.write(f"{header}\n")
                output_file.write(f"{sequence}\n")

def length_fasta(input_file, output_file):
    current_header = ""
    current_sequence = []
    for line in input_file:
        line = line.strip()
        if line.startswith('>'):
            if current_header:
                output_file.write(f"{current_header}\t{len(''.join(current_sequence))}\n")
            current_header = line
            current_sequence = []
        else:
            current_sequence.append(line)
    if current_header:
        output_file.write(f"{current_header}\t{len(''.join(current_sequence))}\n")

def trim_fasta(input_file, output_file, from_left, from_right, add_trimmed_to_name):
    current_header = ""
    current_sequence = []
    for line in input_file:
        line = line.strip()
        if line.startswith('>'):
            if current_header:
                sequence_str = ''.join(current_sequence)
                trimmed_seq = sequence_str[from_left:len(sequence_str) - from_right]
                if add_trimmed_to_name:
                    output_file.write(f"{current_header}_{trimmed_seq}\n")
                else:
                    output_file.write(f"{current_header}\n")
                output_file.write(f"{trimmed_seq}\n")
            current_header = line
            current_sequence = []
        else:
            current_sequence.append(line)
    if current_header:
        sequence_str = ''.join(current_sequence)
        trimmed_seq = sequence_str[from_left:len(sequence_str) - from_right]
        if add_trimmed_to_name:
            output_file.write(f"{current_header}_{trimmed_seq}\n")
        else:
            output_file.write(f"{current_header}\n")
        output_file.write(f"{trimmed_seq}\n")

def import_genbank(input_file, output_file):
    full_sequence = []
    for line in input_file:
        line = line.strip()
        # Pattern to remove leading spaces, line number, and then spaces within the sequence
        # It captures the sequence part (letters and spaces) after the number
        match = re.match(r'^\s*\d+\s+([a-zA-Z\s]+)$', line)
        if match:
            # Remove spaces from the captured sequence part and append
            full_sequence.append(match.group(1).replace(' ', ''))
    
    if full_sequence:
        output_file.write(">seq\n")
        output_file.write("".join(full_sequence) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="A utility for FASTA/FASTQ file manipulation.",
        usage="fasta.py <command> [options]",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-v', '--version', action='version', version='%(prog)s 0.1.0',
        help="show program's version number and exit"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    unwrap_parser = subparsers.add_parser(
        'unwrap',
        help='Convert multi-line FASTA to two-column TSV.',
        description='Convert multi-line FASTA to two-column TSV.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    unwrap_parser.add_argument(
        '-f', '--file',
        required=True,
        help="Input FASTA file (or '-' for stdin). Supports .gz."
    )
    unwrap_parser.add_argument(
        '-o', '--output',
        default='-',
        help="Output TSV file (or '-' for stdout). Default: stdout."
    )
    unwrap_parser.set_defaults(func=lambda args: unwrap_fasta(_open_file(args.file, 'r'), _open_file(args.output, 'w')))

    wrap_parser = subparsers.add_parser(
        'wrap',
        help='Convert two-column TSV to FASTA format.',
        description='Convert two-column TSV to FASTA format.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    wrap_parser.add_argument(
        '-f', '--file',
        required=True,
        help="Input TSV file (or '-' for stdin). Supports .gz."
    )
    wrap_parser.add_argument(
        '-o', '--output',
        default='-',
        help="Output FASTA file (or '-' for stdout). Default: stdout."
    )
    wrap_parser.set_defaults(func=lambda args: wrap_fastq(_open_file(args.file, 'r'), _open_file(args.output, 'w')))

    length_parser = subparsers.add_parser(
        'length',
        help='Calculate sequence lengths from FASTA.',
        description='Calculate sequence lengths from FASTA and output as TSV.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    length_parser.add_argument(
        '-f', '--file',
        required=True,
        help="Input FASTA file (or '-' for stdin). Supports .gz."
    )
    length_parser.add_argument(
        '-o', '--output',
        default='-',
        help="Output TSV file (or '-' for stdout). Default: stdout."
    )
    length_parser.set_defaults(func=lambda args: length_fasta(_open_file(args.file, 'r'), _open_file(args.output, 'w')))

    trim_parser = subparsers.add_parser(
        'trim',
        help='Trim sequences from FASTA file.',
        description='Trim sequences from FASTA file based on specified positions.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    trim_parser.add_argument(
        '-f', '--file',
        required=True,
        help="Input FASTA file (or '-' for stdin). Supports .gz."
    )
    trim_parser.add_argument(
        '-o', '--output',
        default='-',
        help="Output FASTA file (or '-' for stdout). Default: stdout."
    )
    trim_parser.add_argument(
        '--from_left',
        type=int,
        default=0,
        help="Number of bases to trim from the left (start). Default: 0."
    )
    trim_parser.add_argument(
        '--from_right',
        type=int,
        default=0,
        help="Number of bases to trim from the right (end). Default: 0."
    )
    trim_parser.add_argument(
        '--add_trimmed_to_name',
        action='store_true',
        help="Append the trimmed sequence to the FASTA header."
    )
    trim_parser.set_defaults(func=lambda args: trim_fasta(
        _open_file(args.file, 'r'),
        _open_file(args.output, 'w'),
        args.from_left,
        args.from_right,
        args.add_trimmed_to_name
    ))
    
    import_genbank_parser = subparsers.add_parser(
        'import_genbank',
        help='Convert GenBank sequence format to FASTA.',
        description='Extracts sequence from GenBank-like format and outputs as FASTA.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    import_genbank_parser.add_argument(
        '-f', '--file',
        required=True,
        help="Input GenBank sequence file (or '-' for stdin). Supports .gz."
    )
    import_genbank_parser.add_argument(
        '-o', '--output',
        default='-',
        help="Output FASTA file (or '-' for stdout). Default: stdout."
    )
    import_genbank_parser.set_defaults(func=lambda args: import_genbank(_open_file(args.file, 'r'), _open_file(args.output, 'w')))

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
