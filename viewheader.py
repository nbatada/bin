#!/usr/bin/env python3
import argparse
import sys

def parse_input_and_display(input_source, sep='\t'):
    header_line = input_source.readline().strip()
    data_line = input_source.readline().strip()

    headers = header_line.split(sep)
    data_entries = data_line.split(sep)

    for i, (header, entry) in enumerate(zip(headers, data_entries)):
        if len(entry.strip()) > 40:
            entry = entry[:40] + '...[truncated]'
        print(f"{i}: {header.strip()}: {entry.strip()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Path to the input file')
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r') as f:
            parse_input_and_display(f)
    else:
        parse_input_and_display(sys.stdin)
