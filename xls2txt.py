#!/usr/bin/env python3

import pandas as pd
import sys
import os

def xls_to_txt(file_path):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        # Write to standard output as tab-separated values
        df.to_csv(sys.stdout, sep='\t', index=False)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} <file_path>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    xls_to_txt(file_path)

