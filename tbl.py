#!/usr/bin/env python3
import sys
import argparse
import re
import pandas as pd
import codecs # For handling escape sequences
from io import StringIO # For handling piped input with pandas
from collections import Counter

CHUNK_SIZE = 10000 # Default chunk size for low-memory processing

def _parse_column_arg(value, df_columns, is_header_present, arg_name="column"):
    """
    Parses a column argument, converting 1-indexed to 0-indexed, or resolving column name.
    Returns the 0-indexed integer position.
    """
    try:
        # If it's a number, assume 1-indexed and convert to 0-indexed
        col_idx = int(value)
        if col_idx < 1:
            raise ValueError(f"Error: {arg_name} index '{value}' must be 1 or greater (1-indexed).")
        # Check if 0-indexed is out of bounds, allowing to-col to be one past for insertion
        if col_idx - 1 > len(df_columns):
            raise IndexError(f"Error: {arg_name} index '{value}' (1-indexed) is out of bounds. Max column index is {len(df_columns)}.")
        return col_idx - 1
    except ValueError as e:
        # If not a number, assume it's a column name (only if header is present)
        if not is_header_present:
            raise ValueError(f"Error: Cannot use column name '{value}' for {arg_name} when no header is present (--header=None). Use 1-indexed integer.")
        if value not in df_columns:
            raise ValueError(f"Error: Column '{value}' not found in header for {arg_name}. Available columns: {list(df_columns)}.")
        return df_columns.get_loc(value) # Get 0-indexed position by name
    except IndexError as e:
        raise e # Re-raise if it's an IndexError from bounds check

def _parse_multiple_columns_arg(values, df_columns, is_header_present, arg_name="columns"):
    """
    Parses a comma-separated string of column indices/names, returning a list of 0-indexed integers.
    """
    if values.lower() == "all":
        return list(range(len(df_columns)))
    
    col_indices = []
    for val in values.split(','):
        val = val.strip()
        if not val:
            continue
        try:
            col_indices.append(_parse_column_arg(val, df_columns, is_header_present, arg_name))
        except (ValueError, IndexError) as e:
            raise type(e)(f"Error parsing {arg_name} '{values}': {e}") # Re-raise with context
    return col_indices


def _clean_string_for_header_and_data(s):
    """Applies cleanup rules to a string."""
    if not isinstance(s, str):
        return s # Return as is if not a string (e.g., NaN, None)
    s = s.lower()
    s = s.replace(' ', '_')
    s = re.sub(r'[^\w_]', '', s) # Remove non-alphanumeric or underscore characters
    s = re.sub(r'_{2,}', '_', s) # Squeeze multiple underscores
    return s

def _print_verbose(args, message):
    """Prints a message to stderr if verbose mode is enabled."""
    if args.verbose:
        sys.stderr.write(f"VERBOSE: {message}\n")

def _setup_arg_parser():
    """Sets up and returns the argument parser for the field_manipulate script."""
    parser = argparse.ArgumentParser(
        description="""A command-line tool for manipulating table fields.

Usage:
    field_manipulate.py [GLOBAL_OPTIONS] <operation> [OPERATION_SPECIFIC_OPTIONS]

Global Options affect how input is read and general script behavior.
Operations perform specific data manipulations and have their own options.
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Global arguments group
    global_options_group = parser.add_argument_group("Global Options")
    global_options_group.add_argument(
        "-s", "--sep", default="\t",
        help="(-s, --sep) Input/output field separator (default: tab). "
             "Supports escape sequences like '\\t', '\\n'."
    )
    global_options_group.add_argument(
        "--header", type=str, default="0", choices=["0", "None"],
        help="(--header) Specify header row. '0' for first row as header (default), "
             "'None' for no header. Column operations are 1-indexed relative to data."
    )
    global_options_group.add_argument(
        "--verbose", action="store_true",
        help="(--verbose) Enable verbose output for debugging to stderr."
    )
    global_options_group.add_argument(
        "-r", "--row-index",
        help="(-r, --row-index) Specify a column (1-indexed or name) to be treated as a row identifier. "
             "This column will be preserved and typically shown first in output for relevant operations."
    )
    global_options_group.add_argument(
        "--lowmem", action="store_true",
        help="(--lowmem) Process data in chunks to reduce memory usage. Not all operations support this fully. "
             "Operations fully supporting low-memory: grep, tr, strip, prefix_add, cleanup_values, value_counts, numeric_map, regex_capture. "
             "The 'sort' operation is incompatible with --lowmem."
    )


    subparsers = parser.add_subparsers(
        dest="operation",
        help="""Available operations. Each operation has its own specific arguments.
Use 'field_manipulate.py <operation> --help' for details on each operation."""
    )

    # --- Move Operation ---
    parser_move = subparsers.add_parser(
        "move", help="Move a column. Required: -i, -j."
    )
    parser_move.add_argument(
        "-i", "--from-col", required=True,
        help="(-i, --from-col) Source column index (1-indexed) or name."
    )
    parser_move.add_argument(
        "-j", "--to-col", required=True,
        help="(-j, --to-col) Destination column index (1-indexed) or name. If beyond existing columns, appends."
    )

    # --- col_insert Operation (Renamed from Insert) ---
    parser_col_insert = subparsers.add_parser(
        "col_insert", help="Insert a new column. Required: -i, -v."
    )
    parser_col_insert.add_argument(
        "-i", "--col-idx", required=True,
        help="(-i, --col-idx) Column index (1-indexed) or name where the new column will be inserted."
    )
    parser_col_insert.add_argument(
        "-v", "--value", required=True,
        help="(-v, --value) Value to insert into the new column. Supports escape sequences."
    )
    parser_col_insert.add_argument(
        "--new-header", default="new_column",
        help="(--new-header) Header name for the new column (default: 'new_column')."
    )

    # --- col_drop Operation (Renamed from Drop) ---
    parser_col_drop = subparsers.add_parser(
        "col_drop", help="Drop columns. Required: -i."
    )
    parser_col_drop.add_argument(
        "-i", "--cols-to-drop", required=True,
        help="(-i, --cols-to-drop) Comma-separated list of column indices (1-indexed) or names to drop. Use 'all' to drop all columns."
    )

    # --- Grep Operation (formerly Query) ---
    parser_grep = subparsers.add_parser(
        "grep", help="Filter rows. Required: -i, (-p | --starts-with | --ends-with)."
    )
    parser_grep_group = parser_grep.add_mutually_exclusive_group(required=True)
    parser_grep.add_argument(
        "-i", "--col-idx", required=True,
        help="(-i, --col-idx) Column index (1-indexed) or name to grep."
    )
    parser_grep_group.add_argument(
        "-p", "--pattern",
        help="(-p, --pattern) Regular expression pattern to search for."
    )
    parser_grep_group.add_argument(
        "--starts-with",
        help="(--starts-with) String to check if column value starts with."
    )
    parser_grep_group.add_argument(
        "--ends-with",
        help="(--ends-with) String to check if column value ends with."
    )

    # --- Split Operation ---
    parser_split = subparsers.add_parser(
        "split", help="Split a column. Required: -i, -d."
    )
    parser_split.add_argument(
        "-i", "--col-idx", required=True,
        help="(-i, --col-idx) Column index (1-indexed) or name to split."
    )
    parser_split.add_argument(
        "-d", "--delimiter", required=True,
        help="(-d, --delimiter) Delimiter to split the column by. Supports escape sequences."
    )
    parser_split.add_argument(
        "--new-header-prefix", default="split_col",
        help="(--new-header-prefix) Prefix for new headers (e.g., 'split_col_1', 'split_col_2')."
    )

    # --- Join Operation (formerly Merge) ---
    parser_join = subparsers.add_parser(
        "join", help="Join columns. Required: -i."
    )
    parser_join.add_argument(
        "-i", "--cols-to-join", required=True,
        help="(-i, --cols-to-join) Comma-separated list of column indices (1-indexed) or names to join."
    )
    parser_join.add_argument(
        "-d", "--delimiter", default="",
        help="(-d, --delimiter) Delimiter to use when joining columns (default: no delimiter). Supports escape sequences."
    )
    parser_join.add_argument(
        "--new-header", default="joined_column",
        help="(--new-header) Header name for the new joined column (default: 'joined_column')."
    )
    parser_join.add_argument(
        "-j", "--target-col-idx",
        help="(-j, --target-col-idx) 1-indexed target column index or name for the new joined column. "
             "If not specified, replaces the first column in --cols-to-join."
    )

    # --- Tr Operation (formerly Translate) ---
    parser_tr = subparsers.add_parser(
        "tr", help="Map values. Required: -i, (-d | (--from-val & --to-val))."
    )
    parser_tr.add_argument(
        "-i", "--col-idx", required=True,
        help="(-i, --col-idx) Column index (1-indexed) or name to tr."
    )
    
    # Mutually exclusive group for translation source
    tr_source_group = parser_tr.add_mutually_exclusive_group(required=True)
    tr_source_group.add_argument(
        "-d", "--dict-file",
        help="(-d, --dict-file) Path to a two-column file (key<sep>value) for tr mapping, "
             "using the main --sep as the dictionary file separator."
    )
    tr_source_group.add_argument( # This will be the trigger for single replacement mode
        "--from-val",
        help="(--from-val) Original value to tr from (for single translation). Supports escape sequences."
    )
    parser_tr.add_argument( # This argument is not part of the M-E group, but depends on --from-val
        "--to-val",
        help="(--to-val) New value to tr to (for single translation). Supports escape sequences."
    )
    parser_tr.add_argument(
        "--regex", action="store_true",
        help="(--regex) Treat --from-val as a regular expression pattern when performing single translation (default is literal)."
    )

    parser_tr.add_argument(
        "--new-header", default="_translated",
        help="(--new-header) Suffix for the new tr column's header (e.g., 'OriginalCol_translated'). "
             "If only '_translated' is given, it's appended to the original header. "
             "If a full name is given, it replaces it. (default: '_translated')"
    )
    parser_tr.add_argument(
        "--in-place", action="store_true",
        help="(--in-place) Modify the column in place instead of adding a new column."
    )

    # --- Sort Operation ---
    parser_sort = subparsers.add_parser(
        "sort", help="Sort table. Required: -i. Incompatible with --lowmem."
    )
    parser_sort.add_argument(
        "-i", "--by", required=True,
        help="(-i, --by) Comma-separated list of column indices (1-indexed) or names to sort by."
    )
    parser_sort.add_argument(
        "--desc", action="store_true",
        help="(--desc) Sort in descending order (default is ascending)."
    )
    parser_sort.add_argument(
        "-p", "--pattern",
        help="(-p, --pattern) Regex pattern to extract numeric values for sorting (e.g., '\\b([0-9]+[\\.]?[0-9]+?)[A-Z]'). "
             "Only the first captured group will be used. Requires --col-idx."
    )
    parser_sort.add_argument(
        "--suffix-map",
        help="(--suffix-map) Comma-separated key-value pairs for size suffix mapping (e.g., 'K:1000,M:1000000,G:1000000000'). "
             "Used with --pattern for numeric sorting."
    )

    # --- Cleanup Header Operation ---
    parser_cleanup_header = subparsers.add_parser(
        "cleanup_header", help="Clean up header names: remove special chars, replace spaces with underscores, lowercase."
    )

    # --- Cleanup Values Operation (formerly Cleanup Data) ---
    parser_cleanup_values = subparsers.add_parser(
        "cleanup_values", help="Clean values in columns. Required: -i."
    )
    parser_cleanup_values.add_argument(
        "-i", "--cols-to-clean", required=True,
        help="(-i, --cols-to-clean) Comma-separated list of column indices (1-indexed) or names to clean. Use 'all' to clean all columns."
    )

    # --- Prefix Add Operation ---
    parser_prefix_add = subparsers.add_parser(
        "prefix_add", help="Add prefix. Required: -i, -v."
    )
    parser_prefix_add.add_argument(
        "-i", "--cols-to-prefix", required=True,
        help="(-i, --cols-to-prefix) Comma-separated list of column indices (1-indexed) or names to add prefix to. Use 'all' for all columns."
    )
    parser_prefix_add.add_argument(
        "-v", "--string", required=True,
        help="(-v, --string) String to add as a prefix. Supports escape sequences."
    )
    parser_prefix_add.add_argument(
        "-d", "--delimiter", default="",
        help="(-d, --delimiter) Delimiter between the prefix and the original value (default: no delimiter). Supports escape sequences."
    )

    # --- Value Counts Operation (formerly Summarize) ---
    parser_value_counts = subparsers.add_parser(
        "value_counts", help="List top N frequent values. Required: -i."
    )
    parser_value_counts.add_argument(
        "-n", "--top-n", type=int, default=5,
        help="(-n, --top-n) Number of top frequent values to list (default: 5)."
    )
    parser_value_counts.add_argument(
        "-i", "--cols-to-count", required=True,
        help="(-i, --cols-to-count) Comma-separated list of column indices (1-indexed) or names to count values for. Use 'all' for all columns."
    )

    # --- Strip Operation ---
    parser_strip = subparsers.add_parser(
        "strip", help="Remove pattern from values. Required: -i, -p."
    )
    parser_strip.add_argument(
        "-i", "--col-idx", required=True,
        help="(-i, --col-idx) Column index (1-indexed) or name to strip characters from."
    )
    parser_strip.add_argument(
        "-p", "--pattern", required=True,
        help="(-p, --pattern) Regular expression pattern to remove from values."
    )
    parser_strip.add_argument(
        "--new-header", default="_stripped",
        help="(--new-header) Suffix for the new stripped column's header (e.g., 'OriginalCol_stripped'). "
             "If only '_stripped' is given, it's appended to the original header. "
             "If a full name is given, it replaces it. (default: '_stripped')"
    )
    parser_strip.add_argument(
        "--in-place", action="store_true",
        help="(--in-place) Modify the column in place instead of adding a new column."
    )

    # --- Numeric Map Operation ---
    parser_numeric_map = subparsers.add_parser(
        "numeric_map", help="Map unique strings to numbers. Required: -i."
    )
    parser_numeric_map.add_argument( # Corrected argument name
        "-i", "--col-idx", required=True,
        help="(-i, --col-idx) Column index (1-indexed) or name to map to numeric values."
    )
    parser_numeric_map.add_argument(
        "--new-header",
        help="(--new-header) Header name for the new numeric mapped column (default: 'numeric_map_of_ORIGINAL_COLUMN_NAME')."
    )

    # --- Regex Capture Operation ---
    parser_regex_capture = subparsers.add_parser(
        "regex_capture", help="Extract substrings using regex capture groups. Required: -i, -p."
    )
    parser_regex_capture.add_argument(
        "-i", "--col-idx", required=True,
        help="(-i, --col-idx) Column index (1-indexed) or name to apply the regex to."
    )
    parser_regex_capture.add_argument(
        "-p", "--pattern", required=True,
        help="(-p, --pattern) Regular expression pattern with at least one capturing group '()' (e.g., 'ID=([^;]+)')."
    )
    parser_regex_capture.add_argument(
        "--new-header", default="_captured",
        help="(--new-header) Suffix for the new captured column's header (e.g., 'OriginalCol_captured'). "
             "If only '_captured' is given, it's appended to the original header. "
             "If a full name is given, it replaces it. (default: '_captured')"
    )

    # --- View Operation ---
    parser_view = subparsers.add_parser(
        "view", help="Print formatted data."
    )
    parser_view.add_argument(
        "--max-rows", type=int, default=20,
        help="(--max-rows) Maximum number of rows to display (default: 20)."
    )
    parser_view.add_argument(
        "--max-cols", type=int, default=None,
        help="(--max-cols) Maximum number of columns to display. Default: all columns."
    )

    # --- Cut Operation ---
    parser_cut = subparsers.add_parser(
        "cut", help="Select columns. Required: -p."
    )
    parser_cut.add_argument(
        "-p", "--pattern", required=True,
        help="(-p, --pattern) String or regular expression pattern to match against column headers/indices."
    )
    parser_cut.add_argument(
        "--regex", action="store_true",
        help="(--regex) Treat the pattern as a regular expression (default is literal string match)."
    )

    # --- View Header Operation ---
    parser_viewheader = subparsers.add_parser(
        "viewheader", help="List headers."
    )

    # --- row_insert Operation (new) ---
    parser_row_insert = subparsers.add_parser(
        "row_insert", help="Insert a new row at a specified 1-indexed position. Use -i 0 to insert as a header."
    )
    parser_row_insert.add_argument(
        "-i", "--row-idx", type=int, default=0,
        help="(-i, --row-idx) 1-indexed row position for the new row. Use 0 to insert as a header (default: 0)."
    )
    parser_row_insert.add_argument(
        "-v", "--values",
        help="(-v, --values) Comma-separated values for the new row. Supports escape sequences. If -i 0 and not provided, generates generic 'col1', 'col2' etc."
    )
    # Note: row_insert will generally not be low-memory compatible for row_idx > 0

    # --- row_drop Operation (new) ---
    parser_row_drop = subparsers.add_parser(
        "row_drop", help="Delete one or more rows by their 1-indexed position. Use -i 0 to remove the first line (header/first data row)."
    )
    parser_row_drop.add_argument(
        "-i", "--row-idx", type=int, required=True,
        help="(-i, --row-idx) 1-indexed row position to drop. Use 0 to remove the first line (header/first data row)."
    )
    # Note: row_drop will generally not be low-memory compatible


    return parser

# --- Operation Handler Functions ---
def _handle_move(df, args, input_sep, is_header_present, row_idx_col_name):
    from_col_0_idx = _parse_column_arg(args.from_col, df.columns, is_header_present, "--from-col")
    to_col_0_idx_user = _parse_column_arg(args.to_col, df.columns, is_header_present, "--to-col")
    to_col_0_idx = min(to_col_0_idx_user, df.shape[1])
    _print_verbose(args, f"Moving column '{df.columns[from_col_0_idx]}' (0-indexed: {from_col_0_idx}) to position (0-indexed: {to_col_0_idx}).")
    col_to_move_name = df.columns[from_col_0_idx]
    col_to_move_data = df.pop(col_to_move_name)
    df.insert(to_col_0_idx, col_to_move_name, col_to_move_data)
    return df

def _handle_col_insert(df, args, input_sep, is_header_present, row_idx_col_name):
    col_0_idx = _parse_column_arg(args.col_idx, df.columns, is_header_present, "--col-idx")
    insert_value = codecs.decode(args.value, 'unicode_escape')
    new_col_name = args.new_header
    if is_header_present and new_col_name in df.columns:
        original_new_col_name = new_col_name; i = 1
        while f"{original_new_col_name}_{i}" in df.columns: i += 1
        new_col_name = f"{original_new_col_name}_{i}"
        _print_verbose(args, f"Header '{original_new_col_name}' already exists. Using unique header '{new_col_name}'.")
    df.insert(col_0_idx, new_col_name, insert_value)
    return df

def _handle_col_drop(df, args, input_sep, is_header_present, row_idx_col_name):
    cols_to_drop_0_idx = _parse_multiple_columns_arg(args.cols_to_drop, df.columns, is_header_present, "--cols-to-drop")
    cols_to_drop_names = [df.columns[i] for i in cols_to_drop_0_idx]
    _print_verbose(args, f"Dropping columns: {cols_to_drop_names} (0-indexed: {cols_to_drop_0_idx}).")
    df = df.drop(columns=cols_to_drop_names)
    return df

def _handle_grep(df, args, input_sep, is_header_present, row_idx_col_name):
    col_0_idx = _parse_column_arg(args.col_idx, df.columns, is_header_present, "--col-idx")
    target_col_name = df.columns[col_0_idx]
    target_col = df.iloc[:, col_0_idx].astype(str)
    if args.pattern:
        try: df = df[target_col.str.contains(args.pattern, regex=True, na=False)]
        except re.error as e: raise ValueError(f"Error: Invalid regex pattern '{args.pattern}': {e}")
    elif args.starts_with: df = df[target_col.str.startswith(args.starts_with, na=False)]
    elif args.ends_with: df = df[target_col.str.endswith(args.ends_with, na=False)]
    return df

def _handle_split(df, args, input_sep, is_header_present, row_idx_col_name):
    col_0_idx = _parse_column_arg(args.col_idx, df.columns, is_header_present, "--col-idx")
    delimiter = codecs.decode(args.delimiter, 'unicode_escape')
    original_col_name = df.columns[col_0_idx]
    _print_verbose(args, f"Splitting column '{original_col_name}' by delimiter {delimiter!r}.")
    split_cols_df = df.iloc[:, col_0_idx].astype(str).str.split(delimiter, expand=True).fillna('')
    new_split_headers = []
    for i in range(split_cols_df.shape[1]):
        base_header = f"{args.new_header_prefix}_{i + 1}"
        if is_header_present:
            potential_header = f"{original_col_name}_{base_header}"
            if potential_header not in df.columns: new_split_header = potential_header
            else:
                j = 1
                while f"{original_col_name}_{base_header}_{j}" in df.columns: j += 1
                new_split_header = f"{original_col_name}_{base_header}_{j}"
        else: new_split_header = base_header
        new_split_headers.append(new_split_header)
    split_cols_df.columns = new_split_headers
    _print_verbose(args, f"New split column headers: {new_split_headers}.")
    df = df.drop(columns=[original_col_name])
    df = pd.concat([df.iloc[:, :col_0_idx], split_cols_df, df.iloc[:, col_0_idx:]], axis=1)
    return df

def _handle_join(df, args, input_sep, is_header_present, row_idx_col_name):
    cols_to_join_0_idx = _parse_multiple_columns_arg(args.cols_to_join, df.columns, is_header_present, "--cols-to-join")
    if not cols_to_join_0_idx: raise ValueError("Error: No columns specified for join operation.")
    delimiter = codecs.decode(args.delimiter, 'unicode_escape')
    cols_to_join_names = [df.columns[i] for i in cols_to_join_0_idx]
    _print_verbose(args, f"Joining columns: {cols_to_join_names} with delimiter {delimiter!r}.")
    joined_col_data = df.iloc[:, cols_to_join_0_idx[0]].astype(str)
    for i in range(1, len(cols_to_join_0_idx)):
        joined_col_data = joined_col_data + delimiter + df.iloc[:, cols_to_join_0_idx[i]].astype(str)
    if args.target_col_idx is not None:
        insert_loc_0_idx = _parse_column_arg(str(args.target_col_idx), df.columns, is_header_present, "--target-col-idx")
    else: insert_loc_0_idx = cols_to_join_0_idx[0]
    new_col_name = args.new_header
    if is_header_present and new_col_name in df.columns:
        original_new_col_name = new_col_name; i = 1
        while f"{original_new_col_name}_{i}" in df.columns: i += 1
        new_col_name = f"{original_new_col_name}_{i}"
        _print_verbose(args, f"Header '{original_new_col_name}' already exists. Using unique header '{new_col_name}'.")
    cols_to_drop_names = [df.columns[idx] for idx in sorted(cols_to_join_0_idx, reverse=True)]
    df = df.drop(columns=cols_to_drop_names)
    df.insert(min(insert_loc_0_idx, df.shape[1]), new_col_name, joined_col_data.reset_index(drop=True))
    _print_verbose(args, f"Joined column '{new_col_name}' inserted at position (0-indexed: {min(insert_loc_0_idx, df.shape[1])}).")
    return df

def _handle_tr(df, args, input_sep, is_header_present, row_idx_col_name):
    col_0_idx = _parse_column_arg(args.col_idx, df.columns, is_header_present, "--col-idx")
    original_col_name = df.columns[col_0_idx]
    translated_col_data = None
    if args.dict_file:
        translation_map = {}
        try:
            _print_verbose(args, f"Loading translation map from '{args.dict_file}' using separator {input_sep!r}.")
            with open(args.dict_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(input_sep, 1)
                    if len(parts) == 2: translation_map[parts[0]] = parts[1]
                    else: sys.stderr.write(f"Warning: Skipping malformed line in dictionary file: '{line}'\n")
            _print_verbose(args, f"Loaded {len(translation_map)} entries for translation.")
        except FileNotFoundError: raise FileNotFoundError(f"Error: Dictionary file not found: '{args.dict_file}'")
        except Exception as e: raise RuntimeError(f"Error reading dictionary file: {e}")
        translated_col_data = df.iloc[:, col_0_idx].astype(str).apply(lambda x: translation_map.get(x, x))
    elif args.from_val:
        if not args.to_val: raise ValueError("Error: --to-val must be specified when using --from-val.")
        from_val_decoded = codecs.decode(args.from_val, 'unicode_escape')
        to_val_decoded = codecs.decode(args.to_val, 'unicode_escape')
        _print_verbose(args, f"Translating values in column '{original_col_name}' from '{from_val_decoded}' to '{to_val_decoded}' {'(regex)' if args.regex else '(literal)'}.")
        try: translated_col_data = df.iloc[:, col_0_idx].astype(str).str.replace(from_val_decoded, to_val_decoded, regex=args.regex, n=-1)
        except re.error as e: raise ValueError(f"Error: Invalid regex pattern '{from_val_decoded}': {e}")
    else: raise ValueError("Error: For tr operation, either -d/--dict-file or --from-val (with --to-val) must be specified.")
    
    if args.in_place:
        _print_verbose(args, f"Translating column '{original_col_name}' in place.")
        df.iloc[:, col_0_idx] = translated_col_data
    else:
        if args.new_header.startswith("_"): new_col_name = str(original_col_name) + args.new_header
        else: new_col_name = args.new_header
        if is_header_present and new_col_name in df.columns:
            original_new_col_name = new_col_name; i = 1
            while f"{original_new_col_name}_{i}" in df.columns: i += 1
            new_col_name = f"{original_new_col_name}_{i}"
            _print_verbose(args, f"Header '{original_new_col_name}' already exists. Using unique header '{new_col_name}'.")
        df.insert(col_0_idx + 1, new_col_name, translated_col_data.reset_index(drop=True))
        _print_verbose(args, f"Translated column '{new_col_name}' inserted after '{original_col_name}'.")
    return df

def _parse_size_value(value, suffix_map):
    match = re.match(r"(\d+(\.\d+)?)([KMGTP])?", value, re.IGNORECASE)
    if not match:
        try:
            return float(value)
        except ValueError:
            return None # Cannot parse
    num = float(match.group(1))
    suffix = match.group(3)
    if suffix:
        suffix_key = suffix.upper()
        if suffix_key in suffix_map:
            return num * suffix_map[suffix_key]
        else:
            return None # Unknown suffix
    return num

def _handle_sort(df, args, input_sep, is_header_present, row_idx_col_name):
    cols_0_idx = _parse_multiple_columns_arg(args.by, df.columns, is_header_present, "--by")
    sort_by_columns = [df.columns[idx] for idx in cols_0_idx]
    
    if args.pattern:
        if len(cols_0_idx) > 1:
            raise ValueError("Error: --pattern for numeric sorting can only be used with a single column specified by --by.")
        
        target_col_name = sort_by_columns[0]
        
        suffix_map = {}
        if args.suffix_map:
            for pair in args.suffix_map.split(','):
                if ':' in pair:
                    key, val = pair.split(':', 1)
                    try:
                        suffix_map[key.strip().upper()] = float(val.strip())
                    except ValueError:
                        raise ValueError(f"Error: Invalid suffix-map format. '{pair}' is not a valid key:value pair with numeric value.")
                else:
                    raise ValueError(f"Error: Invalid suffix-map format. '{pair}' should be in 'KEY:VALUE' format.")
        
        _print_verbose(args, f"Sorting column '{target_col_name}' numerically using pattern '{args.pattern}' and suffix map {suffix_map}.")

        # Create a temporary column for sorting numeric values
        temp_sort_col_name = f"_temp_numeric_sort_{target_col_name}"
        
        def extract_and_convert(val):
            if not isinstance(val, str):
                return None
            match = re.search(args.pattern, val)
            if match and match.group(1):
                numeric_part = match.group(1)
                full_match = match.group(0) # Get the full string that matched the pattern
                suffix = None
                # Attempt to find a suffix after the numeric part if the regex captures it
                # This assumes the pattern covers the numeric part and optionally a suffix character at the end.
                # If the pattern extracts only the number, we need to re-evaluate the full_match for suffix.
                
                # A more robust way: if pattern has a group for numeric value, and we need suffix from *outside* that group
                # For `du -h` example: "101M", pattern `([0-9.]+)([KMGTP])?`
                # If the pattern is `\b([0-9]+[\.]?[0-9]+?)[A-Z]`, the `[A-Z]` is outside group 1.
                # So we need to match the original string again for the suffix, or ensure pattern captures it in a separate group.
                
                # Given the example `\b([0-9]+[\.]?[0-9]+?)[A-Z]`, the suffix is *not* in group 1.
                # We need to extract the number and the suffix separately if the regex is designed this way.
                
                # Let's refine the extraction based on the provided example pattern.
                # The pattern "\b([0-9]+[\.]?[0-9]+?)[A-Z]" captures the number in group 1.
                # If there's an A-Z character *after* the captured group, it's the suffix.
                
                # We can modify `_parse_size_value` to take the full matched string for suffix parsing.
                # Or, simplify the logic: if the pattern captures the number, apply suffix mapping.
                
                # Simplified approach: If a suffix map is given, assume the pattern extracts the numeric part,
                # and we need to look for a suffix *immediately following* that numeric part in the original string.
                # This requires parsing the *original* string, not just the regex capture.
                
                if suffix_map:
                    # Try to extract the number and the suffix from the *original* string based on the pattern match
                    # This requires the regex to effectively find the number and then we look for the suffix.
                    # This custom parsing function handles both the number and the suffix.
                    return _parse_size_value(full_match, suffix_map)
                else:
                    try:
                        return float(numeric_part)
                    except ValueError:
                        return None # Cannot convert to float
            return None # No match or empty capture

        df[temp_sort_col_name] = df[target_col_name].astype(str).apply(extract_and_convert)
        # Sort by the temporary numeric column, then by the original column for tie-breaking/stability
        df = df.sort_values(by=[temp_sort_col_name, target_col_name], ascending=[not args.desc, True], kind='stable')
        df = df.drop(columns=[temp_sort_col_name]) # Drop the temporary column
    else:
        _print_verbose(args, f"Sorting by columns: {sort_by_columns} in {'descending' if args.desc else 'ascending'} order.")
        df = df.sort_values(by=sort_by_columns, ascending=not args.desc, kind='stable')
    return df

def _handle_cleanup_header(df, args, input_sep, is_header_present, row_idx_col_name):
    if not is_header_present:
        sys.stderr.write("Warning: No header specified (--header=None). 'cleanup_header' has no effect.\n")
        _print_verbose(args, "Skipping cleanup_header as no header is present.")
    else:
        original_headers = list(df.columns)
        df.columns = [_clean_string_for_header_and_data(col) for col in df.columns]
        _print_verbose(args, f"Header cleaned. Original: {original_headers}, New: {list(df.columns)}")
    return df

def _handle_cleanup_values(df, args, input_sep, is_header_present, row_idx_col_name):
    cols_to_clean_0_idx = _parse_multiple_columns_arg(args.cols_to_clean, df.columns, is_header_present, "--cols-to-clean")
    cols_to_clean_names = [df.columns[i] for i in cols_to_clean_0_idx]
    _print_verbose(args, f"Cleaning data in columns: {cols_to_clean_names}.")
    for col_0_idx in cols_to_clean_0_idx:
        df.iloc[:, col_0_idx] = df.iloc[:, col_0_idx].apply(_clean_string_for_header_and_data)
    return df

def _handle_prefix_add(df, args, input_sep, is_header_present, row_idx_col_name):
    cols_to_prefix_0_idx = _parse_multiple_columns_arg(args.cols_to_prefix, df.columns, is_header_present, "--cols-to-prefix")
    prefix_string = codecs.decode(args.string, 'unicode_escape')
    prefix_delimiter = codecs.decode(args.delimiter, 'unicode_escape')
    cols_to_prefix_names = [df.columns[i] for i in cols_to_prefix_0_idx]
    _print_verbose(args, f"Adding prefix '{prefix_string}' with delimiter {prefix_delimiter!r} to columns: {cols_to_prefix_names}.")
    for col_0_idx in cols_to_prefix_0_idx:
        df.iloc[:, col_0_idx] = df.iloc[:, col_0_idx].astype(str).apply(lambda x: f"{prefix_string}{prefix_delimiter}{x}")
    return df

def _handle_value_counts(df, args, input_sep, is_header_present, row_idx_col_name, state=None):
    if state is None: # Full memory mode or first chunk
        value_counts_counts = Counter()
    else: # Subsequent chunks
        value_counts_counts = state

    cols_to_count_0_idx = _parse_multiple_columns_arg(args.cols_to_count, df.columns, is_header_present, "--cols-to-count")
    if not cols_to_count_0_idx: raise ValueError("Error: No columns specified for value_counts operation.")
    
    for col_0_idx in cols_to_count_0_idx:
        chunk_col_name = df.columns[col_0_idx] # df.columns is correct for the current chunk
        if chunk_col_name in df.columns: # Redundant check but safe if columns were manipulated
            col_data = df[chunk_col_name].astype(str)
            value_counts_counts.update(col_data)
        else:
            sys.stderr.write(f"Warning: Column '{col_0_idx+1}' ('{chunk_col_name}') not found in chunk. Skipping.\n")
    
    # If this is the last chunk or full memory mode, print summary and exit
    if df.index.stop == CHUNK_SIZE or state is None: # This heuristic for last chunk is imperfect, better to use an 'is_last_chunk' flag
        sorted_summary = sorted(value_counts_counts.items(), key=lambda item: item[1], reverse=True)
        summary_df = pd.DataFrame(sorted_summary[:args.top_n], columns=['Value', 'Count'])
        sys.stdout.write(f"--- Summary (Top {args.top_n}) ---\n")
        sys.stdout.write(summary_df.to_string(index=False, header=True) + '\n')
        sys.exit(0) # Value counts prints directly and exits

    return df, value_counts_counts # Return df and updated state for next chunk

def _handle_strip(df, args, input_sep, is_header_present, row_idx_col_name):
    col_0_idx = _parse_column_arg(args.col_idx, df.columns, is_header_present, "--col-idx")
    original_col_name = df.columns[col_0_idx]
    try: stripped_col_data = df.iloc[:, col_0_idx].astype(str).str.replace(args.pattern, '', regex=True)
    except re.error as e: raise ValueError(f"Error: Invalid regex pattern '{args.pattern}': {e}")

    if args.in_place:
        _print_verbose(args, f"Stripping pattern '{args.pattern}' from column '{original_col_name}' in place.")
        df.iloc[:, col_0_idx] = stripped_col_data
    else:
        _print_verbose(args, f"Stripping pattern '{args.pattern}' from column '{original_col_name}', adding a new column.")
        if args.new_header.startswith("_"): new_col_name = str(original_col_name) + args.new_header
        else: new_col_name = args.new_header
        if is_header_present and new_col_name in df.columns:
            original_new_col_name = new_col_name; i = 1
            while f"{original_new_col_name}_{i}" in df.columns: i += 1
            new_col_name = f"{original_new_col_name}_{i}"
            _print_verbose(args, f"Header '{original_new_col_name}' already exists. Using unique header '{new_col_name}'.")
        df.insert(col_0_idx + 1, new_col_name, stripped_col_data.reset_index(drop=True))
        _print_verbose(args, f"Stripped column '{new_col_name}' inserted after '{original_col_name}'.")
    return df

def _handle_numeric_map(df, args, input_sep, is_header_present, row_idx_col_name, state=None):
    # State for chunked mode: (numeric_map_unique_values, numeric_map_next_id)
    if state is None:
        numeric_map_unique_values = {}
        numeric_map_next_id = 1
    else:
        numeric_map_unique_values, numeric_map_next_id = state

    col_0_idx = _parse_column_arg(args.col_idx, df.columns, is_header_present, "--col-idx")
    original_col_name = df.columns[col_0_idx]
    _print_verbose(args, f"Mapping unique values in column '{original_col_name}' to numeric values.")
    
    if args.new_header: new_col_name = args.new_header
    else: new_col_name = f"numeric_map_of_{str(original_col_name)}"
    if is_header_present and new_col_name in df.columns and state is None: # Only check for uniqueness once in full memory or first chunk
        original_new_col_name = new_col_name; i = 1
        while f"{original_new_col_name}_{i}" in df.columns: i += 1
        new_col_name = f"{original_new_col_name}_{i}"
        _print_verbose(args, f"Header '{original_new_col_name}' already exists. Using unique header '{new_col_name}'.")

    mapped_series = df.iloc[:, col_0_idx].astype(str).apply(
        lambda x: numeric_map_unique_values.setdefault(x, numeric_map_next_id) or numeric_map_unique_values[x]
    )
    current_max_id = max(numeric_map_unique_values.values()) if numeric_map_unique_values else 0
    if current_max_id >= numeric_map_next_id:
        numeric_map_next_id = current_max_id + 1

    df.insert(df.columns.get_loc(original_col_name) + 1, new_col_name, mapped_series)
    _print_verbose(args, f"Numeric mapped column '{new_col_name}' inserted after '{original_col_name}'.")
    
    # For full memory, print mapping at the end (or pass to main to print after all chunks)
    # The `main` loop should ideally collect this and print if it's the last chunk.
    # For now, it's handled by returning the state for chunked, and handled outside for full.
    
    return df, (numeric_map_unique_values, numeric_map_next_id) # Return df and updated state for chunked mode

def _handle_regex_capture(df, args, input_sep, is_header_present, row_idx_col_name):
    col_0_idx = _parse_column_arg(args.col_idx, df.columns, is_header_present, "--col-idx")
    
    if not is_header_present:
        base_col_name = f"captured_column_{col_0_idx + 1}"
    else:
        base_col_name = df.columns[col_0_idx]
    
    if args.new_header.startswith("_"):
        new_col_name = str(base_col_name) + args.new_header
    else:
        new_col_name = args.new_header
    
    if is_header_present and new_col_name in df.columns:
        original_new_col_name = new_col_name
        i = 1
        while f"{original_new_col_name}_{i}" in df.columns:
            i += 1
        new_col_name = f"{original_new_col_name}_{i}"
        _print_verbose(args, f"Header '{original_new_col_name}' already exists. Using unique header '{new_col_name}'.")

    _print_verbose(args, f"Applying regex '{args.pattern}' to column '{base_col_name}', creating new column '{new_col_name}'.")
    try:
        captured_data = df.iloc[:, col_0_idx].astype(str).apply(
            lambda x: ';'.join(re.findall(args.pattern, x))
        )
    except re.error as e:
        raise ValueError(f"Error: Invalid regex pattern '{args.pattern}': {e}")

    df.insert(len(df.columns), new_col_name, captured_data.reset_index(drop=True))
    return df

def _handle_view(df, args, input_sep, is_header_present, row_idx_col_name, raw_first_line_content=None):
    _print_verbose(args, f"Viewing data. Max rows: {args.max_rows}, Max cols: {args.max_cols}.")
    pd.set_option('display.max_rows', args.max_rows)
    pd.set_option('display.max_columns', args.max_cols)
    pd.set_option('display.width', None)
    pd.set_option('display.colheader_justify','left')
    display_df = df.copy()
    if row_idx_col_name and row_idx_col_name in display_df.columns:
        cols = [row_idx_col_name] + [col for col in display_df.columns if col != row_idx_col_name]
        display_df = display_df[cols]
        _print_verbose(args, f"Moved row index column '{row_idx_col_name}' to the front for display.")
    sys.stdout.write(display_df.to_string(index=True, header=is_header_present) + '\n')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.colheader_justify')
    sys.exit(0) # View prints directly and exits

def _handle_cut(df, args, input_sep, is_header_present, row_idx_col_name):
    pattern = args.pattern
    _print_verbose(args, f"Cutting columns with pattern '{pattern}' (regex: {args.regex}).")
    selected_columns = []
    for col_name in df.columns:
        if args.regex:
            try:
                if re.search(pattern, str(col_name)): selected_columns.append(col_name)
            except re.error as e: raise ValueError(f"Error: Invalid regex pattern '{pattern}': {e}")
        else:
            if pattern in str(col_name): selected_columns.append(col_name)
    if row_idx_col_name and row_idx_col_name in df.columns and row_idx_col_name not in selected_columns:
        selected_columns = [row_idx_col_name] + selected_columns
        _print_verbose(args, f"Included specified row index column '{row_idx_col_name}' in cut output.")
    if not selected_columns:
        sys.stderr.write(f"Warning: No columns matched the pattern '{pattern}'. Outputting empty data.\n")
        df = pd.DataFrame(columns=[])
    else:
        df = df[selected_columns]
        _print_verbose(args, f"Selected columns: {selected_columns}.")
    return df

def _handle_viewheader(df, args, input_sep, is_header_present, row_idx_col_name, raw_first_line_content):
    _print_verbose(args, "Listing headers and their 1-indexed positions.")
    header_output = []
    # If df is empty but has columns (e.g., from an input with only a header)
    if df.empty and not df.columns.empty:
        for i, col_name in enumerate(df.columns):
            header_output.append(f"{i+1}\t{col_name}")
    elif not df.empty:
        current_columns_list = list(df.columns)
        if not is_header_present and raw_first_line_content: # If no header, and we have raw first line
            first_row_values = raw_first_line_content 
            
            for i, col_value in enumerate(first_row_values):
                col_display_name = col_value if pd.notna(col_value) else f"Column_{i+1}"
                indicator = " (Row Index)" if row_idx_col_name and current_columns_list[i] == row_idx_col_name else ""
                header_output.append(f"{i+1}\t{col_display_name}{indicator}")
        else: # Either header is present, or df is not empty and raw_first_line_content is not available (e.g., chunked)
            for i, col_name in enumerate(current_columns_list):
                indicator = " (Row Index)" if row_idx_col_name and col_name == row_idx_col_name else ""
                header_output.append(f"{i+1}\t{col_name}{indicator}")
    
    if not header_output and is_header_present: # Only warn if header was expected but none found
        sys.stderr.write("No columns found to display. The input might be empty or malformed.\n")
    else:
        sys.stdout.write("\n".join(header_output) + '\n')
    sys.exit(0)

def _handle_row_insert(df, args, input_sep, is_header_present, row_idx_col_name):
    # This block is for row_insert when it's NOT the low-memory optimized -i 0 case.
    # So, it's either not lowmem, or row_idx > 0.
    # args.row_idx is 1-indexed, convert to 0-indexed for pandas
    insert_idx_0_based = args.row_idx - 1 

    # Determine values for the new row
    new_row_values = []
    if args.values:
        new_row_values = [codecs.decode(v.strip(), 'unicode_escape') for v in args.values.split(',')]
    elif df.empty: # Empty input, but need to infer columns from values for first row.
        sys.stderr.write("Error: Cannot insert row into empty table without explicit values to determine column count.\n")
        sys.exit(1)
    elif not args.values: # Inserting into non-empty data without values
        num_cols = df.shape[1]
        new_row_values = [f"col{i+1}" for i in range(num_cols)]
        _print_verbose(args, f"Generated generic column names for new row: {new_row_values}")
    
    # Ensure new_row_values matches DataFrame columns (or can infer if df is empty)
    num_target_cols = len(df.columns)
    
    if len(new_row_values) > num_target_cols:
        new_row_values = new_row_values[:num_target_cols]
        _print_verbose(args, f"Truncated new row values to match column count ({num_target_cols}).")
    elif len(new_row_values) < num_target_cols:
        new_row_values.extend([''] * (num_target_cols - len(new_row_values)))
        _print_verbose(args, f"Padded new row values with empty strings to match column count ({num_target_cols}).")


    new_row_df = pd.DataFrame([new_row_values], columns=df.columns)
    
    if insert_idx_0_based >= len(df): # Inserting at or beyond the end
        df = pd.concat([df, new_row_df], ignore_index=True)
    else: # Inserting in the middle
        df_before = df.iloc[:insert_idx_0_based]
        df_after = df.iloc[insert_idx_0_based:]
        df = pd.concat([df_before, new_row_df, df_after], ignore_index=True)

    _print_verbose(args, f"Inserted row at 1-indexed position {args.row_idx}. New DataFrame shape: {df.shape}")
    return df

def _handle_row_drop(df, args, input_sep, is_header_present, row_idx_col_name):
    # This operation is explicitly not low-memory compatible (checked earlier).
    # args.row_idx is 1-indexed, convert to 0-indexed for pandas
    drop_idx_0_based = args.row_idx - 1

    if drop_idx_0_based < 0 or drop_idx_0_based >= len(df):
        raise IndexError(f"Error: Row index {args.row_idx} is out of bounds. Table has {len(df)} rows.\n")
    
    df = df.drop(df.index[drop_idx_0_based]).reset_index(drop=True)
    _print_verbose(args, f"Dropped row at 1-indexed position {args.row_idx}. New DataFrame shape: {df.shape}")
    return df


# Dispatch table for operations
OPERATION_HANDLERS = {
    "move": _handle_move,
    "col_insert": _handle_col_insert,
    "col_drop": _handle_col_drop,
    "grep": _handle_grep,
    "split": _handle_split,
    "join": _handle_join,
    "tr": _handle_tr,
    "sort": _handle_sort,
    "cleanup_header": _handle_cleanup_header,
    "cleanup_values": _handle_cleanup_values,
    "prefix_add": _handle_prefix_add,
    "value_counts": _handle_value_counts,
    "strip": _handle_strip,
    "numeric_map": _handle_numeric_map,
    "regex_capture": _handle_regex_capture,
    "view": _handle_view,
    "cut": _handle_cut,
    "viewheader": _handle_viewheader,
    "row_insert": _handle_row_insert,
    "row_drop": _handle_drop,
}

# --- Input/Output Functions ---
def _read_input_data(args, input_sep, header_param, is_header_present, use_chunked_pandas_processing):
    """
    Reads input data into a Pandas DataFrame or a TextFileReader.
    Returns (DataFrame/iterator, raw_first_line_content_list_if_no_header).
    """
    df = None
    raw_first_line_content = [] # Used only if header is explicitly None
    
    if use_chunked_pandas_processing:
        try:
            text_file_reader = pd.read_csv(
                sys.stdin, sep=input_sep, header=header_param, dtype=str,
                chunksize=CHUNK_SIZE, iterator=True
            )
            first_chunk_df = next(text_file_reader)
            
            if first_chunk_df.empty and args.operation not in ["viewheader", "view", "value_counts", "regex_capture"]:
                sys.stderr.write(f"Error: Input data is empty. '{args.operation}' requires input data. Exiting.\n")
                sys.exit(1)
            
            # Re-adjust columns for subsequent chunks if header was None
            if not is_header_present:
                first_chunk_df.columns = pd.Index(range(first_chunk_df.shape[1]))

            # Create a generator that yields the first chunk and then the rest
            def chunk_generator():
                yield first_chunk_df
                yield from text_file_reader
            
            return chunk_generator(), raw_first_line_content # Return iterator
        except StopIteration:
            sys.stderr.write(f"Warning: Input data is empty. Cannot perform '{args.operation}'. Exiting.\n")
            sys.exit(0)
        except pd.errors.EmptyDataError:
            sys.stderr.write(f"Warning: Input data is empty. Cannot perform '{args.operation}'. Exiting.\n")
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"Error reading initial chunk for low-memory processing: {e}\n")
            sys.exit(1)
    else: # Full memory load
        try:
            # Read all content to check if empty or get first line if no header
            csv_content = sys.stdin.read() 
            
            if not csv_content.strip() and args.operation not in ["viewheader", "view", "value_counts"]:
                sys.stderr.write(f"Error: Input data is empty. '{args.operation}' requires input data. Exiting.\n")
                sys.exit(1)
            
            if not csv_content.strip(): # Still empty after checking against allowed ops
                df = pd.DataFrame(columns=[] if is_header_present else []) # Empty DF, with or without header
                return df, raw_first_line_content # Return empty DF

            # Read the entire content into a StringIO for pandas
            csv_data = StringIO(csv_content)
            
            # If header is explicitly None, capture the first line before pandas processes it
            if not is_header_present:
                current_pos = csv_data.tell()
                first_line = csv_data.readline().strip()
                raw_first_line_content = first_line.split(input_sep) if first_line else []
                csv_data.seek(current_pos) # Reset for pandas
            
            df = pd.read_csv(csv_data, sep=input_sep, header=header_param, dtype=str)
            return df, raw_first_line_content
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=[] if is_header_present else [])
            _print_verbose(args, "Input data is empty. Proceeding with an empty DataFrame.")
            return df, raw_first_line_content
        except Exception as e:
            sys.stderr.write(f"Error reading input data: {e}\n")
            sys.exit(1)

def _write_output_data(df_or_chunk, args, input_sep, is_header_present, output_header_printed):
    """
    Writes DataFrame or chunk to stdout. Manages header printing for chunked output.
    Returns updated output_header_printed status.
    """
    try:
        if args.operation in ["view", "viewheader", "value_counts"]:
            # These operations handle their own output and sys.exit()
            return output_header_printed # No change needed
        
        if isinstance(df_or_chunk, pd.DataFrame): # Full DataFrame mode
            df_or_chunk.to_csv(sys.stdout, sep=input_sep, index=False, header=is_header_present, encoding='utf-8')
            return True # Header printed (or not, but it's done)
        
        # Chunked mode
        if not output_header_printed and is_header_present:
            df_or_chunk.to_csv(sys.stdout, sep=input_sep, index=False, header=True, encoding='utf-8')
            return True # Header printed for first chunk
        else:
            df_or_chunk.to_csv(sys.stdout, sep=input_sep, index=False, header=False, encoding='utf-8')
            return output_header_printed # Status remains the same
    except BrokenPipeError:
        # Expected when piping to utilities like 'head'
        pass
    except Exception as e:
        sys.stderr.write(f"Error writing output: {e}\n")
        sys.exit(1)
    return output_header_printed # In case of other errors, maintain status


def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()

    # If no operation is specified (e.g., just `field_manipulate`), print help
    if not hasattr(args, 'operation') or args.operation is None:
        parser.print_help()
        sys.exit(0)

    # Prepare the command string for printing in _execute_operation
    command_str_for_log = [sys.argv[0]]
    if args.operation: command_str_for_log.append(args.operation)
    if hasattr(args, 'col_idx') and args.col_idx is not None: command_str_for_log.extend(['-i', str(args.col_idx)])
    if hasattr(args, 'pattern') and args.pattern is not None: command_str_for_log.extend(['-p', f"'{args.pattern}'"])
    if hasattr(args, 'values') and args.values is not None: command_str_for_log.extend(['-v', f"'{args.values}'"])
    if hasattr(args, 'from_col') and args.from_col is not None: command_str_for_log.extend(['--from-col', str(args.from_col)])
    if hasattr(args, 'to_col') and args.to_col is not None: command_str_for_log.extend(['--to-col', str(args.to_col)])
    if hasattr(args, 'cols_to_drop') and args.cols_to_drop is not None: command_str_for_log.extend(['-i', f"'{args.cols_to_drop}'"]) # For col_drop
    if hasattr(args, 'cols_to_join') and args.cols_to_join is not None: command_str_for_log.extend(['-i', f"'{args.cols_to_join}'"]) # For join
    if hasattr(args, 'by') and args.by is not None: command_str_for_log.extend(['-i', f"'{args.by}'"])


    input_sep = codecs.decode(args.sep, 'unicode_escape')
    header_param = 0 if args.header == "0" else None
    is_header_present = (args.header == "0")

    # Check for incompatible operations with lowmem
    if args.lowmem and args.operation == "sort":
        sys.stderr.write("Error: 'sort' operation cannot be performed in low-memory mode (--lowmem) as it requires loading all data.\n")
        sys.exit(1)
    
    # Identify row_insert -i 0 and row_drop -i 0 as special low-memory cases
    is_lowmem_optimized_row_op = False
    if args.operation == "row_insert" and args.row_idx == 0:
        is_lowmem_optimized_row_op = True
    elif args.operation == "row_drop" and args.row_idx == 0:
        is_lowmem_optimized_row_op = True

    if args.lowmem and args.operation in ["row_insert", "row_drop"] and not is_lowmem_optimized_row_op:
        sys.stderr.write(f"Error: '{args.operation}' operation is not compatible with low-memory mode (--lowmem) except for 'row_insert -i 0' and 'row_drop -i 0'. It requires loading all data.\n")
        sys.exit(1)


    lowmem_supported_ops_pandas_chunk = ["value_counts", "numeric_map", "grep", "tr", "strip", "prefix_add", "cleanup_values", "regex_capture"]
    use_chunked_pandas_processing = args.lowmem and args.operation in lowmem_supported_ops_pandas_chunk

    # --- Execute low-memory optimized row operations directly ---
    if is_lowmem_optimized_row_op:
        sys.stderr.write(f"# Processing: {' '.join(command_str_for_log)}\n")
        if args.operation == "row_insert":
            _print_verbose(args, "Performing low-memory optimized 'row_insert -i 0'.")
            first_line_of_input = sys.stdin.readline().strip()
            num_cols = 0
            if first_line_of_input:
                num_cols = len(first_line_of_input.split(input_sep))
            elif args.values:
                num_cols = len(args.values.split(','))
            else:
                num_cols = 1 
            new_header_values = [codecs.decode(v.strip(), 'unicode_escape') for v in args.values.split(',')] if args.values else [f"col{i+1}" for i in range(num_cols)]
            if len(new_header_values) > num_cols: new_header_values = new_header_values[:num_cols]
            elif len(new_header_values) < num_cols: new_header_values.extend([''] * (num_cols - len(new_header_values)))
            try:
                sys.stdout.write(input_sep.join(new_header_values) + '\n')
                if first_line_of_input: sys.stdout.write(first_line_of_input + '\n')
                for line in sys.stdin: sys.stdout.write(line)
            except BrokenPipeError: pass
            sys.exit(0)
        elif args.operation == "row_drop":
            _print_verbose(args, "Performing low-memory optimized 'row_drop -i 0'.")
            try:
                first_line = sys.stdin.readline()
                if not first_line.strip(): sys.stderr.write("Warning: Attempted to drop first line, but input stream might be empty or first line was blank.\n")
                for line in sys.stdin: sys.stdout.write(line)
            except BrokenPipeError: pass
            sys.exit(0)
    
    # --- For all other operations (either full-memory or chunked Pandas processing) ---
    sys.stderr.write(f"# Processing: {' '.join(command_str_for_log)}\n")
    df_or_chunks, raw_first_line_content = _read_input_data(args, input_sep, header_param, is_header_present, use_chunked_pandas_processing)
    
    row_idx_col_name = None
    if args.row_index:
        # Determine row_idx_col_name from the first chunk or the full DataFrame
        if use_chunked_pandas_processing:
            # Need to peek at the first chunk's columns without consuming it
            # This is complex with a generator, so we'll rely on the operation handlers
            # to do the initial column parsing or pass the first chunk's columns
            # For simplicity, will try to get it from the first chunk if it exists.
            try:
                # To get initial columns for row_index from a generator:
                # Read the first chunk, get its columns, then chain it back.
                peek_chunk = next(df_or_chunks)
                initial_df_columns = peek_chunk.columns
                # Re-adjust columns if header was None (as _read_input_data does this for chunks)
                if not is_header_present:
                    initial_df_columns = pd.Index(range(peek_chunk.shape[1]))
                row_idx_0_based = _parse_column_arg(args.row_index, initial_df_columns, is_header_present, "--row-index")
                row_idx_col_name = initial_df_columns[row_idx_0_based]
                # Now, create a new generator including the peeked chunk
                df_or_chunks = (item for item in [peek_chunk] + list(df_or_chunks)) # Re-wrap for chunked
                _print_verbose(args, f"Row index column specified: '{row_idx_col_name}' (0-indexed: {row_idx_0_based}) from first chunk.")
            except StopIteration:
                sys.stderr.write("Error: Input is empty, cannot determine --row-index column.\n")
                sys.exit(1)
        else: # Full DataFrame mode
            if isinstance(df_or_chunks, pd.DataFrame):
                row_idx_0_based = _parse_column_arg(args.row_index, df_or_chunks.columns, is_header_present, "--row-index")
                row_idx_col_name = df_or_chunks.columns[row_idx_0_based]
                _print_verbose(args, f"Row index column specified: '{row_idx_col_name}' (0-indexed: {row_idx_0_based}).")
            else:
                sys.stderr.write(f"Error: Cannot determine --row-index column for '{args.operation}' in current mode.\n")
                sys.exit(1)


    output_header_printed = False
    operation_state = {} # To hold state for chunked operations like numeric_map

    try:
        if use_chunked_pandas_processing:
            for chunk_df in df_or_chunks: # df_or_chunks is an iterator here
                # Ensure chunk columns align with initial_df_columns for consistent parsing
                if is_header_present and not chunk_df.empty and len(chunk_df.columns) == len(initial_df_columns):
                    chunk_df.columns = initial_df_columns
                elif not is_header_present:
                    chunk_df.columns = pd.Index(range(chunk_df.shape[1]))

                processed_chunk = chunk_df
                # Handle operations that might return state (e.g., numeric_map, value_counts)
                if args.operation in ["numeric_map", "value_counts"]:
                    processed_chunk, operation_state[args.operation] = OPERATION_HANDLERS[args.operation](
                        chunk_df, args, input_sep, is_header_present, row_idx_col_name,
                        state=operation_state.get(args.operation) # Pass current state
                    )
                    if args.operation == "value_counts" and sys.exit.called: # Check if value_counts exited
                        return
                else:
                    processed_chunk = OPERATION_HANDLERS[args.operation](processed_chunk, args, input_sep, is_header_present, row_idx_col_name)

                # Reorder row index column if specified
                if row_idx_col_name and row_idx_col_name in processed_chunk.columns:
                    cols = [row_idx_col_name] + [col for col in processed_chunk.columns if col != row_idx_col_name]
                    processed_chunk = processed_chunk[cols]

                output_header_printed = _write_output_data(processed_chunk, args, input_sep, is_header_present, output_header_printed)
            
            # For chunked operations that need a final print/action after all chunks (e.g., value_counts if not exited yet)
            if args.operation == "numeric_map" and operation_state.get('numeric_map'):
                # Print the final mapping for numeric_map after all chunks are processed
                numeric_map_unique_values, _ = operation_state['numeric_map']
                _print_verbose(args, f"Final Mapping (Value -> Numeric ID):")
                unique_values = sorted(numeric_map_unique_values.items(), key=lambda item: item[1])
                for val, idx in unique_values:
                    _print_verbose(args, f"  '{val}' -> {idx}")


        else: # Full DataFrame processing
            processed_df = df_or_chunks # df_or_chunks is a DataFrame here

            # Handle operation execution
            # Pass raw_first_line_content to viewheader as it needs it when no header is present
            if args.operation in ["view", "viewheader"]:
                OPERATION_HANDLERS[args.operation](processed_df, args, input_sep, is_header_present, row_idx_col_name, raw_first_line_content=raw_first_line_content)
            elif args.operation in ["numeric_map", "value_counts"]: # These return state for full DF
                processed_df, _ = OPERATION_HANDLERS[args.operation](
                    processed_df, args, input_sep, is_header_present, row_idx_col_name, state=None
                )
            else:
                processed_df = OPERATION_HANDLERS[args.operation](processed_df, args, input_sep, is_header_present, row_idx_col_name)

            # Reorder row index column if specified
            if row_idx_col_name and row_idx_col_name in processed_df.columns:
                cols = [row_idx_col_name] + [col for col in processed_df.columns if col != row_idx_col_name]
                processed_df = processed_df[cols]
                _print_verbose(args, f"Reordered final output to place row index column '{row_idx_col_name}' first.")

            _write_output_data(processed_df, args, input_sep, is_header_present, output_header_printed) # output_header_printed will be False initially
            
    except (ValueError, IndexError, FileNotFoundError, RuntimeError) as e:
        sys.stderr.write(f"{e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"An unexpected error occurred during processing: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
