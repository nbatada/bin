#!/usr/bin/env python3
import sys
import argparse
import re
import pandas as pd
import codecs  # For handling escape sequences
from io import StringIO  # For piped input handling with pandas
from collections import Counter

# Default chunk size when processing input in low-memory mode.
CHUNK_SIZE = 10000

# --------------------------
# Utility Functions
# --------------------------
def _clean_string_for_header_and_data(s):
    """Cleans up a string for headers and data."""
    if not isinstance(s, str):
        return s
    s = s.lower()
    s = s.replace(' ', '_')
    s = re.sub(r'[^\w_]', '', s)
    s = re.sub(r'_{2,}', '_', s)
    return s

def get_unique_header(candidate, df):
    """Returns a unique header name given a candidate and the current DataFrame headers."""
    if candidate not in df.columns:
        return candidate
    base = candidate
    i = 1
    while f"{base}_{i}" in df.columns:
        i += 1
    return f"{base}_{i}"

def _print_verbose(args, message):
    """Prints verbose output if enabled."""
    if args.verbose:
        sys.stderr.write(f"VERBOSE: {message}\n")

def _parse_column_arg(value, df_columns, is_header_present, arg_name="column"):
    """
    Parses a column argument (1-indexed numeric or header name) and returns the corresponding 0-indexed column.
    """
    try:
        col_idx = int(value)
        if col_idx < 1:
            raise ValueError(f"Error: {arg_name} index '{value}' must be >= 1 (1-indexed).")
        if col_idx - 1 > len(df_columns):
            raise IndexError(f"Error: {arg_name} index '{value}' is out of bounds. Max is {len(df_columns)}.")
        return col_idx - 1
    except ValueError:
        if not is_header_present:
            raise ValueError(f"Error: Cannot use column name '{value}' for {arg_name} when no header is present (--header=None). Use a 1-indexed integer.")
        if value not in df_columns:
            raise ValueError(f"Error: Column '{value}' not found in header for {arg_name}. Available: {list(df_columns)}.")
        return df_columns.get_loc(value)
    except IndexError as e:
        raise e

def _parse_multiple_columns_arg(values, df_columns, is_header_present, arg_name="columns"):
    """
    Parses a comma-separated string of columns (numeric or names) and returns a list of their 0-indexed positions.
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
            raise type(e)(f"Error parsing {arg_name} '{values}': {e}")
    return col_indices


###--
def _setup_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "A command-line tool for manipulating table fields.\n\n"
            "Usage:\n"
            "    tbl.py [GLOBAL_OPTIONS] <operation> [OPERATION_SPECIFIC_OPTIONS]\n\n"
            "Global Options affect how input is read and overall behavior.\n"
            "Operations perform specific data manipulations.\n"
            "Use 'tbl.py <operation> --help' for details on each."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Global Options
    global_options = parser.add_argument_group("Global Options")
    global_options.add_argument(
        "-f", "--file", type=argparse.FileType('r'), default=sys.stdin,
        help="Input file (default: stdin)."
    )
    global_options.add_argument(
        "-s", "--sep", default="\t",
        help="Field separator (default: tab). Supports escape sequences (e.g., '\\t', '\\n')."
    )
    global_options.add_argument(
        "--noheader", action="store_true",
        help="Indicate that the input does not have a header row. By default, the first row is used as the header."
    )
    global_options.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose debug output to stderr."
    )
    global_options.add_argument(
        "-r", "--row-index",
        help="Specify a column (1-indexed or name) to serve as the row identifier."
    )
    global_options.add_argument(
        "--lowmem", action="store_true",
        help="Process data in chunks to reduce memory usage. Not all operations support low-memory mode."
    )
    
    # Subparsers for operations
    subparsers = parser.add_subparsers(dest="operation", help="Available operations. Use 'tbl.py <operation> --help' for details.")
    
    # MOVE
    parser_move = subparsers.add_parser("move", help="Move a column. Required: -i and -j.")
    parser_move.add_argument("-i", required=True, help="Source column (1-indexed or name).")
    parser_move.add_argument("-j", required=True, help="Destination column (1-indexed or name).")
    
    # COL_INSERT
    parser_col_insert = subparsers.add_parser("col_insert", help="Insert a new column. Required: -i and -v.")
    parser_col_insert.add_argument("-i", required=True, help="Column position (1-indexed or name) for insertion.")
    parser_col_insert.add_argument("-v", "--value", required=True, help="Value to populate the new column.")
    parser_col_insert.add_argument("--new-header", default="new_column", help="Header name for the new column (default: 'new_column').")
    
    # COL_DROP
    parser_col_drop = subparsers.add_parser("col_drop", help="Drop columns. Required: -i.")
    parser_col_drop.add_argument("-i", required=True, help="Comma-separated list of column(s) (1-indexed or names) to drop. Use 'all' to drop all columns.")
    
    # GREP
    parser_grep = subparsers.add_parser("grep", help="Filter rows. Required: -i and one of -p, --starts-with, --ends-with, or --word-file.")
    grep_group = parser_grep.add_mutually_exclusive_group(required=True)
    parser_grep.add_argument("-i", required=True, help="Column to apply the grep filter (1-indexed or name).")
    grep_group.add_argument("-p", "--pattern", help="Regex pattern to search for.")
    grep_group.add_argument("--starts-with", help="String that the column value should start with.")
    grep_group.add_argument("--ends-with", help="String that the column value should end with.")
    grep_group.add_argument("--word-file", help="File containing words (one per line) to match against the column values.")
    parser_grep.add_argument("--substring-match", action="store_true", 
                               help="Allow substring matching when using a word file.")
    parser_grep.add_argument("--tokenize", action="store_true", 
                               help="Split the target field on '.' and use the first token for matching.")
    parser_grep.add_argument("--list-missing-words", action="store_true",
                               help="Report words from the word file not found in the input.")
    parser_grep.add_argument("-v", "--invert", action="store_true",
                               help="Invert match: select rows that do NOT match the specified criteria.")
    
    # SPLIT
    parser_split = subparsers.add_parser("split", help="Split a column. Required: -i and -d.")
    parser_split.add_argument("-i", required=True, help="Column to split (1-indexed or name).")
    parser_split.add_argument("-d", "--delimiter", required=True, help="Delimiter to split the column by. Supports escape sequences.")
    parser_split.add_argument("--new-header-prefix", default="split_col", help="Prefix for the new columns (default: 'split_col').")
    
    # JOIN
    parser_join = subparsers.add_parser("join", help="Join columns. Required: -i. Optionally, -j specifies target column.")
    parser_join.add_argument("-i", required=True, help="Comma-separated list of columns (1-indexed or names) to join.")
    parser_join.add_argument("-d", "--delimiter", default="", help="Delimiter to insert between joined values (default: no delimiter). Supports escape sequences.")
    parser_join.add_argument("--new-header", default="joined_column", help="Header for the resulting joined column (default: 'joined_column').")
    parser_join.add_argument("-j", "--target-col-idx", help="Target column (1-indexed or name) where the joined column will be placed.")
    
    # TR (Translate)
    parser_tr = subparsers.add_parser("tr", help="Translate values. Required: -i and either -d (dict file) or --from-val with --to-val.")
    parser_tr.add_argument("-i", required=True, help="Column to translate (1-indexed or name).")
    tr_group = parser_tr.add_mutually_exclusive_group(required=True)
    tr_group.add_argument("-d", "--dict-file", help="Path to a two-column file (key<sep>value) for mapping. Uses the main --sep as separator.")
    tr_group.add_argument("--from-val", help="Value to translate from (for single translation). Supports escape sequences.")
    parser_tr.add_argument("--to-val", help="Value to translate to (for single translation). Supports escape sequences.")
    parser_tr.add_argument("--regex", action="store_true", help="Treat --from-val as a regex pattern (default is literal).")
    parser_tr.add_argument("--new-header", default="_translated", help="Suffix or new header for the translated column (default: '_translated').")
    parser_tr.add_argument("--in-place", action="store_true", help="Replace the original column with the translated values.")
    
    # SORT
    parser_sort = subparsers.add_parser("sort", help="Sort table. Required: -i. (Not compatible with lowmem mode.)")
    parser_sort.add_argument("-i", required=True, help="Column to sort by (1-indexed or name). Only one column should be specified.")
    parser_sort.add_argument("--desc", action="store_true", help="Sort in descending order (default is ascending).")
    parser_sort.add_argument("-p", "--pattern", help="Regex pattern to extract a numeric key from the column.")
    parser_sort.add_argument("--suffix-map", help="Comma-separated key:value pairs for suffix mapping (used with --pattern).")
    parser_sort.add_argument("--expand-scientific", action="store_true", 
                             help="Use a default mapping for scientific suffixes and override any supplied --suffix-map.")
    parser_sort.add_argument("-ps", "--pattern-string", help="Optional additional regex pattern for a secondary string sort key.")
    
    # CLEANUP HEADER
    parser_cleanup_header = subparsers.add_parser("cleanup_header", help="Clean header names (lowercase, remove special characters, replace spaces with underscores).")
    
    # CLEANUP VALUES
    parser_cleanup_values = subparsers.add_parser("cleanup_values", help="Clean values in specified columns. Required: -i.")
    parser_cleanup_values.add_argument("-i", required=True, help="Comma-separated list of columns (1-indexed or names) to clean. Use 'all' to clean every column.")
    
    # PREFIX ADD
    parser_prefix_add = subparsers.add_parser("prefix_add", help="Add a prefix to column values. Required: -i and -v.")
    parser_prefix_add.add_argument("-i", required=True, help="Comma-separated list of columns (1-indexed or names) to prepend with a prefix. Use 'all' for every column.")
    parser_prefix_add.add_argument("-v", "--string", required=True, help="The prefix string to add. Supports escape sequences.")
    parser_prefix_add.add_argument("-d", "--delimiter", default="", help="Delimiter to insert between the prefix and the original value (default: none). Supports escape sequences.")
    
    # VALUE COUNTS
    parser_value_counts = subparsers.add_parser("value_counts", help="Count top occurring values. Required: -i.")
    parser_value_counts.add_argument("-n", "--top-n", type=int, default=5, help="Number of top values to display (default: 5).")
    parser_value_counts.add_argument("-i", required=True, help="Comma-separated list of columns (1-indexed or names) to count. Use 'all' for every column.")
    
    # STRIP
    parser_strip = subparsers.add_parser("strip", help="Remove a regex pattern from column values. Required: -i and -p.")
    parser_strip.add_argument("-i", required=True, help="Column to strip (1-indexed or name).")
    parser_strip.add_argument("-p", "--pattern", required=True, help="Regex pattern to remove from the column values.")
    parser_strip.add_argument("--new-header", default="_stripped", help="Suffix or new header for the column after stripping (default: '_stripped').")
    parser_strip.add_argument("--in-place", action="store_true", help="Modify the column in place instead of creating a new column.")
    
    # NUMERIC MAP
    parser_numeric_map = subparsers.add_parser("numeric_map", help="Map unique string values to numbers. Required: -i.")
    parser_numeric_map.add_argument("-i", required=True, help="Column (1-indexed or name) whose unique values are to be mapped to numbers.")
    parser_numeric_map.add_argument("--new-header", help="Header for the new numeric mapping column (default: 'numeric_map_of_ORIGINAL_COLUMN_NAME').")
    
    # REGEX CAPTURE
    parser_regex_capture = subparsers.add_parser("regex_capture", help="Capture substrings using a regex capturing group. Required: -i and -p.")
    parser_regex_capture.add_argument("-i", required=True, help="Column on which to apply the regex (1-indexed or name).")
    parser_regex_capture.add_argument("-p", "--pattern", required=True, help="Regex pattern with at least one capturing group (e.g., '_(S[0-9]+)\\.' ).")
    parser_regex_capture.add_argument("--new-header", default="_captured", help="Suffix or new header for the captured column (default: '_captured').")
    
    # VIEW
    parser_view = subparsers.add_parser("view", help="Display the data in a formatted table.")
    parser_view.add_argument("--max-rows", type=int, default=20, help="Maximum number of rows to display (default: 20).")
    parser_view.add_argument("--max-cols", type=int, default=None, help="Maximum number of columns to display (default: all columns).")
    
    # CUT
    parser_cut = subparsers.add_parser("cut", help="Cut/select columns. Required: -p.")
    parser_cut.add_argument("-p", "--pattern", required=True, help="String or regex pattern to match column names for selection.")
    parser_cut.add_argument("--regex", action="store_true", help="Interpret the pattern as a regex (default is a literal match).")
    
    # VIEWHEADER
    parser_viewheader = subparsers.add_parser("viewheader", help="Display header names and positions.")
    
    # ROW_INSERT
    parser_row_insert = subparsers.add_parser("row_insert", help="Insert a new row at a specified 1-indexed position. Use -i 0 to insert at the header.")
    parser_row_insert.add_argument("-i", "--row-idx", type=int, default=0, help="Row position for insertion (1-indexed, 0 for header insertion).")
    parser_row_insert.add_argument("-v", "--values", help="Comma-separated list of values for the new row. Supports escape sequences.")
    
    # ROW_DROP
    parser_row_drop = subparsers.add_parser("row_drop", help="Delete row(s) at a specified 1-indexed position. Use -i 0 to drop the header row.")
    parser_row_drop.add_argument("-i", "--row-idx", type=int, required=True, help="Row position to drop (1-indexed, 0 drops the header).")
    
    return parser

###---
# --------------------------
# Operation Handler Functions
# --------------------------
def _handle_move(df, args, input_sep, is_header_present, row_idx_col_name):
    from_idx = _parse_column_arg(args.i, df.columns, is_header_present, "source column (-i)")
    to_idx = _parse_column_arg(args.j, df.columns, is_header_present, "destination column (-j)")
    if to_idx > df.shape[1]:
        to_idx = df.shape[1]
    _print_verbose(args, f"Moving column '{df.columns[from_idx]}' from index {from_idx} to {to_idx}.")
    col_name = df.columns[from_idx]
    data = df.pop(col_name)
    df.insert(to_idx, col_name, data)
    return df

def _handle_col_insert(df, args, input_sep, is_header_present, row_idx_col_name):
    pos = _parse_column_arg(args.i, df.columns, is_header_present, "column (-i)")
    value = codecs.decode(args.value, 'unicode_escape')
    new_header = args.new_header
    if is_header_present and new_header in df.columns:
        new_header = get_unique_header(new_header, df)
        _print_verbose(args, f"New header exists; using '{new_header}'.")
    df.insert(pos, new_header, value)
    return df

def _handle_col_drop(df, args, input_sep, is_header_present, row_idx_col_name):
    indices = _parse_multiple_columns_arg(args.i, df.columns, is_header_present, "columns (-i)")
    names = [df.columns[i] for i in indices]
    _print_verbose(args, f"Dropping columns: {names}.")
    df = df.drop(columns=names)
    return df

def _handle_grep(df, args, input_sep, is_header_present, row_idx_col_name, state=None):
    col = _parse_column_arg(args.i, df.columns, is_header_present, "column (-i)")
    series = df.iloc[:, col].astype(str)
    if args.word_file:
        try:
            with open(args.word_file, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Error reading word file '{args.word_file}': {e}")
        if not words:
            sys.stderr.write(f"Warning: Word file '{args.word_file}' is empty. No filtering performed.\n")
            return df, state
        if args.list_missing_words:
            if state is None:
                state = {}
            if "matched_words" not in state:
                state["matched_words"] = set()
        if args.tokenize:
            tokens = series.str.split('.', n=1).str[0]
            if args.substring_match:
                mask = tokens.apply(lambda token: any(word in token for word in words))
                if args.list_missing_words:
                    chunk_matches = {w for token in tokens.dropna() for w in words if w in token}
                    state["matched_words"].update(chunk_matches)
            else:
                mask = tokens.isin(words)
                if args.list_missing_words:
                    state["matched_words"].update(set(tokens.dropna().unique()).intersection(words))
        else:
            if args.substring_match:
                pattern = "(" + "|".join(map(re.escape, words)) + ")"
            else:
                pattern = r"\b(?:" + "|".join(map(re.escape, words)) + r")\b"
            mask = series.str.contains(pattern, regex=True, na=False)
            if args.list_missing_words:
                chunk_matches = set()
                try:
                    for text in series.dropna():
                        found = re.findall(pattern, text)
                        chunk_matches.update(found)
                except re.error as e:
                    raise ValueError(f"Error processing regex pattern '{pattern}': {e}")
                state["matched_words"].update(chunk_matches)
        df = df[~mask] if args.invert else df[mask]
    elif args.pattern:
        try:
            mask = series.str.contains(args.pattern, regex=True, na=False)
        except re.error as e:
            raise ValueError(f"Error: Invalid regex pattern '{args.pattern}': {e}")
        df = df[~mask] if args.invert else df[mask]
    elif args.starts_with:
        mask = series.str.startswith(args.starts_with, na=False)
        df = df[~mask] if args.invert else df[mask]
    elif args.ends_with:
        mask = series.str.endswith(args.ends_with, na=False)
        df = df[~mask] if args.invert else df[mask]
    return df, state

def _handle_split(df, args, input_sep, is_header_present, row_idx_col_name):
    col = _parse_column_arg(args.i, df.columns, is_header_present, "column (-i)")
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    original = df.columns[col]
    _print_verbose(args, f"Splitting column '{original}' with delimiter {delim!r}.")
    split_df = df.iloc[:, col].astype(str).str.split(delim, expand=True).fillna('')
    new_headers = []
    for i in range(split_df.shape[1]):
        candidate = f"{original}_{args.new_header_prefix}_{i+1}"
        new_headers.append(get_unique_header(candidate, df) if is_header_present else f"{args.new_header_prefix}_{i+1}")
    split_df.columns = new_headers
    _print_verbose(args, f"New split headers: {new_headers}.")
    df = df.drop(columns=[original])
    left = df.iloc[:, :col]
    right = df.iloc[:, col:]
    df = pd.concat([left, split_df, right], axis=1)
    return df

def _handle_join(df, args, input_sep, is_header_present, row_idx_col_name):
    indices = _parse_multiple_columns_arg(args.i, df.columns, is_header_present, "columns (-i)")
    if not indices:
        raise ValueError("Error: No columns specified for join operation.")
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    names = [df.columns[i] for i in indices]
    _print_verbose(args, f"Joining columns {names} with delimiter {delim!r}.")
    joined = df.iloc[:, indices[0]].astype(str)
    for i in range(1, len(indices)):
        joined += delim + df.iloc[:, indices[i]].astype(str)
    if args.target_col_idx:
        target = _parse_column_arg(args.target_col_idx, df.columns, is_header_present, "target column (-j)")
    else:
        target = indices[0]
    new_header = args.new_header
    if is_header_present:
        new_header = get_unique_header(new_header, df)
    drop_names = [df.columns[i] for i in sorted(indices, reverse=True)]
    df = df.drop(columns=drop_names)
    pos = min(target, df.shape[1])
    df.insert(pos, new_header, joined.reset_index(drop=True))
    _print_verbose(args, f"Inserted joined column '{new_header}' at index {pos}.")
    return df

def _handle_tr(df, args, input_sep, is_header_present, row_idx_col_name):
    col = _parse_column_arg(args.i, df.columns, is_header_present, "column (-i)")
    original = df.columns[col]
    translated = None
    if args.dict_file:
        mapping = {}
        try:
            _print_verbose(args, f"Loading translation mapping from '{args.dict_file}' using separator {input_sep!r}.")
            with open(args.dict_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(input_sep, 1)
                    if len(parts) == 2:
                        mapping[parts[0]] = parts[1]
                    else:
                        sys.stderr.write(f"Warning: Skipping malformed line: '{line}'.\n")
            _print_verbose(args, f"Loaded {len(mapping)} translations.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Dictionary file not found: '{args.dict_file}'")
        except Exception as e:
            raise RuntimeError(f"Error reading dictionary file: {e}")
        translated = df.iloc[:, col].astype(str).apply(lambda x: mapping.get(x, x))
    elif args.from_val:
        if not args.to_val:
            raise ValueError("Error: Must specify --to-val when using --from-val.")
        from_val = codecs.decode(args.from_val, 'unicode_escape')
        to_val = codecs.decode(args.to_val, 'unicode_escape')
        _print_verbose(args, f"Translating values in '{original}' from '{from_val}' to '{to_val}' ({'regex' if args.regex else 'literal'}).")
        try:
            translated = df.iloc[:, col].astype(str).str.replace(from_val, to_val, regex=args.regex)
        except re.error as e:
            raise ValueError(f"Error: Invalid regex '{from_val}': {e}")
    else:
        raise ValueError("Error: For tr operation, specify either a dict file (-d) or both --from-val and --to-val.")
    if args.in_place:
        df.iloc[:, col] = translated
    else:
        if args.new_header.startswith("_"):
            new_header = str(original) + args.new_header
        else:
            new_header = args.new_header
        if is_header_present:
            new_header = get_unique_header(new_header, df)
        df.insert(col+1, new_header, translated.reset_index(drop=True))
        _print_verbose(args, f"Inserted translated column '{new_header}' after '{original}'.")
    return df

def extract_numeric(value, pattern, suffix_map=None):
    if not isinstance(value, str):
        return None
    m = re.search(pattern, value)
    if m:
        if suffix_map:
            return _parse_size_value(m.group(0), suffix_map)
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def _parse_size_value(value, suffix_map):
    m = re.match(r"(\d+(?:\.\d+)?)([KMGTP])?", value, re.IGNORECASE)
    if not m:
        try:
            return float(value)
        except ValueError:
            return None
    num = float(m.group(1))
    suffix = m.group(2)
    if suffix:
        key = suffix.upper()
        if key in suffix_map:
            return num * suffix_map[key]
        else:
            return None
    return num

def _handle_sort(df, args, input_sep, is_header_present, row_idx_col_name):
    cols = _parse_multiple_columns_arg(args.i, df.columns, is_header_present, "column (-i)")
    if len(cols) != 1:
        raise ValueError("Error: Sort operation requires a single column specified by -i.")
    target = df.columns[cols[0]]
    # Use default regex if none provided.
    if args.expand_scientific and not args.pattern:
        args.pattern = r"(\d+(?:\.\d+)?)([KMGTP])?"
    elif not args.pattern:
        args.pattern = r"(\d+)"
    if args.expand_scientific:
        suffix_map = {'K': 1000, 'M': 1000000, 'G': 1000000000, 'T': 1000000000000, 'P': 1000000000000000}
    elif args.suffix_map:
        suffix_map = {}
        for pair in args.suffix_map.split(','):
            if ':' in pair:
                key, val = pair.split(':', 1)
                try:
                    suffix_map[key.strip().upper()] = float(val.strip())
                except ValueError:
                    raise ValueError(f"Error: Invalid suffix-map entry '{pair}'.")
            else:
                raise ValueError(f"Error: Suffix-map entry '{pair}' is malformed. Expected KEY:VALUE.")
    else:
        suffix_map = {}
    _print_verbose(args, f"Sorting column '{target}' using pattern '{args.pattern}' with suffix map {suffix_map}.")
    temp_numeric = f"_temp_numeric_sort_{target}"
    def debug_conversion(val):
        original_val = str(val)
        cleaned_val = original_val.strip()
        numeric_val = extract_numeric(cleaned_val, args.pattern, suffix_map)
        _print_verbose(args, f"DEBUG: Original '{original_val}' => Cleaned '{cleaned_val}' => Numeric {numeric_val}")
        return numeric_val
    df[temp_numeric] = df[target].astype(str).apply(debug_conversion)
    sort_keys = [temp_numeric]
    if args.pattern_string:
        temp_secondary = f"_temp_string_sort_{target}"
        def extract_secondary(val):
            m = re.search(args.pattern_string, val)
            return m.group(1) if m else val
        df[temp_secondary] = df[target].astype(str).apply(extract_secondary)
        sort_keys.append(temp_secondary)
    sort_keys.append(target)
    ascending = not args.desc
    df = df.sort_values(by=sort_keys, ascending=ascending, kind='stable')
    drop_cols = [temp_numeric]
    if args.pattern_string:
        drop_cols.append(temp_secondary)
    df.drop(columns=drop_cols, inplace=True)
    return df

def _handle_cleanup_header(df, args, input_sep, is_header_present, row_idx_col_name):
    if not is_header_present:
        sys.stderr.write("Warning: '--header=None' provided. 'cleanup_header' will have no effect.\n")
        _print_verbose(args, "Skipping cleanup_header (no header).")
    else:
        original = list(df.columns)
        df.columns = [_clean_string_for_header_and_data(col) for col in df.columns]
        _print_verbose(args, f"Cleaned header. Before: {original}  After: {list(df.columns)}")
    return df

def _handle_cleanup_values(df, args, input_sep, is_header_present, row_idx_col_name):
    indices = _parse_multiple_columns_arg(args.i, df.columns, is_header_present, "columns (-i)")
    names = [df.columns[i] for i in indices]
    _print_verbose(args, f"Cleaning values in columns: {names}.")
    for i in indices:
        df.iloc[:, i] = df.iloc[:, i].apply(_clean_string_for_header_and_data)
    return df

def _handle_prefix_add(df, args, input_sep, is_header_present, row_idx_col_name):
    indices = _parse_multiple_columns_arg(args.i, df.columns, is_header_present, "columns (-i)")
    prefix = codecs.decode(args.string, 'unicode_escape')
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    names = [df.columns[i] for i in indices]
    _print_verbose(args, f"Adding prefix '{prefix}' with delimiter '{delim}' to columns: {names}.")
    for i in indices:
        df.iloc[:, i] = df.iloc[:, i].astype(str).apply(lambda x: f"{prefix}{delim}{x}")
    return df

def _handle_value_counts(df, args, input_sep, is_header_present, row_idx_col_name, state=None):
    if state is None:
        counter = Counter()
    else:
        counter = state
    indices = _parse_multiple_columns_arg(args.i, df.columns, is_header_present, "columns (-i)")
    if not indices:
        raise ValueError("Error: No columns specified for value_counts.")
    for i in indices:
        col = df.columns[i]
        counter.update(df[col].astype(str))
    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    summary = pd.DataFrame(sorted_counts[:args.top_n], columns=['Value', 'Count'])
    sys.stdout.write(f"--- Top {args.top_n} Values ---\n")
    sys.stdout.write(summary.to_string(index=False, header=True) + '\n')
    sys.exit(0)

def _handle_strip(df, args, input_sep, is_header_present, row_idx_col_name):
    col = _parse_column_arg(args.i, df.columns, is_header_present, "column (-i)")
    original = df.columns[col]
    try:
        new_series = df.iloc[:, col].astype(str).str.replace(args.pattern, '', regex=True)
    except re.error as e:
        raise ValueError(f"Error: Invalid regex pattern '{args.pattern}': {e}")
    if args.in_place:
        df.iloc[:, col] = new_series
    else:
        if args.new_header.startswith("_"):
            new_header = str(original) + args.new_header
        else:
            new_header = args.new_header
        if is_header_present:
            new_header = get_unique_header(new_header, df)
        df.insert(col+1, new_header, new_series.reset_index(drop=True))
        _print_verbose(args, f"Inserted stripped column '{new_header}' after '{original}'.")
    return df

def _handle_numeric_map(df, args, input_sep, is_header_present, row_idx_col_name, state=None):
    if state is None:
        mapping = {}
        next_id = 1
    else:
        mapping, next_id = state
    col = _parse_column_arg(args.i, df.columns, is_header_present, "column (-i)")
    original = df.columns[col]
    _print_verbose(args, f"Mapping unique values in '{original}' to numeric identifiers.")
    new_header = args.new_header if args.new_header else f"numeric_map_of_{original}"
    if is_header_present:
        new_header = get_unique_header(new_header, df)
    def mapper(x):
        nonlocal next_id
        if x not in mapping:
            mapping[x] = next_id
            next_id += 1
        return mapping[x]
    mapped = df.iloc[:, col].astype(str).apply(mapper)
    df.insert(col+1, new_header, mapped)
    return df, (mapping, next_id)

def _handle_regex_capture(df, args, input_sep, is_header_present, row_idx_col_name, raw_first_line_content=None):
    col = _parse_column_arg(args.i, df.columns, is_header_present, "column (-i)")
    base = df.columns[col] if is_header_present else f"captured_column_{col+1}"
    if args.new_header.startswith("_"):
        new_header = str(base) + args.new_header
    else:
        new_header = args.new_header
    if is_header_present:
        new_header = get_unique_header(new_header, df)
        _print_verbose(args, f"Using unique header: {new_header}.")
    _print_verbose(args, f"Capturing groups using regex '{args.pattern}' on column '{base}'.")
    try:
        captured = df.iloc[:, col].astype(str).apply(lambda x: ';'.join(re.findall(args.pattern, x)))
    except re.error as e:
        raise ValueError(f"Error: Invalid regex pattern '{args.pattern}': {e}")
    df.insert(len(df.columns), new_header, captured.reset_index(drop=True))
    return df

def _handle_view(df, args, input_sep, is_header_present, row_idx_col_name, raw_first_line_content=None):
    _print_verbose(args, f"Viewing data (max rows: {args.max_rows}, max cols: {args.max_cols}).")
    pd.set_option('display.max_rows', args.max_rows)
    pd.set_option('display.max_columns', args.max_cols)
    pd.set_option('display.width', None)
    pd.set_option('display.colheader_justify', 'left')
    disp = df.copy()
    if row_idx_col_name and row_idx_col_name in disp.columns:
        cols = [row_idx_col_name] + [col for col in disp.columns if col != row_idx_col_name]
        disp = disp[cols]
        _print_verbose(args, f"Moved row-index column '{row_idx_col_name}' to the front.")
    sys.stdout.write(disp.to_string(index=True, header=is_header_present) + '\n')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.colheader_justify')
    sys.exit(0)

def _handle_cut(df, args, input_sep, is_header_present, row_idx_col_name):
    pattern = args.pattern
    _print_verbose(args, f"Selecting columns matching pattern '{pattern}' (regex: {args.regex}).")
    selected = []
    for col in df.columns:
        if args.regex:
            try:
                if re.search(pattern, str(col)):
                    selected.append(col)
            except re.error as e:
                raise ValueError(f"Error: Invalid regex '{pattern}': {e}")
        else:
            if pattern in str(col):
                selected.append(col)
    if row_idx_col_name and row_idx_col_name in df.columns and row_idx_col_name not in selected:
        selected = [row_idx_col_name] + selected
        _print_verbose(args, f"Including row-index column '{row_idx_col_name}' in the output.")
    if not selected:
        sys.stderr.write(f"Warning: No columns matched pattern '{pattern}'. Output will be empty.\n")
        df = pd.DataFrame(columns=[])
    else:
        df = df[selected]
        _print_verbose(args, f"Columns selected: {selected}.")
    return df

def _handle_viewheader(df, args, input_sep, is_header_present, row_idx_col_name, raw_first_line_content):
    _print_verbose(args, "Listing header names with 1-indexed positions.")
    out_lines = []
    if df.empty and df.columns.size:
        for i, col in enumerate(df.columns):
            out_lines.append(f"{i+1}\t{col}")
    elif not df.empty:
        cols = list(df.columns)
        if not is_header_present and raw_first_line_content:
            for i, val in enumerate(raw_first_line_content):
                indicator = " (Row Index)" if row_idx_col_name and cols[i] == row_idx_col_name else ""
                out_lines.append(f"{i+1}\t{val}{indicator}")
        else:
            for i, col in enumerate(cols):
                indicator = " (Row Index)" if row_idx_col_name and col == row_idx_col_name else ""
                out_lines.append(f"{i+1}\t{col}{indicator}")
    if not out_lines and is_header_present:
        sys.stderr.write("No headers found. The input might be empty or malformed.\n")
    else:
        sys.stdout.write("\n".join(out_lines) + '\n')
    sys.exit(0)

def _handle_row_insert(df, args, input_sep, is_header_present, row_idx_col_name):
    insert_pos = args.row_idx - 1  # 0-indexed
    if args.values:
        values = [codecs.decode(v.strip(), 'unicode_escape') for v in args.values.split(',')]
    elif df.empty:
        sys.stderr.write("Error: Cannot insert row into an empty table without explicit values.\n")
        sys.exit(1)
    else:
        values = [f"col{i+1}" for i in range(df.shape[1])]
        _print_verbose(args, f"Generated default row values: {values}")
    if len(values) > df.shape[1]:
        values = values[:df.shape[1]]
        _print_verbose(args, "Truncated row values to match column count.")
    elif len(values) < df.shape[1]:
        values.extend([''] * (df.shape[1] - len(values)))
        _print_verbose(args, "Padded row values to match column count.")
    new_row = pd.DataFrame([values], columns=df.columns)
    if insert_pos >= len(df):
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.concat([df.iloc[:insert_pos], new_row, df.iloc[insert_pos:]], ignore_index=True)
    _print_verbose(args, f"Inserted row at position {args.row_idx}. New shape: {df.shape}.")
    return df

def _handle_row_drop(df, args, input_sep, is_header_present, row_idx_col_name):
    drop_pos = args.row_idx - 1
    if drop_pos < 0 or drop_pos >= len(df):
        raise IndexError(f"Error: Row index {args.row_idx} is out of bounds. Table has {len(df)} rows.")
    df = df.drop(df.index[drop_pos]).reset_index(drop=True)
    _print_verbose(args, f"Dropped row at position {args.row_idx}. New shape: {df.shape}.")
    return df

# --------------------------
# Dispatch Table
# --------------------------
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
    "row_drop": _handle_row_drop,
}

# --------------------------
# Input/Output Functions
# --------------------------
def _read_input_data(args, input_sep, header_param, is_header_present, use_chunked):
    """Reads input into a DataFrame or a generator for chunked processing."""
    raw_first_line = []
    input_stream = args.file  # use file provided by -f/--file
    if use_chunked:
        try:
            reader = pd.read_csv(input_stream, sep=input_sep, header=header_param, dtype=str,
                                 chunksize=CHUNK_SIZE, iterator=True)
            first_chunk = next(reader)
            if first_chunk.empty and args.operation not in ["viewheader", "view", "value_counts", "regex_capture"]:
                sys.stderr.write(f"Error: Input is empty. '{args.operation}' requires non-empty data.\n")
                sys.exit(1)
            if not is_header_present:
                first_chunk.columns = pd.Index(range(first_chunk.shape[1]))
            def generator():
                yield first_chunk
                for chunk in reader:
                    yield chunk
            return generator(), raw_first_line
        except (StopIteration, pd.errors.EmptyDataError):
            sys.stderr.write(f"Warning: Input is empty. Cannot perform '{args.operation}'.\n")
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"Error reading input in low-memory mode: {e}\n")
            sys.exit(1)
    else:
        try:
            content = input_stream.read()
            if not content.strip() and args.operation not in ["viewheader", "view", "value_counts"]:
                sys.stderr.write(f"Error: Input is empty. '{args.operation}' requires data.\n")
                sys.exit(1)
            if not content.strip():
                return pd.DataFrame(columns=[]), raw_first_line
            csv_io = StringIO(content)
            if not is_header_present:
                pos = csv_io.tell()
                first_line = csv_io.readline().strip()
                raw_first_line = first_line.split(input_sep) if first_line else []
                csv_io.seek(pos)
            df = pd.read_csv(csv_io, sep=input_sep, header=header_param, dtype=str)
            return df, raw_first_line
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=[])
            _print_verbose(args, "Empty input; proceeding with an empty DataFrame.")
            return df, raw_first_line
        except Exception as e:
            sys.stderr.write(f"Error reading input data: {e}\n")
            sys.exit(1)

import csv  # Add this import near the top of your script

def _write_output_data(data, args, input_sep, is_header_present, header_printed):
    try:
        # For operations such as view we do not output with CSV formatting.
        if args.operation in ["view", "viewheader", "value_counts"]:
            return header_printed
        if isinstance(data, pd.DataFrame):
            data.to_csv(
                sys.stdout,
                sep=input_sep,
                index=False,
                header=is_header_present,
                encoding='utf-8',
                quoting=csv.QUOTE_NONE,  # disable automatic quoting
                escapechar='\\'
            )
            return True
        if not header_printed and is_header_present:
            data.to_csv(
                sys.stdout,
                sep=input_sep,
                index=False,
                header=True,
                encoding='utf-8',
                quoting=csv.QUOTE_NONE,
                escapechar='\\'
            )
            return True
        else:
            data.to_csv(
                sys.stdout,
                sep=input_sep,
                index=False,
                header=False,
                encoding='utf-8',
                quoting=csv.QUOTE_NONE,
                escapechar='\\'
            )
            return header_printed
    except BrokenPipeError:
        pass
    except Exception as e:
        sys.stderr.write(f"Error writing output: {e}\n")
        sys.exit(1)
    return header_printed

###--
def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()
    if not args.operation:
        parser.print_help()
        sys.exit(0)
    
    # Use --noheader to determine the header handling.
    if args.noheader:
        header_param = None
        is_header_present = False
    else:
        header_param = 0  # Use the first row as header.
        is_header_present = True

    input_sep = codecs.decode(args.sep, 'unicode_escape')
    
    # In low-memory mode, some operations (such as sort) are not allowed.
    if args.lowmem and args.operation == "sort":
        sys.stderr.write("Error: 'sort' operation cannot be performed in low-memory mode (--lowmem).\n")
        sys.exit(1)
    
    lowmem_ops = ["value_counts", "numeric_map", "grep", "tr", "strip", "prefix_add", "cleanup_values", "regex_capture"]
    use_chunked = args.lowmem and args.operation in lowmem_ops

    # Special handling for low-memory row operations.
    lowmem_row = False
    if args.operation in ["row_insert", "row_drop"]:
        if args.row_idx == 0:
            lowmem_row = True
    if args.lowmem and args.operation in ["row_insert", "row_drop"] and not lowmem_row:
        sys.stderr.write(f"Error: '{args.operation}' is not compatible with --lowmem except for row_idx 0.\n")
        sys.exit(1)

    if lowmem_row:
        sys.stderr.write("# Processing: " + " ".join(sys.argv) + "\n")
        if args.operation == "row_insert":
            _print_verbose(args, "Low-memory row_insert (header insertion).")
            first_line = args.file.readline().strip()
            count = len(first_line.split(input_sep)) if first_line else (len(args.values.split(',')) if args.values else 1)
            values = [codecs.decode(v.strip(), 'unicode_escape') for v in args.values.split(',')] if args.values else [f"col{i+1}" for i in range(count)]
            if len(values) > count:
                values = values[:count]
            sys.stdout.write(input_sep.join(values) + '\n')
            if first_line:
                sys.stdout.write(first_line + '\n')
            for line in args.file:
                sys.stdout.write(line)
            sys.exit(0)
        elif args.operation == "row_drop":
            _print_verbose(args, "Low-memory row_drop (dropping first row).")
            try:
                first_line = args.file.readline()
                if not first_line.strip():
                    sys.stderr.write("Warning: Input may be empty when dropping first row.\n")
                for line in args.file:
                    sys.stdout.write(line)
            except BrokenPipeError:
                pass
            sys.exit(0)

    df_or_chunks, raw_first_line = _read_input_data(args, input_sep, header_param, is_header_present, use_chunked)

    row_idx_col = None
    if args.row_index:
        if use_chunked:
            try:
                first_chunk = next(df_or_chunks)
                cols = first_chunk.columns if is_header_present else pd.Index(range(first_chunk.shape[1]))
                idx = _parse_column_arg(args.row_index, cols, is_header_present, "row-index")
                row_idx_col = cols[idx]
                df_or_chunks = (item for item in [first_chunk] + list(df_or_chunks))
                _print_verbose(args, f"Row-index column: '{row_idx_col}' (index {idx}).")
            except StopIteration:
                sys.stderr.write("Error: Input is empty; cannot determine row-index column.\n")
                sys.exit(1)
        else:
            if isinstance(df_or_chunks, pd.DataFrame):
                idx = _parse_column_arg(args.row_index, df_or_chunks.columns, is_header_present, "row-index")
                row_idx_col = df_or_chunks.columns[idx]
                _print_verbose(args, f"Row-index column: '{row_idx_col}' (index {idx}).")
            else:
                sys.stderr.write("Error: Cannot determine row-index column in current mode.\n")
                sys.exit(1)

    header_printed = False
    op_state = {}
    handler = OPERATION_HANDLERS.get(args.operation)
    if handler is None:
        sys.stderr.write(f"Error: Unsupported operation '{args.operation}'.\n")
        sys.exit(1)

    # Dispatch operations with the correct number of arguments.
    if use_chunked:
        for chunk in df_or_chunks:
            if not is_header_present:
                chunk.columns = pd.Index(range(chunk.shape[1]))
            if args.operation in ["grep", "numeric_map"]:
                processed, op_state[args.operation] = handler(
                    chunk, args, input_sep, is_header_present, row_idx_col,
                    state=op_state.get(args.operation, {})
                )
            elif args.operation in ["view", "regex_capture", "viewheader"]:
                processed = handler(
                    chunk, args, input_sep, is_header_present, row_idx_col, raw_first_line
                )
            else:
                processed = handler(
                    chunk, args, input_sep, is_header_present, row_idx_col
                )
            if row_idx_col and row_idx_col in processed.columns:
                cols_order = [row_idx_col] + [col for col in processed.columns if col != row_idx_col]
                processed = processed[cols_order]
            header_printed = _write_output_data(processed, args, input_sep, is_header_present, header_printed)
        if args.operation == "grep" and args.word_file and args.list_missing_words:
            with open(args.word_file, 'r', encoding='utf-8') as f:
                word_list = [line.strip() for line in f if line.strip()]
            matched = op_state.get("grep", {}).get("matched_words", set())
            missing = set(word_list) - matched
            sys.stderr.write("Words not seen in input: (n=" + str(len(missing)) + ") " + ", ".join(sorted(missing)) + "\n")
    else:
        if args.operation in ["grep", "numeric_map"]:
            processed_df, state = handler(df_or_chunks, args, input_sep, is_header_present, row_idx_col)
        elif args.operation in ["view", "regex_capture", "viewheader"]:
            processed_df = handler(df_or_chunks, args, input_sep, is_header_present, row_idx_col, raw_first_line)
        else:
            processed_df = handler(df_or_chunks, args, input_sep, is_header_present, row_idx_col)
        if args.operation == "grep" and args.word_file and args.list_missing_words:
            with open(args.word_file, 'r', encoding='utf-8') as f:
                word_list = [line.strip() for line in f if line.strip()]
            matched = state.get("matched_words", set()) if state else set()
            missing = set(word_list) - matched
            sys.stderr.write("Words not seen in input: " + ", ".join(sorted(missing)) + "\n")
        _write_output_data(processed_df, args, input_sep, is_header_present, header_printed)

if __name__ == "__main__":
    main()
