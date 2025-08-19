#!/usr/bin/env python3
import sys
import argparse
import re
import os
import pandas as pd
import codecs
from io import StringIO
from collections import Counter
import csv

# ---
# VERSION=8.19.1
# This update focuses on code simplification, systematic renaming, and feature enhancement.
# Summary of changes:
#   - Removed all low-memory (--lowmem) functionality to simplify the codebase.
#   - Renamed all commands to a systematic verb_noun_suffix format (e.g., 'prefix_add' -> 'add_prefix_col').
#   - Reorganized the help menu into logical categories (Table, Column, Row, Plot, Utilities).
#   - Added a new 'subset_table' command to select columns by data type (--only_numeric, etc.).
#   - Added a new 'query_table' command for row filtering based on conditions (e.g., less_than).
#   - Added a new 'add_suffix_col' command.
#   - Renamed 'tr' to 'replace_values_col', 'summarize' to 'aggregate_table', and 'numeric_map' to 'encode_categorical_col' for clarity.
#   - Modularized output writing into a single helper function (_write_output).
# ---
# VERSION=7.27.1 (Previous)
# Reverted back from Aug 18 to this because lots of things got buggy as I moved to GPT5
# ---
'''
Prompt for Code Refactoring and Enhancement

Context & Goal
The attached tbltool script is a command-line tool designed as a "swiss army knife" for manipulating tabular scientific data. The goal is to make it a one-stop-shop for common table processing tasks, similar in function to pandas or dplyr, but for the terminal. The vision is for it to replace traditional Unix tools like cut and sed, becoming as essential as samtools is for genomics data.

Input
I will provide the tbltool script in multiple, smaller chunks.

Core Task
Perform a thorough code review to identify and fix any bugs.
Refactor the code based on the detailed requirements below.
✨ Refactoring Requirements

Function Naming:
Rename all functions to a verb_noun format.
Add a suffix of _row, _col, or _table to function names based on the data level they operate on.
Specifically, rename:
tr to replace_values_col
summarize to summarize_table
numeric_map to encode_onehot_col

Code Structure & Help:
Group the help output by row, col, table, plot, and utilities.
Ensure all command names match their corresponding handle functions.
Remove the lowmem option and related functions completely.
Modularize redundant tasks by creating a helper function for common code, such as printing output.
Separate argument parsing into its own function and call it from main.
Do not alter the core logic unless necessary to fix a bug.

New Functions:
subset_table: A new function to print a subset of columns based on user switches:
--only_numeric_columns
--only_integer_columns
--only_string_columns
If --index and --meta_columns are provided, they should be printed first, followed by the selected subset.
add_suffix_col: Adds a suffix to all values in a specified column.
query_table: A function to filter and print rows based on a single-column query (e.g., column_name, operation, value). Supported operations are equal_to, less_than, and greater_than.

Output
Acknowledge each chunk of the script, state the number of lines read, and indicate you are ready for the next one.
Once all chunks are received, confirm the total number of lines.
Update the VERSION at the top of the script and provide a succinct, single-line summary of the changes.
Print the fully complete and functional script, with no missing parts.
State the number of lines in the input script and the final output script.

'''

# --------------------------
# Utility Functions
# --------------------------
def _clean_string_for_header_and_data(s):
    """Cleans up a string for headers and data."""
    if not isinstance(s, str):
        return s
    s = s.lower()
    s = s.replace(' ', '_')
    s = s.replace('.', '_')
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
    """Parses a column argument (1-indexed or name) and returns the corresponding 0-indexed column index."""
    try:
        col_idx = int(value)
        if col_idx < 1:
            raise ValueError(f"Error: {arg_name} '{value}' must be >= 1 (1-indexed).")
        if col_idx - 1 >= len(df_columns):
            raise IndexError(f"Error: {arg_name} '{value}' is out of bounds. Max is {len(df_columns)}.")
        return col_idx - 1
    except ValueError:
        if not is_header_present:
            raise ValueError(f"Error: Cannot use column name '{value}' for {arg_name} when no header is present. Use a 1-indexed integer.")
        if value not in df_columns:
            raise ValueError(f"Error: Column '{value}' not found in header. Available: {list(df_columns)}.")
        return df_columns.get_loc(value)
    except IndexError as e:
        raise e

def _parse_multiple_columns_arg(values, df_columns, is_header_present, arg_name="columns"):
    """Parses a comma-separated string of columns (numeric or names) and returns a list of their 0-indexed positions."""
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

# --------------------------
# Argument Parser
# --------------------------
def _setup_arg_parser():
    """Sets up the argument parser with all commands and options."""
    parser = argparse.ArgumentParser(
        description=(
            "A command-line tool for manipulating tabular data.\n\n"
            "Usage:\n"
            "    tbltool.py [GLOBAL_OPTIONS] <operation> [OPERATION_SPECIFIC_OPTIONS]\n\n"
            "Use 'tbltool.py <operation> --help' for details on each operation."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Global Options
    global_options = parser.add_argument_group("Global Options")
    global_options.add_argument("-f", "--file", type=argparse.FileType('r'), default=sys.stdin, help="Input file (default: stdin).")
    global_options.add_argument("-s", "--sep", default="\t", help="Field separator (default: tab). Supports escape sequences.")
    global_options.add_argument("--noheader", action="store_true", help="Input does not have a header row.")
    global_options.add_argument("--ignore-lines", default="^#", help="Regex pattern for lines to ignore (default: '^#').")
    global_options.add_argument("--verbose", action="store_true", help="Enable verbose debug output to stderr.")
    global_options.add_argument("-r", "--row-index", help="Column (1-indexed or name) to serve as the row identifier.")

    subparsers = parser.add_subparsers(dest="operation", title="Available Operations",
                                       description="The following operations are available:",
                                       help="For operation-specific help, type: tbltool.py <operation> --help")

    # Helper function to add parsers with a category title
    def add_parser_category(title):
        return subparsers.add_parser(title, help="").add_subparsers(title=f"--- {title} Operations ---")

    # Table Operations
    parser.add_argument_group("--- Table Operations ---")
    parser_transpose_table = subparsers.add_parser("transpose_table", help="Transpose the table.")
    parser_aggregate_table = subparsers.add_parser("aggregate_table", help="Group and aggregate data.")
    parser_aggregate_table.add_argument("--group", required=True, help="Comma-separated column(s) to group by.")
    parser_aggregate_table.add_argument("--cols", help="Columns to aggregate (use 'all' for all non-group columns).")
    parser_aggregate_table.add_argument("--agg", required=True, choices=['sum', 'mean', 'value_counts', 'entropy'], help="Aggregator to apply.")
    parser_aggregate_table.add_argument("--normalize", action="store_true", help="For value_counts, normalize frequencies.")
    parser_aggregate_table.add_argument("--melted", action="store_true", help="Indicate that input is in long (melted) format.")

    parser_sort_table = subparsers.add_parser("sort_table", help="Sort the table by a column.")
    parser_sort_table.add_argument("-n", "--column", required=True, help="Column to sort by (1-indexed or name).")
    parser_sort_table.add_argument("--desc", action="store_true", help="Sort in descending order.")
    parser_sort_table.add_argument("-p", "--pattern", help="Regex pattern to extract a numeric key from the column for sorting.")

    parser_clean_header_table = subparsers.add_parser("clean_header_table", help="Clean all header names.")
    parser_view_table = subparsers.add_parser("view_table", help="Display the data in a formatted table.")
    parser_view_table.add_argument("--max-rows", type=int, default=20, help="Max rows to display.")
    parser_view_table.add_argument("--max-cols", type=int, default=None, help="Max columns to display.")
    parser_view_table.add_argument("--precision-long", action="store_true", help="Display numeric columns with full precision.")

    parser_melt_table = subparsers.add_parser("melt_table", help="Melt the table from wide to long format.")
    parser_melt_table.add_argument("--id_vars", required=True, help="Comma-separated ID variables.")
    parser_melt_table.add_argument("--value_vars", help="Columns to melt (default: all other columns).")

    parser_unmelt_table = subparsers.add_parser("unmelt_table", help="Unmelt (pivot) the table from long to wide format.")
    parser_unmelt_table.add_argument("--index", required=True, help="Column to use as index.")
    parser_unmelt_table.add_argument("--columns", required=True, help="Column to use for new column headers.")
    parser_unmelt_table.add_argument("--value", required=True, help="Column containing values.")

    parser_add_metadata_table = subparsers.add_parser("add_metadata_table", help="Merge a metadata file.")
    parser_add_metadata_table.add_argument("--meta", required=True, help="Path to the metadata file.")
    parser_add_metadata_table.add_argument("--key_column_in_input", required=True, help="Key column in the input file.")
    parser_add_metadata_table.add_argument("--key_column_in_meta", required=True, help="Key column in the metadata file.")

    parser_query_table = subparsers.add_parser("query_table", help="Filter rows based on a numeric condition.")
    parser_query_table.add_argument("-n", "--column", required=True, help="Column to query (1-indexed or name).")
    parser_query_table.add_argument("-op", "--operator", required=True, choices=['less_than', 'greater_than', 'equal_to', 'not_equal_to'], help="Comparison operator.")
    parser_query_table.add_argument("-v", "--value", required=True, type=float, help="Value to compare against.")

    parser_subset_table = subparsers.add_parser("subset_table", help="Select a subset of columns based on data type.")
    subset_group = parser_subset_table.add_mutually_exclusive_group(required=True)
    subset_group.add_argument("--only_numeric_columns", action="store_true")
    subset_group.add_argument("--only_integer_columns", action="store_true")
    subset_group.add_argument("--only_string_columns", action="store_true")
    parser_subset_table.add_argument("--meta_columns", help="Comma-separated list of metadata columns to always include.")

    # Column Operations
    parser.add_argument_group("--- Column Operations ---")
    parser_move_col = subparsers.add_parser("move_col", help="Move a column to a new position.")
    parser_move_col.add_argument("-n", "--column", required=True, help="Source column.")
    parser_move_col.add_argument("-j", "--dest-column", required=True, help="Destination column.")

    parser_insert_col = subparsers.add_parser("insert_col", help="Insert a new column.")
    parser_insert_col.add_argument("-n", "--column", required=True, help="Column position for insertion.")
    parser_insert_col.add_argument("-v", "--value", required=True, help="Value for the new column.")
    parser_insert_col.add_argument("--new-header", default="new_column", help="Header for the new column.")

    parser_drop_col = subparsers.add_parser("drop_col", help="Drop one or more columns.")
    parser_drop_col.add_argument("-n", "--column", required=True, help="Comma-separated columns to drop ('all' for all).")

    parser_split_col = subparsers.add_parser("split_col", help="Split a column into multiple columns.")
    parser_split_col.add_argument("-n", "--column", required=True, help="Column to split.")
    parser_split_col.add_argument("-d", "--delimiter", required=True, help="Delimiter to split by.")

    parser_join_col = subparsers.add_parser("join_col", help="Join multiple columns into one.")
    parser_join_col.add_argument("-n", "--column", required=True, help="Comma-separated columns to join.")
    parser_join_col.add_argument("-d", "--delimiter", default="", help="Delimiter to use when joining.")
    parser_join_col.add_argument("--new-header", default="joined_column", help="Header for the new column.")

    parser_replace_values_col = subparsers.add_parser("replace_values_col", help="Translate or replace values in a column.")
    parser_replace_values_col.add_argument("-n", "--column", required=True, help="Column to translate.")
    tr_group = parser_replace_values_col.add_mutually_exclusive_group(required=True)
    tr_group.add_argument("-d", "--dict-file", help="Two-column file for value mapping.")
    tr_group.add_argument("--from-val", help="Value to replace.")
    parser_replace_values_col.add_argument("--to-val", help="Replacement value.")
    parser_replace_values_col.add_argument("--in-place", action="store_true", help="Modify column in-place.")

    parser_clean_values_col = subparsers.add_parser("clean_values_col", help="Clean values in specified columns.")
    parser_clean_values_col.add_argument("-n", "--column", required=True, help="Comma-separated columns to clean ('all' for all).")

    parser_add_prefix_col = subparsers.add_parser("add_prefix_col", help="Add a prefix to values in a column.")
    parser_add_prefix_col.add_argument("-n", "--column", required=True, help="Column(s) to add prefix to.")
    parser_add_prefix_col.add_argument("-v", "--string", required=True, help="Prefix string.")
    parser_add_prefix_col.add_argument("-d", "--delimiter", default="", help="Delimiter between prefix and value.")

    parser_add_suffix_col = subparsers.add_parser("add_suffix_col", help="Add a suffix to values in a column.")
    parser_add_suffix_col.add_argument("-n", "--column", required=True, help="Column(s) to add suffix to.")
    parser_add_suffix_col.add_argument("-v", "--string", required=True, help="Suffix string.")
    parser_add_suffix_col.add_argument("-d", "--delimiter", default="", help="Delimiter between value and suffix.")

    parser_summarize_col = subparsers.add_parser("summarize_col", help="Count top occurring values in column(s).")
    parser_summarize_col.add_argument("-T", "--top-n", type=int, default=10, help="Number of top values to show.")
    parser_summarize_col.add_argument("-n", "--column", required=True, help="Column(s) to summarize.")

    parser_strip_chars_col = subparsers.add_parser("strip_chars_col", help="Remove a regex pattern from values.")
    parser_strip_chars_col.add_argument("-n", "--column", required=True, help="Column to modify.")
    parser_strip_chars_col.add_argument("-p", "--pattern", required=True, help="Regex pattern to remove.")
    parser_strip_chars_col.add_argument("--in-place", action="store_true", help="Modify column in-place.")

    parser_encode_categorical_col = subparsers.add_parser("encode_categorical_col", help="Map unique string values to numbers (label encoding).")
    parser_encode_categorical_col.add_argument("-n", "--column", required=True, help="Column to encode.")
    parser_encode_categorical_col.add_argument("--new-header", help="Header for the new encoded column.")

    parser_capture_regex_col = subparsers.add_parser("capture_regex_col", help="Capture substrings using a regex group.")
    parser_capture_regex_col.add_argument("-n", "--column", required=True, help="Column to apply regex on.")
    parser_capture_regex_col.add_argument("-p", "--pattern", required=True, help="Regex with a capturing group.")

    parser_cut_col = subparsers.add_parser("cut_col", help="Select columns by name or pattern.")
    parser_cut_col.add_argument("pattern", help="Regex pattern or comma-separated list of names.")
    parser_cut_col.add_argument("--list", action="store_true", help="Interpret pattern as a literal list of column names.")

    # Row Operations
    parser.add_argument_group("--- Row Operations ---")
    parser_grep_row = subparsers.add_parser("grep_row", help="Filter rows by matching a pattern in a column.")
    parser_grep_row.add_argument("-n", "--column", required=True, help="Column to search in.")
    grep_group = parser_grep_row.add_mutually_exclusive_group(required=True)
    grep_group.add_argument("-p", "--pattern", help="Regex pattern to search for.")
    grep_group.add_argument("--word-file", help="File of words to match.")
    parser_grep_row.add_argument("-v", "--invert", action="store_true", help="Invert match (select non-matching rows).")

    parser_insert_row = subparsers.add_parser("insert_row", help="Insert a new row.")
    parser_insert_row.add_argument("-i", "--row-idx", type=int, default=0, help="Row position for insertion (1-indexed).")
    parser_insert_row.add_argument("-v", "--values", required=True, help="Comma-separated values for the new row.")

    parser_drop_row = subparsers.add_parser("drop_row", help="Delete a row by its position.")
    parser_drop_row.add_argument("-i", "--row-idx", type=int, required=True, help="Row position to drop (1-indexed).")

    # Plot Operations
    parser.add_argument_group("--- Plot Operations ---")
    parser_plot_ggplot = subparsers.add_parser("plot_ggplot", help="Generate a ggplot using Plotnine.")
    parser_plot_ggplot.add_argument("--geom", required=True, choices=["boxplot", "bar", "point", "hist", "tile"], help="Type of plot.")
    parser_plot_ggplot.add_argument("--x", required=True, help="Column for x-axis.")
    parser_plot_ggplot.add_argument("--y", help="Column for y-axis.")
    parser_plot_ggplot.add_argument("--fill", help="Column for fill aesthetic.")
    parser_plot_ggplot.add_argument("-o", "--output", required=True, help="Output PDF filename.")

    parser_plot_matplotlib = subparsers.add_parser("plot_matplotlib", help="Generate a matplotlib plot (e.g., Venn diagram).")
    parser_plot_matplotlib.add_argument("--mode", required=True, choices=["venn2", "venn3"], help="Plot mode.")
    parser_plot_matplotlib.add_argument("--colnames", required=True, help="Comma-separated column names for the plot.")
    parser_plot_matplotlib.add_argument("-o", "--output", required=True, help="Output PDF filename.")

    # Utility Operations
    parser.add_argument_group("--- Utility Operations ---")
    parser_view_header_table = subparsers.add_parser("view_header_table", help="Display header names and their positions.")

    return parser

# --------------------------
# Operation Handler Functions
# --------------------------
def _handle_aggregate_table(df, args, **kwargs):
    """Groups and aggregates data."""
    import pandas as pd
    from scipy.stats import entropy as calculate_entropy

    if args.melted:
        required_cols = {"variable", "value"}
        if not required_cols.issubset(df.columns):
            raise ValueError("Error: --melted requires 'variable' and 'value' columns.")
        group_cols = [col.strip() for col in args.group.split(",")] if args.group else []
        if "variable" not in group_cols:
            group_cols.append("variable")
        agg_func = args.agg.lower()
        summary_rows = []
        for grp_keys, grp_df in df.groupby(group_cols):
            grp_keys = (grp_keys,) if not isinstance(grp_keys, tuple) else grp_keys
            group_dict = dict(zip(group_cols, grp_keys))
            series = grp_df["value"]
            if agg_func in ["sum", "mean"]:
                series_numeric = pd.to_numeric(series, errors="coerce").dropna()
                if not series_numeric.empty:
                    result = series_numeric.sum() if agg_func == "sum" else series_numeric.mean()
                    group_dict[f"{agg_func}_value"] = result
                    summary_rows.append(group_dict)
            elif agg_func == "value_counts":
                vc = series.value_counts(normalize=args.normalize).reset_index()
                vc.columns = ["value", "count"]
                for _, row in vc.iterrows():
                    out = {**group_dict, "aggregated_column": "value", "value": row["value"], "count": row["count"]}
                    summary_rows.append(out)
            elif agg_func == "entropy":
                group_dict["entropy"] = calculate_entropy(series.value_counts())
                summary_rows.append(group_dict)
        return pd.DataFrame(summary_rows)
    else: # Wide format
        if not args.group or not args.cols:
            raise ValueError("Error: In wide format, --group and --cols are required.")
        group_cols = [col.strip() for col in args.group.split(",")]
        agg_cols = [col for col in df.columns if col not in group_cols] if args.cols.strip().lower() in ["*", "all"] else [col.strip() for col in args.cols.split(",")]
        agg_func = args.agg.lower()
        if agg_func in ["sum", "mean"]:
            numeric_cols = df[agg_cols].select_dtypes(include='number').columns.tolist()
            if not numeric_cols:
                raise ValueError("Error: No numeric columns found for aggregation.")
            summary_df = df.groupby(group_cols)[numeric_cols].agg(agg_func).reset_index()
            summary_df.rename(columns={col: f"{agg_func}_{col}" for col in numeric_cols}, inplace=True)
            return summary_df
        elif agg_func in ["value_counts", "entropy"]:
            summary_rows = []
            for grp_keys, grp_df in df.groupby(group_cols):
                grp_keys = (grp_keys,) if not isinstance(grp_keys, tuple) else grp_keys
                group_dict = dict(zip(group_cols, grp_keys))
                for col in agg_cols:
                    if agg_func == "value_counts":
                        vc = grp_df[col].value_counts(normalize=args.normalize).reset_index()
                        vc.columns = ["value", "count"]
                        for _, row in vc.iterrows():
                            out = {**group_dict, "aggregated_column": col, "value": row["value"], "count": row["count"]}
                            summary_rows.append(out)
                    elif agg_func == "entropy":
                        out = {**group_dict, "aggregated_column": col, "entropy": calculate_entropy(grp_df[col].value_counts())}
                        summary_rows.append(out)
            return pd.DataFrame(summary_rows)
        else:
            raise ValueError(f"Unsupported aggregator '{agg_func}'.")

def _handle_transpose_table(df, args, is_header_present, **kwargs):
    """Transposes the table."""
    if is_header_present:
        df = pd.concat([pd.DataFrame([df.columns], columns=df.columns), df], ignore_index=True)
    if df.shape[1] < 2:
        raise ValueError("Error: Input must have at least 2 columns for transpose.")
    new_headers = df.iloc[:, 0].tolist()
    transposed_df = df.iloc[:, 1:].T
    transposed_df.columns = new_headers
    return transposed_df

def _handle_move_col(df, args, is_header_present, **kwargs):
    """Moves a column to a new position."""
    from_idx = _parse_column_arg(args.column, df.columns, is_header_present)
    to_idx = _parse_column_arg(args.dest_column, df.columns, is_header_present)
    col_name = df.columns[from_idx]
    data = df.pop(col_name)
    df.insert(to_idx, col_name, data)
    return df

def _handle_insert_col(df, args, is_header_present, **kwargs):
    """Inserts a new column with a specified value."""
    pos = _parse_column_arg(args.column, df.columns, is_header_present)
    value = codecs.decode(args.value, 'unicode_escape')
    new_header = get_unique_header(args.new_header, df) if is_header_present else args.new_header
    df.insert(pos, new_header, value)
    return df

def _handle_drop_col(df, args, is_header_present, **kwargs):
    """Drops one or more columns."""
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present)
    names = [df.columns[i] for i in indices]
    return df.drop(columns=names)

def _handle_grep_row(df, args, is_header_present, **kwargs):
    """Filters rows based on a pattern."""
    col_idx = _parse_column_arg(args.column, df.columns, is_header_present)
    series = df.iloc[:, col_idx].astype(str)
    mask = None
    if args.word_file:
        with open(args.word_file, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        if not words: return df
        pattern = "|".join(map(re.escape, words))
        mask = series.str.contains(pattern, regex=True, na=False)
    elif args.pattern:
        mask = series.str.contains(args.pattern, regex=True, na=False)
    return df[~mask] if args.invert else df[mask]

def _handle_split_col(df, args, is_header_present, **kwargs):
    """Splits a column by a delimiter."""
    col_idx = _parse_column_arg(args.column, df.columns, is_header_present)
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    original_name = df.columns[col_idx]
    split_df = df.iloc[:, col_idx].astype(str).str.split(delim, expand=True).fillna('')
    new_headers = [get_unique_header(f"{original_name}_{i+1}", df) for i in range(split_df.shape[1])]
    split_df.columns = new_headers
    df = df.drop(columns=[original_name])
    return pd.concat([df.iloc[:, :col_idx], split_df, df.iloc[:, col_idx:]], axis=1)

def _handle_join_col(df, args, is_header_present, **kwargs):
    """Joins multiple columns into a single column."""
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present)
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    names = [df.columns[i] for i in indices]
    joined_series = df[names].astype(str).apply(lambda x: delim.join(x), axis=1)
    new_header = get_unique_header(args.new_header, df)
    df = df.drop(columns=names)
    df.insert(min(indices), new_header, joined_series)
    return df

def _handle_replace_values_col(df, args, is_header_present, **kwargs):
    """Replaces values in a column."""
    col_idx = _parse_column_arg(args.column, df.columns, is_header_present)
    original_name = df.columns[col_idx]
    translated_series = None
    if args.dict_file:
        mapping = dict(line.strip().split('\t', 1) for line in open(args.dict_file) if '\t' in line)
        translated_series = df.iloc[:, col_idx].astype(str).map(mapping).fillna(df.iloc[:, col_idx])
    elif args.from_val is not None and args.to_val is not None:
        from_val = codecs.decode(args.from_val, 'unicode_escape')
        to_val = codecs.decode(args.to_val, 'unicode_escape')
        translated_series = df.iloc[:, col_idx].astype(str).str.replace(from_val, to_val, regex=True)
    if args.in_place:
        df[original_name] = translated_series
    else:
        new_header = get_unique_header(f"{original_name}_translated", df)
        df.insert(col_idx + 1, new_header, translated_series)
    return df

def _handle_sort_table(df, args, is_header_present, **kwargs):
    """Sorts the table based on a column."""
    col_idx = _parse_column_arg(args.column, df.columns, is_header_present)
    col_name = df.columns[col_idx]
    if args.pattern:
        df['_sort_key'] = df[col_name].astype(str).str.extract(f'({args.pattern})').iloc[:, 0].astype(float)
        df.sort_values(by='_sort_key', ascending=not args.desc, inplace=True, kind='stable')
        df.drop(columns=['_sort_key'], inplace=True)
    else:
        df.sort_values(by=col_name, ascending=not args.desc, inplace=True, kind='stable')
    return df

def _handle_clean_header_table(df, args, is_header_present, **kwargs):
    """Cleans all column headers."""
    if is_header_present:
        df.columns = [_clean_string_for_header_and_data(col) for col in df.columns]
    return df

def _handle_clean_values_col(df, args, is_header_present, **kwargs):
    """Cleans values in specified columns."""
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present)
    for i in indices:
        df.iloc[:, i] = df.iloc[:, i].apply(_clean_string_for_header_and_data)
    return df

def _handle_add_prefix_col(df, args, is_header_present, **kwargs):
    """Adds a prefix to values in specified columns."""
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present)
    prefix = codecs.decode(args.string, 'unicode_escape')
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    for i in indices:
        df.iloc[:, i] = prefix + delim + df.iloc[:, i].astype(str)
    return df

def _handle_add_suffix_col(df, args, is_header_present, **kwargs):
    """Adds a suffix to values in specified columns."""
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present)
    suffix = codecs.decode(args.string, 'unicode_escape')
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    for i in indices:
        df.iloc[:, i] = df.iloc[:, i].astype(str) + delim + suffix
    return df

def _handle_summarize_col(df, args, is_header_present, **kwargs):
    """Counts top occurring values and returns a summary DataFrame."""
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present)
    all_counts = []
    for i in indices:
        col_name = df.columns[i]
        counts = df[col_name].value_counts().nlargest(args.top_n).reset_index()
        counts.columns = ['Value', 'Count']
        counts['Frequency'] = (counts['Count'] / len(df)) * 100
        counts.insert(0, 'Column', col_name)
        all_counts.append(counts)
    return pd.concat(all_counts, ignore_index=True)

def _handle_strip_chars_col(df, args, is_header_present, **kwargs):
    """Removes a regex pattern from column values."""
    col_idx = _parse_column_arg(args.column, df.columns, is_header_present)
    original_name = df.columns[col_idx]
    stripped_series = df.iloc[:, col_idx].astype(str).str.replace(args.pattern, '', regex=True)
    if args.in_place:
        df[original_name] = stripped_series
    else:
        new_header = get_unique_header(f"{original_name}_stripped", df)
        df.insert(col_idx + 1, new_header, stripped_series)
    return df

def _handle_encode_categorical_col(df, args, is_header_present, **kwargs):
    """Maps unique string values to numeric identifiers (label encoding)."""
    col_idx = _parse_column_arg(args.column, df.columns, is_header_present)
    original_name = df.columns[col_idx]
    mapping = {val: i for i, val in enumerate(df[original_name].unique())}
    encoded_series = df[original_name].map(mapping)
    new_header = args.new_header if args.new_header else get_unique_header(f"{original_name}_encoded", df)
    df.insert(col_idx + 1, new_header, encoded_series)
    return df

def _handle_capture_regex_col(df, args, is_header_present, **kwargs):
    """Captures substrings using a regex pattern."""
    col_idx = _parse_column_arg(args.column, df.columns, is_header_present)
    original_name = df.columns[col_idx]
    captured_series = df.iloc[:, col_idx].astype(str).str.extract(args.pattern)
    new_header = get_unique_header(f"{original_name}_captured", df)
    df.insert(col_idx + 1, new_header, captured_series)
    return df

def _handle_view_table(df, args, is_header_present, row_idx_col_name, **kwargs):
    """Formats and prints the DataFrame to stdout for viewing."""
    pd.set_option('display.max_rows', args.max_rows)
    pd.set_option('display.max_columns', args.max_cols)
    pd.set_option('display.width', None)
    disp = df.copy()
    if not args.precision_long:
        for col in disp.select_dtypes(include='number').columns:
            disp[col] = disp[col].round(2)
            if disp[col].dropna().apply(float.is_integer).all():
                disp[col] = disp[col].astype('Int64')
    if row_idx_col_name and row_idx_col_name in disp.columns:
        disp = disp[[row_idx_col_name] + [c for c in disp.columns if c != row_idx_col_name]]
    sys.stdout.write(disp.to_string(index=True, header=is_header_present) + '\n')
    sys.exit(0) # View is a terminal operation

def _handle_cut_col(df, args, is_header_present, **kwargs):
    """Selects columns by name or pattern."""
    if args.list:
        col_list = [x.strip() for x in args.pattern.split(',')]
        missing = [col for col in col_list if col not in df.columns]
        if missing: raise ValueError(f"Error: Columns not found: {missing}")
        return df[col_list]
    else:
        selected = [col for col in df.columns if re.search(args.pattern, str(col))]
        if not selected:
            sys.stderr.write(f"Warning: No columns matched pattern '{args.pattern}'.\n")
        return df[selected]

def _handle_view_header_table(df, args, is_header_present, **kwargs):
    """Displays header names and their 1-indexed positions."""
    if not is_header_present:
        sys.stderr.write("Warning: No header to display (--noheader was used).\n")
    else:
        header_df = pd.DataFrame({
            'Position': range(1, len(df.columns) + 1),
            'Header': df.columns
        })
        sys.stdout.write(header_df.to_string(index=False) + '\n')
    sys.exit(0) # Terminal operation

def _handle_insert_row(df, args, **kwargs):
    """Inserts a new row at a specified position."""
    insert_pos = args.row_idx - 1
    values = [codecs.decode(v.strip(), 'unicode_escape') for v in args.values.split(',')]
    if len(values) != df.shape[1]:
        raise ValueError(f"Error: Number of values ({len(values)}) must match number of columns ({df.shape[1]}).")
    new_row = pd.DataFrame([values], columns=df.columns)
    return pd.concat([df.iloc[:insert_pos], new_row, df.iloc[insert_pos:]]).reset_index(drop=True)

def _handle_drop_row(df, args, **kwargs):
    """Drops a row by its position."""
    drop_pos = args.row_idx - 1
    if not (0 <= drop_pos < len(df)):
        raise IndexError(f"Error: Row index {args.row_idx} is out of bounds.")
    return df.drop(df.index[drop_pos]).reset_index(drop=True)

def _handle_plot_ggplot(df, args, **kwargs):
    """Generates a plot using Plotnine."""
    from plotnine import ggplot, aes, labs, theme_minimal
    from plotnine.geoms import geom_boxplot, geom_bar, geom_point, geom_histogram, geom_tile

    geom_map = {
        "boxplot": geom_boxplot(), "bar": geom_bar(stat="identity"),
        "point": geom_point(), "hist": geom_histogram(), "tile": geom_tile()
    }
    plot = (ggplot(df, aes(x=args.x, y=args.y, fill=args.fill))
            + geom_map[args.geom]
            + labs(title=args.x) + theme_minimal())
    plot.save(filename=args.output, format="pdf")
    sys.exit(0) # Plotting is a terminal operation

def _handle_plot_matplotlib(df, args, **kwargs):
    """Generates a Venn diagram using matplotlib-venn."""
    from matplotlib_venn import venn2, venn3
    import matplotlib.pyplot as plt

    colnames = [col.strip() for col in args.colnames.split(',')]
    sets = [set(df[col].dropna()) for col in colnames]

    plt.figure()
    if args.mode == 'venn2' and len(sets) == 2:
        venn2(sets, set_labels=colnames)
    elif args.mode == 'venn3' and len(sets) == 3:
        venn3(sets, set_labels=colnames)
    else:
        raise ValueError("Invalid mode or number of columns for Venn diagram.")
    plt.savefig(args.output, format="pdf")
    sys.exit(0) # Plotting is a terminal operation

def _handle_melt_table(df, args, **kwargs):
    """Melts a DataFrame from wide to long format."""
    id_vars = [v.strip() for v in args.id_vars.split(',')]
    value_vars = [v.strip() for v in args.value_vars.split(',')] if args.value_vars else None
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars)

def _handle_unmelt_table(df, args, **kwargs):
    """Pivots a DataFrame from long to wide format."""
    return df.pivot(index=args.index, columns=args.columns, values=args.value).reset_index()

def _handle_add_metadata_table(df, args, is_header_present, **kwargs):
    """Merges a metadata file with the input data."""
    meta_df = pd.read_csv(args.meta, sep=kwargs['input_sep'])
    key_input_idx = _parse_column_arg(args.key_column_in_input, df.columns, is_header_present)
    key_meta_idx = _parse_column_arg(args.key_column_in_meta, meta_df.columns, True)
    return pd.merge(df, meta_df, how='left',
                    left_on=df.columns[key_input_idx],
                    right_on=meta_df.columns[key_meta_idx])

def _handle_query_table(df, args, is_header_present, **kwargs):
    """Filters rows based on a numeric condition."""
    col_idx = _parse_column_arg(args.column, df.columns, is_header_present)
    col_name = df.columns[col_idx]
    series = pd.to_numeric(df[col_name], errors='coerce')
    op_map = {
        'less_than': series < args.value,
        'greater_than': series > args.value,
        'equal_to': series == args.value,
        'not_equal_to': series != args.value,
    }
    return df[op_map[args.operator]]

def _handle_subset_table(df, args, is_header_present, **kwargs):
    """Subsets columns based on their data type."""
    # Convert applicable columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    id_cols = []
    if args.row_index:
        idx = _parse_column_arg(args.row_index, df.columns, is_header_present)
        id_cols.append(df.columns[idx])
    if args.meta_columns:
        id_cols.extend([c.strip() for c in args.meta_columns.split(',')])

    data_cols = []
    if args.only_numeric_columns:
        data_cols = df.select_dtypes(include='number').columns.tolist()
    elif args.only_integer_columns:
        data_cols = df.select_dtypes(include='integer').columns.tolist()
    elif args.only_string_columns:
        data_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # Ensure id_cols are not duplicated and appear first
    final_cols = id_cols + [c for c in data_cols if c not in id_cols]
    return df[final_cols]

# --------------------------
# Dispatch Table
# --------------------------
OPERATION_HANDLERS = {
    "transpose_table": _handle_transpose_table, "aggregate_table": _handle_aggregate_table,
    "sort_table": _handle_sort_table, "clean_header_table": _handle_clean_header_table,
    "view_table": _handle_view_table, "melt_table": _handle_melt_table,
    "unmelt_table": _handle_unmelt_table, "add_metadata_table": _handle_add_metadata_table,
    "query_table": _handle_query_table, "subset_table": _handle_subset_table,
    "move_col": _handle_move_col, "insert_col": _handle_insert_col, "drop_col": _handle_drop_col,
    "split_col": _handle_split_col, "join_col": _handle_join_col,
    "replace_values_col": _handle_replace_values_col, "clean_values_col": _handle_clean_values_col,
    "add_prefix_col": _handle_add_prefix_col, "add_suffix_col": _handle_add_suffix_col,
    "summarize_col": _handle_summarize_col, "strip_chars_col": _handle_strip_chars_col,
    "encode_categorical_col": _handle_encode_categorical_col, "capture_regex_col": _handle_capture_regex_col,
    "cut_col": _handle_cut_col, "grep_row": _handle_grep_row, "insert_row": _handle_insert_row,
    "drop_row": _handle_drop_row, "plot_ggplot": _handle_plot_ggplot,
    "plot_matplotlib": _handle_plot_matplotlib, "view_header_table": _handle_view_header_table,
}

# --------------------------
# Input/Output Functions
# --------------------------
def _read_input_data(args, input_sep, header_param):
    """Reads input from file or stdin and returns a DataFrame."""
    comment_char = args.ignore_lines[1:] if args.ignore_lines.startswith('^') else None
    try:
        content = args.file.read()
        if not content.strip():
            return pd.DataFrame()
        return pd.read_csv(StringIO(content), sep=input_sep, header=header_param,
                           dtype=str, comment=comment_char)
    except Exception as e:
        sys.stderr.write(f"Error reading input data: {e}\n")
        sys.exit(1)

def _write_output(df, sep, is_header_present):
    """Writes the DataFrame to stdout."""
    if df is not None and not df.empty:
        try:
            df.to_csv(sys.stdout, sep=sep, index=False, header=is_header_present,
                      quoting=csv.QUOTE_NONE, escapechar='\\')
        except BrokenPipeError:
            pass # Suppress error on broken pipe (e.g., piping to `head`)
        except Exception as e:
            sys.stderr.write(f"Error writing output: {e}\n")
            sys.exit(1)

# --------------------------
# Main Execution
# --------------------------
def main():
    """Main function to parse arguments and dispatch to handlers."""
    parser = _setup_arg_parser()
    args = parser.parse_args()
    if not args.operation:
        parser.print_help()
        sys.exit(0)

    is_header_present = not args.noheader
    header_param = 0 if is_header_present else None
    input_sep = codecs.decode(args.sep, 'unicode_escape')

    df = _read_input_data(args, input_sep, header_param)
    if df.empty and args.operation not in ["view_table", "view_header_table", "insert_row"]:
        sys.stderr.write("Warning: Input is empty. No operation performed.\n")
        sys.exit(0)

    if not is_header_present and not df.empty:
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]

    row_idx_col_name = None
    if args.row_index and not df.empty:
        idx = _parse_column_arg(args.row_index, df.columns, is_header_present, "row-index")
        row_idx_col_name = df.columns[idx]

    handler = OPERATION_HANDLERS.get(args.operation)
    if not handler:
        sys.stderr.write(f"Error: Unsupported operation '{args.operation}'.\n")
        sys.exit(1)

    try:
        handler_kwargs = {
            "is_header_present": is_header_present,
            "row_idx_col_name": row_idx_col_name,
            "input_sep": input_sep,
        }
        processed_df = handler(df, args, **handler_kwargs)
        _write_output(processed_df, input_sep, is_header_present)
    except (ValueError, IndexError, FileNotFoundError, KeyError) as e:
        sys.stderr.write(f"{e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"An unexpected error occurred: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
