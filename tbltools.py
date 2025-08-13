#!/usr/bin/env python3
import sys
import numpy as np
import argparse
import re
import os
import pandas as pd
import codecs  # For handling escape sequences
from io import StringIO  # For piped input handling with pandas
from collections import Counter
import csv  # For CSV formatting
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import math
# LAST UPDATED ON:  11 Aug 2025

# Default chunk size when processing input in low-memory mode.
CHUNK_SIZE = 10000

# Operations that support --lowmem (chunked) processing
LOWMEM_OPS = [
    "stats",
    "factorize",
    "filter",
    "replace",
    "strip",
    "prefix",
    "cleanup_values",
    "extract",
]

# --------------------------
# Utility Functions
# --------------------------
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

def _corr_from_df(M: pd.DataFrame, method="spearman") -> pd.DataFrame:
    """
    Pairwise correlation with robust NA handling and clipping to [-1, 1].
    Ensures diagonal==1 and no NaN/Inf anywhere.
    """
    corr = M.corr(method=method, min_periods=1).astype(float)
    corr = (corr + corr.T) / 2.0                # symmetrize
    np.fill_diagonal(corr.values, 1.0)
    corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    corr = corr.fillna(0.0).clip(-1.0, 1.0)
    return corr

def _link_from_corr(corr: pd.DataFrame, method="average"):
    """
    Convert a correlation matrix to a SciPy linkage via 1 - corr.
    Safe against NaN/Inf and non-symmetric inputs.
    """
    c = (corr + corr.T) / 2.0
    np.fill_diagonal(c.values, 1.0)
    c = c.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    condensed = squareform(1.0 - c.values, checks=False)
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=1.0)
    return linkage(condensed, method=method)

def _helper_extract_numeric_columns(df, exclude_cols=None):
    """
    Return (numeric_df, numeric_columns) where columns are coerced to numeric
    if *all* non-null values are numeric. Columns in exclude_cols are ignored.
    """
    if exclude_cols is None:
        exclude_cols = []
    numeric_columns = []
    numeric_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in exclude_cols:
            continue
        converted = pd.to_numeric(df[col], errors='coerce')
        non_null = df[col].dropna()
        if not non_null.empty and converted[non_null.index].isna().any():
            continue
        numeric_columns.append(col)
        numeric_df[col] = converted
    return numeric_df, numeric_columns

# Back-compat alias so any older calls still work
extract_numeric_columns = _helper_extract_numeric_columns

def _helper_extract_numeric_columns(df, exclude_cols=None):
    """
    Extract numeric columns from the DataFrame by attempting to convert each column
    using pd.to_numeric. Columns whose non-null values fail conversion are omitted.

    Non-numeric columns that are not explicitly excluded via --index/--row_annotations
    are simply ignored.

    Parameters:
      df (pd.DataFrame): The input DataFrame.
      exclude_cols (list, optional): List of column names to exclude from conversion.

    Returns:
      (numeric_df, numeric_columns): 
         numeric_df: A DataFrame containing only numeric columns (with converted values).
         numeric_columns: A list of column names that were successfully converted.
    """
    if exclude_cols is None:
        exclude_cols = []
    numeric_columns = []
    numeric_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in exclude_cols:
            continue
        converted = pd.to_numeric(df[col], errors='coerce')
        non_null = df[col].dropna()
        # Only treat column as numeric if every non-null value converts successfully.
        if not non_null.empty and converted[non_null.index].isna().any():
            continue
        numeric_columns.append(col)
        numeric_df[col] = converted
    return numeric_df, numeric_columns


def _clean_string_for_header_and_data(s):
    """Cleans up a string for headers and data."""
    if not isinstance(s, str):
        return s
    s = s.lower()
    s = s.replace(' ', '_')
    s = s.replace('.', '_')
    s = re.sub(r'[^\w_]', '', s)
    s = re.sub(r'_{2,}', '_', s)  # tr -squeeze
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

def remove_ansi(text):
    """Remove ANSI escape sequences from the given text."""
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

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
            raise ValueError(f"Error: {arg_name} '{value}' must be >= 1 (1-indexed).")
        if col_idx - 1 >= len(df_columns):
            raise IndexError(f"Error: {arg_name} '{value}' is out of bounds. Max is {len(df_columns)}.")
        return col_idx - 1
    except ValueError:
        if not is_header_present:
            raise ValueError(f"Error: Cannot use column name '{value}' for {arg_name} when no header is present (--noheader option was provided). Use a 1-indexed integer.")
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

# --------------------------
# Global Numeric Formatter
# --------------------------
def _format_numeric_columns(df):
    """
    For each column in a DataFrame, if every non-null value in that column is convertible
    to a numeric type then:
       - If all such values are effectively integers (within a small threshold), cast
         the column to a pandas nullable integer type.
       - Otherwise, round numeric values to two decimal places.
    If any non-null value is not convertible to a number, the column is left unchanged.
    """
    threshold = 1e-8
    for col in df.columns:
        series = df[col]
        numeric_series = pd.to_numeric(series, errors='coerce')
        if series.dropna().shape[0] != numeric_series.dropna().shape[0]:
            continue
        if numeric_series.dropna().apply(lambda x: abs(x - round(x)) < threshold).all():
            df[col] = numeric_series.astype("Int64")
        else:
            df[col] = numeric_series.round(2)
    return df

# --------------------------
# Argument Parser
# --------------------------
def _setup_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "A samtools-like command-line tool for manipulating table fields.\n\n"
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
        "--ignore_lines", default="^#",
        help="Regex for lines to ignore (default: '^#' means lines starting with '#')."
    )
    global_options.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose debug output to stderr."
    )
    global_options.add_argument(
        "-r", "--row_index",
        help="Specify a column (1-indexed or name) to serve as the row identifier."
    )
    global_options.add_argument(
        "--lowmem", action="store_true",
        help="Process data in chunks to reduce memory usage. Not all operations support low-memory mode."
    )
    global_options.add_argument(
        "--chunksize", type=int, default=CHUNK_SIZE,
        help=f"Chunk size for --lowmem (default: {CHUNK_SIZE})."
    )
    
    # Subparsers for operations
    subparsers = parser.add_subparsers(dest="operation", 
                                       help="Available operations. Use 'tbl.py <operation> --help' for details.")
    
    # TRANSPOSE
    subparsers.add_parser(
        "transpose",
        help="Transpose the input table. The first column's values are used as the header in the new output table."
    )
    
    # MOVE
    parser_move = subparsers.add_parser("move", help="Move a column. Required: --column and --dest_column.")
    parser_move.add_argument("-n", "--column", required=True, 
                             help="Source column (1-indexed or name).")
    parser_move.add_argument("-j", "--dest_column", required=True, 
                             help="Destination column (1-indexed or name).")
    
    # COL_INSERT (add_col)
    parser_col_insert = subparsers.add_parser("add_col",
                                              help="Insert a new column. Required: --column and --value.")
    parser_col_insert.add_argument("-n", "--column", required=True, 
                                   help="Column position (1-indexed or name) for insertion.")
    parser_col_insert.add_argument("-v", "--value", required=True, 
                                   help="Value to populate the new column.")
    parser_col_insert.add_argument("--new_header", default="new_column", 
                                   help="Header name for the new column (default: 'new_column').")
    
    # COL_DROP (drop_col)
    parser_col_drop = subparsers.add_parser("drop_col",
                                            help="Drop columns. Required: --column.")
    parser_col_drop.add_argument("-n", "--column", required=True, 
                                 help="Comma-separated list of column(s) (1-indexed or names) to drop. Use 'all' to drop all columns.")
    
    # GREP rows (filter)
    parser_grep = subparsers.add_parser("filter", 
                            help="Filter rows. Required: --column and one of --pattern, --starts_with, --ends_with, or --word_file.")
    grep_group = parser_grep.add_mutually_exclusive_group(required=True)
    parser_grep.add_argument("-n", "--column", required=True, 
                             help="Column to apply the filter (1-indexed or name).")
    grep_group.add_argument("-p", "--pattern", help="Regex pattern to search for.")
    grep_group.add_argument("--starts_with", help="String that the column value should start with.")
    grep_group.add_argument("--ends_with", help="String that the column value should end with.")
    grep_group.add_argument("--word_file", help="File containing words (one per line) to match against the column values.")
    parser_grep.add_argument("--substring_match", action="store_true",
                               help="Allow substring matching when using a word file.")
    parser_grep.add_argument("--tokenize", action="store_true",
                               help="Split the target field on '.' and use the first token for matching.")
    parser_grep.add_argument("--list_missing_words", action="store_true",
                               help="Report words from the word file not found in the input.")
    parser_grep.add_argument("-v", "--invert", action="store_true",
                               help="Invert match: select rows that do NOT match the specified criteria.")

    # AGGR (aggregate)
    parser_aggr = subparsers.add_parser("aggregate",
        help="Group and aggregate data via common functions: sum, mean, list, value_counts, entropy.")
    parser_aggr.add_argument("--group", required=True,
                             help="Comma-separated list of column(s) to group by.")
    parser_aggr.add_argument("--cols", "--columns", dest="cols", required=True,
                             help="Comma-separated list of column(s) to aggregate (or '*' for all non-group columns).")
    parser_aggr.add_argument("--agg", required=True,
                             help="Aggregator to apply. Supported: 'sum', 'mean', 'list', 'value_counts', 'entropy'.")
    parser_aggr.add_argument("--normalize", action="store_true",
                             help="For value_counts, normalize the frequencies within each group.")
    parser_aggr.add_argument("--melted", action="store_true",
                             help="Indicate that the input is in melted (long) format.")

    # SPLIT (split_col)
    parser_split = subparsers.add_parser("split_col", help="Split a column. Required: --column and --delimiter.")
    parser_split.add_argument("-n", "--column", required=True,
                              help="Column to split (1-indexed or name).")
    parser_split.add_argument("-d", "--delimiter", required=True,
                              help="Delimiter to split the column by. Supports escape sequences.")
    parser_split.add_argument("--new_header_prefix", default="split_col",
                              help="Prefix for the new columns (default: 'split_col').")
    
    # JOIN (join_col)
    parser_join = subparsers.add_parser("join_col", 
        help="Join columns. Required: --column. Optionally, --target_column specifies the destination for the joined column.")
    parser_join.add_argument("-n", "--column", required=True,
                             help="Comma-separated list of columns (1-indexed or names) to join.")
    parser_join.add_argument("-d", "--delimiter", default="",
                             help="Delimiter to insert between joined values (default: no delimiter). Supports escape sequences.")
    parser_join.add_argument("--new_header", default="joined_column",
                             help="Header for the resulting joined column (default: 'joined_column').")
    parser_join.add_argument("-j", "--target_column",
                             help="Target column (1-indexed or name) where the joined column will be placed.")
    
    # TR (replace)
    parser_tr = subparsers.add_parser("replace", 
        help="Translate values. Required: --column and either --dict_file or --from_val with --to_val.")
    parser_tr.add_argument("-n", "--column", required=True,
                             help="Column to translate (1-indexed or name).")
    tr_group = parser_tr.add_mutually_exclusive_group(required=True)
    tr_group.add_argument("-d", "--dict_file", help="Path to a two-column file (key<sep>value) for mapping. Uses the main --sep as separator.")
    tr_group.add_argument("--from_val", help="Value to translate from (for single translation). Supports escape sequences.")
    parser_tr.add_argument("--to_val", help="Value to translate to (for single translation). Supports escape sequences.")
    parser_tr.add_argument("--regex", action="store_true",
                             help="Treat --from_val as a regex pattern (default is literal).")
    parser_tr.add_argument("--new_header", default="_translated",
                             help="Suffix or new header for the translated column (default: '_translated').")
    parser_tr.add_argument("--in_place", action="store_true",
                             help="Replace the original column with the translated values.")
    
    # SORT
    parser_sort = subparsers.add_parser("sort", 
        help="Sort table. Required: --column. (Not compatible with lowmem mode.)")
    parser_sort.add_argument("-n", "--column", required=True,
                             help="Column to sort by (1-indexed or name). Only one column should be specified.")
    parser_sort.add_argument("--desc", action="store_true",
                             help="Sort in descending order (default is ascending).")
    parser_sort.add_argument("-p", "--pattern", help="Regex pattern to extract a numeric key from the column.")
    parser_sort.add_argument("--suffix_map", help="Comma-separated key:value pairs for suffix mapping (used with --pattern).")
    parser_sort.add_argument("--expand_scientific", action="store_true", 
                             help="Use a default mapping for scientific suffixes and override any supplied --suffix_map.")
    parser_sort.add_argument("-ps", "--pattern_string", help="Optional additional regex pattern for a secondary string sort key.")
    
    # CLEANUP HEADER
    parser_cleanup_header = subparsers.add_parser("cleanup_header",
                                                  help="Clean header names (lowercase, remove special characters, replace spaces with underscores).")
    
    # CLEANUP VALUES
    parser_cleanup_values = subparsers.add_parser("cleanup_values",
                                                  help="Clean values in specified columns. Required: --column.")
    parser_cleanup_values.add_argument("-n", "--column", required=True,
                                        help="Comma-separated list of columns (1-indexed or names) to clean. Use 'all' to clean every column.")
    
    # PREFIX ADD (prefix)
    parser_prefix_add = subparsers.add_parser("prefix",
                                              help="Add a prefix to column values. Required: --column and --string.")
    parser_prefix_add.add_argument("-n", "--column", required=True,
                                   help="Comma-separated list of columns (1-indexed or names) to prepend with a prefix. Use 'all' for every column.")
    parser_prefix_add.add_argument("-v", "--string", required=True,
                                   help="The prefix string to add. Supports escape sequences.")
    parser_prefix_add.add_argument("-d", "--delimiter", default="",
                                   help="Delimiter to insert between the prefix and the original value (default: none). Supports escape sequences.")
    
    # VALUE COUNTS (stats)
    parser_value_counts = subparsers.add_parser("stats",
                                                help="Count top occurring values. Required: --column.")
    parser_value_counts.add_argument("-T", "--top_n", type=int, default=5,
                                     help="Number of top values to display (default: 5).")
    parser_value_counts.add_argument("-n", "--column", required=True,
                                     help="Comma-separated list of columns (1-indexed or names) to count. Use 'all' for every column.")
    
    # STRIP
    parser_strip = subparsers.add_parser("strip",
                                          help="Remove a regex pattern from column values. Required: --column and --pattern.")
    parser_strip.add_argument("-n", "--column", required=True,
                              help="Column to strip (1-indexed or name).")
    parser_strip.add_argument("-p", "--pattern", required=True,
                              help="Regex pattern to remove from the column values.")
    parser_strip.add_argument("--new_header", default="_stripped",
                              help="Suffix or new header for the column after stripping (default: '_stripped').")
    parser_strip.add_argument("--in_place", action="store_true",
                              help="Modify the column in place instead of creating a new column.")
    
    # NUMERIC MAP (factorize)
    parser_numeric_map = subparsers.add_parser("factorize",
                                               help="Map unique string values to numbers. Required: --column.")
    parser_numeric_map.add_argument("-n", "--column", required=True,
                                    help="Column (1-indexed or name) whose unique values are to be mapped to numbers.")
    parser_numeric_map.add_argument("--new_header", help="Header for the new numeric mapping column (default: 'numeric_map_of_ORIGINAL_COLUMN_NAME').")
    
    # REGEX CAPTURE (extract)
    parser_regex_capture = subparsers.add_parser("extract",
                                                 help="Capture substrings using a regex capturing group. Required: --column and --pattern.")
    parser_regex_capture.add_argument("-n", "--column", required=True,
                                      help="Column on which to apply the regex (1-indexed or name).")
    parser_regex_capture.add_argument("-p", "--pattern", required=True,
                                      help="Regex pattern with at least one capturing group (e.g., '_(S[0-9]+)\\.' ).")
    parser_regex_capture.add_argument("--new_header", default="_captured",
                                      help="Suffix or new header for the captured column (default: '_captured').")
    
    # VIEW
    parser_view = subparsers.add_parser("view",
                                        help="Display the data in a formatted table.")
    parser_view.add_argument("--max_rows", type=int, default=20,
                             help="Maximum number of rows to display (default: 20).")
    parser_view.add_argument("--max_cols", type=int, default=None,
                             help="Maximum number of columns to display (default: all columns).")
    parser_view.add_argument("--precision_long", action="store_true",
                             help="Display numeric columns with full precision (do not round to 2 decimal places).")
    parser_view.add_argument("--cleanup_numbers", action="store_true",
                             help="Apply numeric cleanup (remove trailing decimals/round numbers) to the output.")
    parser_view.add_argument("--no-pretty-print", dest="pretty_print", action="store_false",
                             help="Output as plain TSV without pretty-print alignment.")
    parser_view.set_defaults(pretty_print=True)
    
    # CUT (select)
    parser_cut = subparsers.add_parser("select",
                                       help="Select columns. Provide either a regex pattern or, if --list is specified, a list of column names.")
    parser_cut.add_argument("pattern", nargs="?", default=None,
                        help=("Either a regex pattern for matching column names "
                              "or, when --list is specified, a comma-separated list of column names (or a file containing column names)."))
    parser_cut.add_argument("--regex", action="store_true",
                        help="Interpret the supplied pattern as a regex (default for --list is literal matching).")
    parser_cut.add_argument("--list", action="store_true",
                        help="Interpret the pattern as a comma-separated list of column names for selection in the given order.")
    
    # VIEWHEADER (headers)
    subparsers.add_parser("headers",
                          help="Display header names and positions.")
    
    # ROW_INSERT
    parser_row_insert = subparsers.add_parser("row_insert",
                                              help="Insert a new row at a specified 1-indexed position. Use --row_idx 0 to insert at the header.")
    parser_row_insert.add_argument("-i", "--row_idx", type=int, default=0,
                                   help="Row position for insertion (1-indexed, 0 for header insertion).")
    parser_row_insert.add_argument("-v", "--values",
                                   help="Comma-separated list of values for the new row. Supports escape sequences.")
    
    # ROW_DROP
    parser_row_drop = subparsers.add_parser("row_drop",
                                            help="Delete row(s) at a specified 1-indexed position. Use --row_idx 0 to drop the header row.")
    parser_row_drop.add_argument("-i", "--row_idx", type=int, required=True,
                                 help="Row position to drop (1-indexed, 0 drops the header).")
    
    # ggplot subcommand using Plotnine (plot)
    parser_ggplot = subparsers.add_parser("plot",
                                          help="Generate a ggplot using Plotnine and save to a PDF/PNG file.")
    parser_ggplot.add_argument("--geom", required=True, choices=["boxplot", "bar", "point", "hist", "tile", "pie"],
                               help="Type of plot to generate. (Note: 'pie' is not supported in ggplot mode; use the matplotlib subcommand instead.)")
    parser_ggplot.add_argument("--x", required=True, help="Column name for x aesthetic.")
    parser_ggplot.add_argument("--y", help="Column name for y aesthetic (required for boxplot and point).")
    parser_ggplot.add_argument("--fill", help="Column name for fill aesthetic (optional).")
    parser_ggplot.add_argument("--facet", help="Facet formula, e.g. 'col1 ~ col2'.")
    parser_ggplot.add_argument("--title", help="Plot title.")
    parser_ggplot.add_argument("--xlab", help="Label for x-axis.")
    parser_ggplot.add_argument("--ylab", help="Label for y-axis.")
    parser_ggplot.add_argument("--xlim", help="x-axis limits as 'min,max'.")
    parser_ggplot.add_argument("--ylim", help="y-axis limits as 'min,max'.")
    parser_ggplot.add_argument("--x-scale-log", action="store_true", help="Use logarithmic scale for x-axis.")
    parser_ggplot.add_argument("--y-scale-log", action="store_true", help="Use logarithmic scale for y-axis.")
    parser_ggplot.add_argument("-o", "--output", required=True, help="Output file name (pdf or png).")
    parser_ggplot.add_argument("--melted", action="store_true", help="Indicate that input data is already melted. (If not provided, data will be auto-detected.)")
    parser_ggplot.add_argument("--id_vars", help="Comma-separated list of columns to use as id_vars when melting (required if not wide).")
    parser_ggplot.add_argument("--value_vars", help="Comma-separated list of columns to melt. If not provided, all columns not in id_vars are melted.")
    parser_ggplot.add_argument("--figure_size", default="8,6",
                               help="Set figure size as width,height in inches (default: 8,6).")
    parser_ggplot.add_argument("--dont_replace_dots_in_colnames", action="store_true",
                               help="Do not replace '.' with '_' in column names.")
    
    # matplotlib subcommand for venn diagrams (plot_mpl)
    parser_mpl = subparsers.add_parser("plot_mpl",
                                       help="Generate a matplotlib-based plot (supports Venn diagrams) and save to a PDF/PNG file.")
    parser_mpl.add_argument("--mode", required=True, choices=["venn2", "venn3"],
                           help="Plot mode for matplotlib: 'venn2' or 'venn3'.")
    parser_mpl.add_argument("--colnames", required=True,
                           help="Comma-separated list of header names to use. (2 names for venn2; 3 names for venn3)")
    parser_mpl.add_argument("--title", help="Plot title.")
    parser_mpl.add_argument("--figure_size", default="8,6",
                           help="Set figure size as width,height in inches (default: 8,6).")
    parser_mpl.add_argument("-o", "--output", required=True,
                           help="Output file name (pdf or png).")
    
    # MELT
    parser_melt = subparsers.add_parser("melt",
                                        help="Melt the input table into a long format.")
    parser_melt.add_argument("--id_vars", required=True,
                             help="Comma-separated list of column names to use as id_vars.")
    parser_melt.add_argument("--value_vars",
                             help="Comma-separated list of column names to be melted. If not provided, all columns not in id_vars are used.")
    parser_melt.add_argument("--var_name", default="variable",
                             help="Name for the new variable column (default: 'variable').")
    parser_melt.add_argument("--value_name", default="value",
                             help="Name for the new value column (default: 'value').")
    
    # UNMELT (pivot)
    parser_unmelt = subparsers.add_parser("pivot",
                                          help="Pivot the melted table back to wide format.")
    parser_unmelt.add_argument("--index", required=True,
                               help="Column name to use as the index (row identifiers).")
    parser_unmelt.add_argument("--columns", required=True,
                               help="Column name that contains variable names (to become new columns).")
    parser_unmelt.add_argument("--value", required=True,
                               help="Column name that contains the values.")
    
    # ADD_METADATA (join_meta)
    parser_add_metadata = subparsers.add_parser("join_meta",
                                                help="Merge a metadata file into the main table based on key columns.")
    parser_add_metadata.add_argument("--meta", required=True,
                                     help="Path to the metadata file (CSV).")
    parser_add_metadata.add_argument("--key_column_in_input", required=True,
                                     help="Key column (name or 1-indexed) in the input file to join on.")
    parser_add_metadata.add_argument("--key_column_in_meta", required=True,
                                     help="Key column (name or 1-indexed) in the metadata file to join on.")
    parser_add_metadata.add_argument("--meta_sep", default=None,
                                     help="Field separator for the metadata file. If not provided, the global --sep is used.")

    # FILTER_COLUMNS (filter_cols)
    parser_filter = subparsers.add_parser("filter_cols",
        help="Filter columns based on criteria. Use --is-numeric, --is-integer, --is-same, --min_value, --max_value. With -v, selection is inverted.")
    parser_filter.add_argument("--is-numeric", action="store_true",
                               help="Match columns that are numeric (all non-null values convert to numbers).")
    parser_filter.add_argument("--is-integer", action="store_true",
                               help="Match columns that are integers (all non-null values numeric and integer-like).")
    parser_filter.add_argument("--is-same", action="store_true",
                               help="Match columns that have the same value (after conversion to numbers).")
    parser_filter.add_argument("--min_value", type=float,
                               help="Match columns where all non-null numeric values are at least this value.")
    parser_filter.add_argument("--max_value", type=float,
                               help="Match columns where all non-null numeric values are at most this value.")
    parser_filter.add_argument("-v", "--invert", action="store_true",
                               help="Invert match: keep only matching columns if set.")
    parser_filter.add_argument("--index", required=True,
                               help="Column (1-indexed or name) that will remain as index in the output (not evaluated for filtering).")
    parser_filter.add_argument("--keep_columns",
                               help="Comma-separated list of column names to always keep (ignored during filtering) and placed immediately after the index column in the output.")
    

    # PCA Plot
    parser_pca = subparsers.add_parser("pca",
                                       help=("Perform PCA on numeric columns and plot PC1 vs PC2. "
                                             "Optional color/shape legends outside, plus a loadings panel "
                                             "(VizDimLoadings-style) for PC1/PC2.")
                                       )
    parser_pca.add_argument("--index",
                            help="Column (1-indexed or name) to use as point labels (not used in PCA).")
    parser_pca.add_argument("--color_by",
                            help="Column (1-indexed or name) for point colors (categorical).")
    parser_pca.add_argument("--shape_by",
                            help="Column (1-indexed or name) for point marker shape (categorical).")
    parser_pca.add_argument("--figure_size", default="12,8",
                            help="Figure size as width,height in inches (default: 12,8).")
    parser_pca.add_argument("-o", "--output", required=True,
                            help="Output file name (pdf or png).")
    
    # PCA controls
    parser_pca.add_argument("--scale", action="store_true",
                            help="Z-scale features before PCA.")
    parser_pca.add_argument("--no_biplot", action="store_true",
                            help="Disable arrows for variable loadings in the scatter.")
    parser_pca.add_argument("--show_loadings", type=int, default=0,
                            help="(deprecated; use --top_loadings) kept for backward compat.")
    parser_pca.add_argument("--top_loadings", type=int, default=10,
                            help="Top-N features by |loading| (across PC1/PC2) to display in loadings panel.")
    
    # Layout/legend
    parser_pca.add_argument("--legend_outside", action="store_true",
                            help="Place color/shape legends outside the scatter (right side).")
    parser_pca.add_argument("--loadings_style", choices=["dots","heatmap"], default="dots",
                            help="Style for loadings panel under the scatter (default: dots).")
    parser_pca.add_argument("--loadings_height", type=float, default=0.28,
                            help="Relative height of the loadings panel (0–1, default 0.28).")
    
    
    # DETECT OUTLIERS (detect_outliers)
    parser_if = subparsers.add_parser("detect_outliers",
                                      help="Detect outliers with Isolation Forest and annotate rows.")
    parser_if.add_argument("--index", help="Label column (1-indexed or name) for readability.")
    parser_if.add_argument("--exclude", help="Comma-separated columns to exclude from features.")
    parser_if.add_argument("--contamination", default="auto",
                           help="Float (0,0.5] or 'auto' (default).")
    parser_if.add_argument("--n_estimators", type=int, default=200)
    parser_if.add_argument("--random_state", type=int, default=42)
    parser_if.add_argument("--top_k", type=int, default=3,
                           help="Top-N features by |z| to summarize per outlier.")
    parser_if.add_argument("--no_scale", action="store_true",
                           help="Do not z-scale features before modeling.")
    parser_if.add_argument("--only_outliers", action="store_true",
                           help="Emit only outlier rows in the output.")
    
    parser_if.add_argument("--plot_bars", action="store_true",
                           help="Also save a grid of bar plots.")
    parser_if.add_argument("--bars_output",
                        help="Output image for bar plots (pdf or png). Default: iforest_bars.pdf")
    parser_if.add_argument("--bars_max_outliers", type=int, default=12,
                           help="Max number of panels in the bar-plot grid (default 12).")
    parser_if.add_argument("--bars_figsize", default="12,8",
                           help="Figure size width,height for the bar-plot grid (default 12,8).")
    parser_if.add_argument("--bars_mode", choices=["by_sample","by_feature"], default="by_feature",
                           help="Bar-plot layout: 'by_feature' (one subplot per outlier feature across samples) "
                           "or 'by_sample' (one subplot per outlier sample with its top-K features).")

    # HEATMAP
    parser_heatmap = subparsers.add_parser(
        "heatmap",
        help=("Heatmap of samples (rows) × features (columns). "
              "Use --corr for feature–feature Spearman correlation.")
    )
    parser_heatmap.add_argument("--index", required=True,
                                help="Column (name or 1-indexed) with sample IDs for row labels.")
    parser_heatmap.add_argument("--row_annotations",
                                help="Comma-separated sample annotation columns to show as colored strips.")
    parser_heatmap.add_argument("--corr", action="store_true",
                                help="Plot a feature–feature correlation heatmap instead.")
    parser_heatmap.add_argument("--zscore", action="store_true",
                                help="Z-score features (columns) before plotting.")
    parser_heatmap.add_argument("--distance", choices=["1-corr", "euclidean"], default="1-corr",
                                help="Distance for dendrograms (default: 1-corr).")
    parser_heatmap.add_argument("--linkage", choices=["average","single","complete","ward"], default="average",
                                help="Linkage method for clustering.")
    parser_heatmap.add_argument("--no_row_dendro", action="store_true",
                                help="Disable row dendrogram.")
    parser_heatmap.add_argument("--no_col_dendro", action="store_true",
                                help="Disable column dendrogram.")
    parser_heatmap.add_argument("--no_cluster", action="store_true",
                                help="Shorthand to disable both row and column dendrograms.")
    parser_heatmap.add_argument("--figure_size", default="10,8",
                                help="Width,height in inches, e.g. '10,8'.")
    parser_heatmap.add_argument("--cmap", default=None,
                                help="Matplotlib colormap name (defaults to 'bwr' when zscored/corr, else 'viridis').")
    parser_heatmap.add_argument("--grid_linewidth", type=float, default=0.0,
                                help="Grid line width between tiles.")
    parser_heatmap.add_argument("--show_values", action="store_true",
                                help="Write numeric values on tiles.")
    parser_heatmap.add_argument("--values_fmt", default=".2f",
                                help="Format for --show_values (default: .2f).")
    parser_heatmap.add_argument("--xtick_fontsize", type=int, default=8)
    parser_heatmap.add_argument("--ytick_fontsize", type=int, default=7)
    parser_heatmap.add_argument("--no_annot_legend", dest="annot_legend", action="store_false", default=True,
                                help="Disable legends for annotation strips (enabled by default).")
    parser_heatmap.add_argument("--right_pad", type=float, default=0.84,
                                help="Figure right padding (0-1) to make space for legends (default: 0.84).")
    parser_heatmap.add_argument("--cbar_left", action="store_true",
                                help="Place colorbar on the left side (default places it on the right).")
    parser_heatmap.add_argument("--max_labels", type=int, default=200,
                                help="Auto-hide x/y tick labels if there are more than this many (default: 200).")
    parser_heatmap.add_argument("-o", "--output", required=True,
                                help="Output file name (.pdf or .png).")
    
    # KV UNPACK (kv_unpack)
    parser_unpack = subparsers.add_parser(
        "kv_unpack",
        help=("Unpack a column containing key-value pairs into separate columns.\n"
              "The column is split using a field separator (default: ';') and a key-value separator (default: '=').\n"
              "Specify the keys to extract using --keys (comma-separated). If a key is missing in a row, NA is used.\n"
              "New columns are named as 'originalColumn_key'. If a key appears more than once in a row (and is specified), an error is raised.")
    )
    parser_unpack.add_argument("-n", "--column", required=True,
                               help="Column (1-indexed or name) to unpack.")
    parser_unpack.add_argument("--keys", required=True,
                               help="Comma-separated list of keys to extract from the column.")
    parser_unpack.add_argument("--field_sep", default=";",
                               help="Field separator used to split the column (default: ';').")
    parser_unpack.add_argument("--kv_sep", default="=",
                               help="Key-value separator used to split key and value (default: '=').")
    
    return parser

# --------------------------
# Operation Handler Functions
# --------------------------
def _handle_unpack_column(df, args, input_sep, is_header_present, row_idx_col_name):
    import pandas as pd

    # Get the target column index and name.
    col_idx = _parse_column_arg(args.column, df.columns, is_header_present, "column")
    original_col = df.columns[col_idx]
    
    # Get the user-provided list of keys (trimmed)
    keys_list = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not keys_list:
        raise ValueError("Error: No keys provided in --keys.")

    # Decode separators (allowing escape sequences)
    field_sep = codecs.decode(args.field_sep, 'unicode_escape')
    kv_sep = codecs.decode(args.kv_sep, 'unicode_escape')

    # Define a helper function to process a single cell.
    def unpack_cell(cell):
        # Return an empty dict if cell is missing or empty.
        if pd.isna(cell) or str(cell).strip() == "":
            return {}
        # Split by the field separator.
        tokens = str(cell).split(field_sep)
        result = {}
        for token in tokens:
            token = token.strip()
            if token == "":
                continue
            # Check that the token contains the key/value separator.
            if kv_sep not in token:
                continue  # ignore tokens that do not match the key-value pattern
            parts = token.split(kv_sep, 1)
            if len(parts) != 2:
                continue
            key_candidate = parts[0].strip()
            value_candidate = parts[1].strip()
            # Remove surrounding quotes if present.
            if (value_candidate.startswith('"') and value_candidate.endswith('"')) or \
               (value_candidate.startswith("'") and value_candidate.endswith("'")):
                value_candidate = value_candidate[1:-1].strip()
            # If the key is one of the desired keys:
            if key_candidate in keys_list:
                # If already seen in this row, raise an error.
                if key_candidate in result:
                    raise ValueError(f"Error: Multiple occurrences of key '{key_candidate}' found in cell: {cell}")
                result[key_candidate] = value_candidate
        return result

    # Process the target column: create a Series of dicts.
    unpacked = df[original_col].apply(unpack_cell)
    
    # For each desired key, create a new column (assign NA if key is absent).
    new_columns = {}
    for key in keys_list:
        new_col_name = f"{original_col}_{key}"
        # Check for collision with existing columns:
        if new_col_name in df.columns:
            raise ValueError(f"Error: Generated column name '{new_col_name}' already exists in the DataFrame.")
        new_series = unpacked.apply(lambda d: d.get(key, pd.NA))
        new_columns[new_col_name] = new_series

    # Insert new columns immediately after the original column.
    # We insert them in reverse order so the final order follows keys_list.
    orig_index = df.columns.get_loc(original_col)
    for key in reversed(keys_list):
        new_col_name = f"{original_col}_{key}"
        new_series = new_columns[new_col_name]
        df.insert(orig_index + 1, new_col_name, new_series)
    
    return df






def _handle_pca(df, args, input_sep, is_header_present, row_idx_col):
    """
    PCA scatter with optional outside legends and a loadings panel.
    - Color by a categorical column (--color_by)
    - Shape by a categorical column (--shape_by)
    - Loadings panel below the scatter:
        * --loadings_style dots (VizDimLoadings style) or heatmap
        * --top_loadings N picks the top |loading| across PC1/PC2
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import math

    # Figure size
    try:
        width, height = map(float, args.figure_size.split(','))
    except Exception as e:
        raise ValueError("Invalid format for --figure_size. Expected width,height") from e

    # Resolve optional columns
    label_col = None
    color_col = None
    shape_col = None
    if getattr(args, "index", None):
        idx = _parse_column_arg(args.index, df.columns, is_header_present, "index")
        label_col = df.columns[idx]
    if getattr(args, "color_by", None):
        idx = _parse_column_arg(args.color_by, df.columns, is_header_present, "color_by")
        color_col = df.columns[idx]
    if getattr(args, "shape_by", None):
        idx = _parse_column_arg(args.shape_by, df.columns, is_header_present, "shape_by")
        shape_col = df.columns[idx]

    # Numeric slice
    exclude = [c for c in [label_col, color_col, shape_col] if c]
    numeric_df, numeric_columns = extract_numeric_columns(df, exclude_cols=exclude)
    if not numeric_columns:
        raise ValueError("No numeric columns found in the dataset for PCA.")

    # NEW: keep all samples; drop all-NA columns; impute NaNs per column
    num = numeric_df.copy()

    # Drop columns that have no finite data at all
    keep_cols = num.columns[num.notna().any(axis=0)]
    num = num.loc[:, keep_cols]
    if num.shape[1] < 2:
        raise ValueError("Not enough numeric features with data for PCA after removing all-NA columns.")

    # Median impute per feature (keeps all rows)
    X_df = num.fillna(num.median(numeric_only=True))

    row_ids = X_df.index
    numeric_columns = list(X_df.columns)   # keep in sync with loadings/features
    
    X = X_df.values.astype(float)
    if getattr(args, "scale", False):
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # PCA
    pca_model = PCA(n_components=2, random_state=None)
    scores = pca_model.fit_transform(X)
    pc1, pc2 = scores[:, 0], scores[:, 1]
    evr = pca_model.explained_variance_ratio_

    # --- Prepare color mapping
    color_values = None
    color_handles = None
    color_title = None
    if color_col:
        s = df.loc[row_ids, color_col].where(df.loc[row_ids, color_col].notna(), "NA").astype(str)
        cats = list(pd.Categorical(s).categories)
        try:
            cmap = plt.colormaps.get_cmap('tab20')
        except AttributeError:
            cmap = plt.cm.get_cmap('tab20')
        color_map = {cat: cmap(i % 20) for i, cat in enumerate(cats)}
        color_values = s.map(color_map).values
        color_title = color_col
        color_handles = [Line2D([0],[0], marker='o', linestyle='',
                                markerfacecolor=color_map[c], markeredgecolor='none',
                                label=str(c)) for c in cats]

    # --- Prepare shape mapping
    marker_values = None
    shape_handles = None
    shape_title = None
    if shape_col:
        s = df.loc[row_ids, shape_col].where(df.loc[row_ids, shape_col].notna(), "NA").astype(str)
        cats = list(pd.Categorical(s).categories)
        markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
        shape_map = {cat: markers[i % len(markers)] for i, cat in enumerate(cats)}
        marker_values = s.map(shape_map).values
        shape_title = shape_col
        shape_handles = [Line2D([0],[0], marker=shape_map[c], linestyle='',
                                markerfacecolor='gray', markeredgecolor='gray',
                                label=str(c)) for c in cats]

    # --- Layout: scatter on top, loadings panel on bottom (2 columns)
    # height ratios: [1 - loadings_height, loadings_height]
    lh = max(0.05, min(0.9, float(getattr(args, "loadings_height", 0.28))))
    fig = plt.figure(figsize=(width, height))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, height_ratios=[1.0 - lh, lh], width_ratios=[1, 1], hspace=0.25, wspace=0.15)
    ax = fig.add_subplot(gs[0, :])  # scatter spans both columns

    # Draw scatter (by groups if shape/color provided)
    if color_values is None and marker_values is None:
        ax.scatter(pc1, pc2, edgecolors='none')
    else:
        # group by (color category, shape category) for consistent legends
        if color_col:
            color_cat = df.loc[row_ids, color_col].astype(str).values
        else:
            color_cat = np.array(["_single"] * len(row_ids))
        if shape_col:
            shape_cat = df.loc[row_ids, shape_col].astype(str).values
        else:
            shape_cat = np.array(["_single"] * len(row_ids))

        for cc in np.unique(color_cat):
            for sc in np.unique(shape_cat):
                mask = (color_cat == cc) & (shape_cat == sc)
                if not np.any(mask):
                    continue
                c = color_map[cc] if color_col else None
                m = shape_map[sc] if shape_col else 'o'
                ax.scatter(pc1[mask], pc2[mask], c=[c] if c else None, marker=m, edgecolors='none')

    ax.set_xlabel(f"PC1 ({evr[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.2f}%)")
    ax.set_title("PCA Plot")
    ax.grid(True, alpha=0.3)

    # Optional tiny biplot arrows (off by default)
    if not getattr(args, "no_biplot", False):
        load = pca_model.components_.T  # (features x PCs)
        # auto scale arrows to ~1/3 of scatter extent
        score_extent = np.max(np.sqrt(pc1**2 + pc2**2)) or 1.0
        load_norm = np.max(np.sqrt(np.sum(load[:, :2]**2, axis=1))) or 1.0
        scale = 0.33 * score_extent / load_norm
        for i, var in enumerate(numeric_columns):
            vx, vy = (load[i, 0] * scale, load[i, 1] * scale)
            ax.arrow(0, 0, vx, vy, color='red', alpha=0.35, width=0.0,
                     length_includes_head=True, head_width=0.02*score_extent)

    # keep track of legends so savefig knows to include them
    legend_artists = []

    # ---- Legends (outside or inside)
    def _place_legends_outside(ax, color_handles, shape_handles, color_title, shape_title):
        blocks = [b for b in [(color_handles, color_title), (shape_handles, shape_title)] if b[0]]
        if not blocks:
            return
        # room on the right for one vs two stacked legends
        fig.subplots_adjust(right=(0.80 if len(blocks) == 1 else 0.70))
        y = 1.00
        for handles, title in blocks:
            leg = ax.legend(handles=handles, title=title,
                            loc="upper left", bbox_to_anchor=(1.01, y),
                            frameon=True, fontsize="small", title_fontsize="small")
            ax.add_artist(leg)
            legend_artists.append(leg)     # <-- register for savefig
            y -= 0.32

    if getattr(args, "legend_outside", False):
        _place_legends_outside(ax, color_handles, shape_handles, color_title, shape_title)
    else:
        if color_handles:
            leg1 = ax.legend(handles=color_handles, title=color_title,
                             loc="best", frameon=True, fontsize="small", title_fontsize="small")
            ax.add_artist(leg1)
        if shape_handles:
            leg2 = ax.legend(handles=shape_handles, title=shape_title,
                             loc="upper left", frameon=True, fontsize="small", title_fontsize="small")
            ax.add_artist(leg2)

    # Labels (avoid clutter)
    if label_col:
        labels = df.loc[row_ids, label_col].astype(str)
        if len(labels) <= 60:
            for x, y, txt in zip(pc1, pc2, labels):
                ax.annotate(txt, (x, y), fontsize=6, xytext=(2, 2), textcoords='offset points')

    # --- Loadings panel (PC1 & PC2)
    # Select top features by max(|loading_pc1|, |loading_pc2|)
    load = pca_model.components_.T  # features x PCs
    pc1_load, pc2_load = load[:, 0], load[:, 1]
    mag = np.maximum(np.abs(pc1_load), np.abs(pc2_load))
    n = int(getattr(args, "top_loadings", 10) or 10)
    keep_idx = np.argsort(-mag)[:max(1, n)]
    feat = [numeric_columns[i] for i in keep_idx]
    pc1_k = pc1_load[keep_idx]
    pc2_k = pc2_load[keep_idx]

    if getattr(args, "loadings_style", "dots") == "heatmap":
        # simple 2-column heatmap fallback
        from matplotlib.colors import TwoSlopeNorm
        ax_hm = fig.add_subplot(gs[1, :])
        mat = np.vstack([pc1_k, pc2_k]).T
        vmax = np.max(np.abs(mat)) or 1.0
        im = ax_hm.imshow(mat, aspect='auto', cmap='bwr', norm=TwoSlopeNorm(0, -vmax, vmax))
        ax_hm.set_yticks(range(len(feat)))
        ax_hm.set_yticklabels(feat, fontsize=8)
        ax_hm.set_xticks([0,1])
        ax_hm.set_xticklabels(["PC1","PC2"], fontsize=9)
        ax_hm.set_title("Top loadings")
        cb = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=8)
    else:
        # VizDimLoadings-style dot plots: left=PC1, right=PC2 (shared y)
        ax_l = fig.add_subplot(gs[1, 0])
        ax_r = fig.add_subplot(gs[1, 1], sharey=ax_l)

        # order features from strongest to weakest for nice stacking
        order = np.argsort(-mag[keep_idx])
        feat = [feat[i] for i in order]
        pc1_k = pc1_k[order]
        pc2_k = pc2_k[order]

        y = np.arange(len(feat))
        maxabs = max(np.max(np.abs(pc1_k)), np.max(np.abs(pc2_k)), 0.1)
        lim = 1.05 * maxabs

        for axd, vals, title in [(ax_l, pc1_k, "PC1 loadings"), (ax_r, pc2_k, "PC2 loadings")]:
            axd.axvline(0, lw=0.8, ls='--', alpha=0.6)
            axd.scatter(vals, y, s=20)
            axd.set_xlim(-lim, lim)
            axd.set_yticks(y)
            axd.set_yticklabels(feat, fontsize=7)
            axd.set_xlabel(title, fontsize=9)
            axd.grid(True, axis='x', alpha=0.2)
        # Only show y tick labels on the left plot
        plt.setp(ax_r.get_yticklabels(), visible=False)

    # Save
    ext = os.path.splitext(args.output)[1].lower().lstrip(".") or "pdf"
    fig.savefig(args.output, format=ext, dpi=300,
                bbox_inches="tight",
                bbox_extra_artists=legend_artists)   # <-- important
    plt.close(fig)
    sys.exit(0)

def _handle_isolation_forest(df, args, input_sep, is_header_present, row_idx_col):
    """
    Detect outlier samples via Isolation Forest and annotate the table.

    Adds columns:
      - iforest_score       (float; higher = more anomalous)
      - iforest_is_outlier  (bool)
      - iforest_topk        (semicolon list of top-K features by |z| for that sample)
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest

    # Resolve optional index col (not required)
    label_col = None
    if getattr(args, "index", None):
        idx = _parse_column_arg(args.index, df.columns, is_header_present, "index")
        label_col = df.columns[idx]

    # Exclusions
    exclude = []
    if label_col:
        exclude.append(label_col)
    if getattr(args, "exclude", None):
        exclude += [c.strip() for c in args.exclude.split(",") if c.strip()]

    # Numeric slice
    numeric_df, numeric_columns = _helper_extract_numeric_columns(df, exclude_cols=exclude)
    if not numeric_columns:
        raise ValueError("No numeric columns found for Isolation Forest.")
    # Keep only complete rows for modeling
    # NEW: keep all samples; drop all-NA columns; median impute
    num = numeric_df.loc[:, numeric_df.notna().any(axis=0)]
    if num.shape[1] == 0:
        raise ValueError("No usable numeric features (all columns are NA).")
    
    X_df = num.fillna(num.median(numeric_only=True))

    # Optional scaling: z-score with protection for constant columns (std==0)
    if getattr(args, "no_scale", False):
        Z = X_df.values
        Z_for_topk = (X_df - X_df.mean()).div(X_df.std(ddof=0).replace(0, np.nan)).fillna(0.0)
    else:
        mean = X_df.mean()
        std = X_df.std(ddof=0).replace(0, np.nan)
        Z_for_topk = (X_df - mean).div(std).fillna(0.0)
        Z = Z_for_topk.values  # model on scaled features

    # Model params
    contamination = getattr(args, "contamination", "auto") or "auto"
    if contamination != "auto":
        try:
            contamination = float(contamination)
        except ValueError:
            raise ValueError("--contamination must be a float or 'auto'.")
    n_estimators = int(getattr(args, "n_estimators", 200) or 200)
    random_state = int(getattr(args, "random_state", 42) or 42)
    top_k = int(getattr(args, "top_k", 3) or 3)

    # Fit
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    model.fit(Z)

    # Scores/preds for modeled rows
    raw = model.score_samples(Z)          # higher = less abnormal
    score = -raw                          # higher = more abnormal
    pred = model.predict(Z)               # 1 normal, -1 outlier
    is_out = (pred == -1)

    # Attach results to a full-length Series aligned to original df
    score_s = pd.Series(index=df.index, dtype=float)
    score_s.loc[X_df.index] = score
    flag_s = pd.Series(False, index=df.index)
    flag_s.loc[X_df.index] = is_out

    # Top-K features by |z|
    topk_s = pd.Series(index=df.index, dtype=object)
    if top_k > 0:
        for ridx in X_df.index:
            zrow = Z_for_topk.loc[ridx].abs().sort_values(ascending=False)
            feats = [f"{c}:{zrow[c]:.2f}" for c in zrow.index[:top_k]]
            topk_s.loc[ridx] = ";".join(feats)

    # Append columns
    out = df.copy()
    out.insert(len(out.columns), "iforest_score", score_s)
    out.insert(len(out.columns), "iforest_is_outlier", flag_s)
    out.insert(len(out.columns), "iforest_topk", topk_s)

    # Optionally keep only outliers
    if getattr(args, "only_outliers", False):
        out = out[out["iforest_is_outlier"] == True]


    # Optionally plot bar charts per outlier
    # Optionally plot bar charts
    if getattr(args, "plot_bars", False):
        import math
        import matplotlib.pyplot as plt

        # Helper: parse fig size
        try:
            bw, bh = map(float, str(getattr(args, "bars_figsize", "12,8")).split(','))
        except Exception:
            bw, bh = 12.0, 8.0

        outliers_idx = list(X_df.index[is_out])
        if len(outliers_idx) == 0:
            sys.stderr.write("NOTE: No outliers detected; no bar plots generated.\n")
        else:
            mode = getattr(args, "bars_mode", "by_feature")

            if mode == "by_sample":
                # --- existing behavior: one subplot per outlier *sample* (top-K signed z)
                max_panels = max(1, int(getattr(args, "bars_max_outliers", 12)))
                panel_ids = outliers_idx[:max_panels]

                n = len(panel_ids)
                ncols = 3 if n >= 3 else n
                nrows = int(math.ceil(n / ncols))
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(bw, bh), squeeze=False)
                axes = axes.ravel()

                # build data & global xlim
                import numpy as np
                max_abs = 0.0
                per_out = {}
                for ridx in panel_ids:
                    zrow = Z_for_topk.loc[ridx]
                    zabs = zrow.abs().sort_values(ascending=False)
                    feats = list(zabs.index[:top_k])
                    vals = zrow[feats].values
                    per_out[ridx] = (feats, vals)
                    if len(vals):
                        max_abs = max(max_abs, float(np.max(np.abs(vals))))
                xlim = 1.2 * (max_abs if max_abs > 0 else 1.0)

                for ax, ridx in zip(axes, panel_ids):
                    feats, vals = per_out[ridx]
                    ax.axvline(0, lw=0.8, ls='--', alpha=0.6)
                    ax.barh(range(len(feats)), vals, align='center')
                    ax.set_yticks(range(len(feats)))
                    ax.set_yticklabels(feats, fontsize=7)
                    label = str(df.loc[ridx, label_col]) if label_col else str(ridx)
                    score_str = f"{score_s.loc[ridx]:.3f}" if pd.notna(score_s.loc[ridx]) else "NA"
                    ax.set_title(f"{label} (score {score_str})", fontsize=9)
                    ax.set_xlim(-xlim, xlim)
                    ax.grid(True, axis='x', alpha=0.2)

                for ax in axes[len(panel_ids):]:
                    ax.axis('off')

            else:
                # --- NEW: one subplot per outlier *feature* across all samples
                # union of top-K features from each outlier sample, ordered by max |z| among outliers
                import numpy as np

                feat_scores = {}
                for ridx in outliers_idx:
                    zrow = Z_for_topk.loc[ridx].abs().sort_values(ascending=False)
                    for f in zrow.index[:top_k]:
                        feat_scores[f] = max(feat_scores.get(f, 0.0), float(abs(Z_for_topk.loc[ridx, f])))

                if not feat_scores:
                    sys.stderr.write("NOTE: No top features found among outliers; no feature panels.\n")
                else:
                    ordered_features = [f for f,_ in sorted(feat_scores.items(), key=lambda kv: -kv[1])]
                    max_panels = max(1, int(getattr(args, "bars_max_outliers", 12)))
                    features_to_plot = ordered_features[:max_panels]

                    n = len(features_to_plot)
                    ncols = 3 if n >= 3 else n
                    nrows = int(math.ceil(n / ncols))
                    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(bw, bh), squeeze=False)
                    axes = axes.ravel()

                    # colors
                    idx_all = list(X_df.index)  # modeled rows
                    is_out_series = pd.Series(False, index=idx_all)
                    is_out_series.loc[outliers_idx] = True
                    col_out = "#d62728"  # red
                    col_in  = "#bdbdbd"  # gray

                    # x labels density
                    show_xlabels = len(idx_all) <= 25

                    # consistent y-limits across panels
                    max_abs = 0.0
                    for f in features_to_plot:
                        vals = Z_for_topk[f].loc[idx_all].values  # signed z
                        if len(vals):
                            max_abs = max(max_abs, float(np.max(np.abs(vals))))
                    ylim = 1.2 * (max_abs if max_abs > 0 else 1.0)

                    for ax, f in zip(axes, features_to_plot):
                        zvals = Z_for_topk[f].loc[idx_all]  # signed z across samples
                        colors = [col_out if is_out_series.loc[i] else col_in for i in idx_all]
                        ax.axhline(0, lw=0.8, ls='--', alpha=0.6)
                        ax.bar(range(len(idx_all)), zvals.values, color=colors)
                        ax.set_title(f, fontsize=9)
                        ax.set_ylim(-ylim, ylim)
                        if show_xlabels:
                            xt = [str(df.loc[i, label_col]) if label_col else str(i) for i in idx_all]
                            ax.set_xticks(range(len(idx_all)))
                            ax.set_xticklabels(xt, rotation=60, ha='right', fontsize=7)
                        else:
                            ax.set_xticks([])
                        ax.grid(True, axis='y', alpha=0.2)

                    for ax in axes[len(features_to_plot):]:
                        ax.axis('off')

                    # simple legend
                    from matplotlib.lines import Line2D
                    handles = [Line2D([0],[0], color=col_out, lw=6, label="Outlier samples"),
                               Line2D([0],[0], color=col_in,  lw=6, label="Non-outliers")]
                    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize='small')

            out_path = getattr(args, "bars_output", None) or "iforest_bars.pdf"
            ext = os.path.splitext(out_path)[1].lower().lstrip(".") or "pdf"
            fig.tight_layout()
            fig.savefig(out_path, format=ext, bbox_inches='tight', dpi=300)
            plt.close(fig)
    return out

#--

def _handle_heatmap(df, args, input_sep, is_header_present, row_idx_col_name):
    """
    Heatmap of samples (rows) x features (columns) with optional dendrograms and sample annotations.
    If --corr is given, a feature-feature Spearman correlation heatmap is shown instead.
    """
    import sys, os
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Figure size
    try:
        if ',' in args.figure_size:
            w, h = map(float, args.figure_size.split(','))
        elif 'x' in args.figure_size.lower():
            w, h = map(float, args.figure_size.lower().split('x'))
        else:
            w = h = float(args.figure_size)
    except Exception:
        sys.stderr.write("Error: --figure_size must be like '8,6' or '8x6'.\n")
        sys.exit(1)

    # --index is required
    if not getattr(args, "index", None):
        sys.stderr.write("Error: --index is required for 'heatmap'.\n")
        sys.exit(1)
    idx = _parse_column_arg(args.index, df.columns, is_header_present, "index")
    label_col = df.columns[idx]

    # Ensure the dataframe index is the sample label so row ticklabels are sample names
    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    df.set_index(label_col, inplace=True, drop=False)

    # Annotations (optional)
    row_ann_cols = []
    if getattr(args, "row_annotations", None):
        row_ann_cols = [c.strip() for c in args.row_annotations.split(',') if c.strip()]

    corr_mode = bool(getattr(args, "corr", False))

    # numeric slice (exclude label + annotations)
    exclude = [label_col] + row_ann_cols
    numeric_df, numeric_columns = _helper_extract_numeric_columns(df, exclude_cols=exclude)

    if corr_mode:
        if not numeric_columns:
            raise ValueError("No numeric columns found for correlation heatmap.")
        data_to_plot = _corr_from_df(numeric_df)                 # feature-feature corr
        row_colors_df = None                                     # not used in corr mode
        center = 0.0
        cmap = getattr(args, "cmap", None) or "bwr_r"            # red=neg, blue=pos
    else:
        if not numeric_columns:
            raise ValueError("No numeric columns found for heatmap.")
        M = numeric_df.copy()

        # z-score per column (optional)
        if getattr(args, "zscore", False):
            mu = M.mean(axis=0)
            sigma = M.std(axis=0, ddof=0).replace(0, np.nan)
            M = (M - mu).div(sigma)
            center = 0.0
            cmap = getattr(args, "cmap", None) or "bwr_r"        # red=neg, blue=pos
        else:
            center = None
            cmap = getattr(args, "cmap", None) or "viridis"

        # drop all-NA (constant) features after scaling
        all_na_cols = [c for c in M.columns if M[c].isna().all()]
        if all_na_cols:
            sys.stderr.write("Warning: Dropping constant/non-informative feature(s): "
                             + ", ".join(all_na_cols) + "\n")
            M = M.drop(columns=all_na_cols)
            numeric_columns = [c for c in numeric_columns if c not in all_na_cols]

        # build row annotation color matrix (aligned to M)
        row_colors_df = None
        if row_ann_cols:
            ann = df.loc[M.index, row_ann_cols]
            row_colors_df = pd.DataFrame(index=ann.index)
            for col in row_ann_cols:
                s = ann[col].astype('string')  # keeps <NA>
                cats = pd.Categorical(s).categories
                pal = sns.color_palette(None, n_colors=max(3, len(cats)))
                color_map = {cat: pal[i % len(pal)] for i, cat in enumerate(cats)}
                # Map with default tuple (avoid Series.fillna(tuple) error)
                row_colors_df[col] = s.map(lambda k: color_map.get(k, (0.8, 0.8, 0.8)))

        data_to_plot = M

    if data_to_plot.empty:
        sys.stderr.write("Error: Nothing to plot (empty matrix).\n")
        sys.exit(1)

    # clustering options
    cluster_rows = not getattr(args, "no_row_dendro", False)
    cluster_cols = not getattr(args, "no_col_dendro", False)
    if getattr(args, "no_cluster", False):
        cluster_rows = False
        cluster_cols = False

    linkage_method = getattr(args, "linkage", "average")
    distance_kind = getattr(args, "distance", "1-corr")  # '1-corr' or 'euclidean'
    if linkage_method == "ward" and distance_kind != "euclidean":
        sys.stderr.write("Warning: 'ward' linkage requires Euclidean distances; switching distance=euclidean.\n")
        distance_kind = "euclidean"

    # Linkages (robust, all finite)
    row_linkage = col_linkage = None
    if cluster_cols:
        if distance_kind == "1-corr":
            cmat = data_to_plot if corr_mode else _corr_from_df(data_to_plot)
            col_linkage = _link_from_corr(cmat, method=linkage_method)
        else:
            Z = data_to_plot.fillna(0.0)
            col_linkage = linkage(np.nan_to_num(pdist(Z.T, 'euclidean')), method=linkage_method)

    if cluster_rows:
        if distance_kind == "1-corr":
            cmat = data_to_plot if corr_mode else _corr_from_df(data_to_plot.T)
            row_linkage = _link_from_corr(cmat, method=linkage_method)
        else:
            Z = data_to_plot.fillna(0.0)
            row_linkage = linkage(np.nan_to_num(pdist(Z, 'euclidean')), method=linkage_method)

    # Colorbar placement and right padding
    cbar_pos = (.02, .8, .04, .18) if getattr(args, "cbar_left", False) else None
    right_pad = float(getattr(args, "right_pad", 0.84))

    # Decide whether to draw tick labels (avoid slow text drawing)
    max_labels = int(getattr(args, "max_labels", 200))
    xticks_on = data_to_plot.shape[1] <= max_labels
    yticks_on = data_to_plot.shape[0] <= max_labels

    plot_df = data_to_plot.copy()
    plot_mask = plot_df.isna()
    plot_df = plot_df.fillna(0.0)

    g = sns.clustermap(
        plot_df,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        row_colors=row_colors_df if (row_colors_df is not None and not corr_mode) else None,
        cmap=cmap,
        center=center,
        xticklabels=xticks_on,
        yticklabels=yticks_on,
        cbar_pos=cbar_pos,
        figsize=(w, h),
        linewidths=getattr(args, "grid_linewidth", 0.0),
    )

    ax = g.ax_heatmap
    ax.set_xlabel(""); ax.set_ylabel("")
    if xticks_on:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center',
                           fontsize=getattr(args, "xtick_fontsize", 8))
    if yticks_on:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=getattr(args, "ytick_fontsize", 7))

    # annotation legends on the right
    if row_ann_cols and not corr_mode and getattr(args, "annot_legend", True):
        legend_ax = g.fig.add_axes([right_pad, 0.15, 0.15, 0.7])  # x,y,w,h
        legend_ax.axis("off")
        y0, dy = 0.95, 0.08
        # reorder annotations to plotting order
        ordered_rows = g.dendrogram_row.reordered_ind if cluster_rows else list(range(plot_df.shape[0]))
        ann_df = df.loc[plot_df.index[ordered_rows], row_ann_cols]

        for j, col in enumerate(row_ann_cols):
            s = ann_df[col].astype('string')
            cats = pd.Categorical(s).categories.tolist()  # excludes NA
            pal = sns.color_palette(None, n_colors=max(3, len(cats)))
            cm = {cat: pal[i % len(pal)] for i, cat in enumerate(cats)}
            patches = [Patch(facecolor=cm[c], edgecolor='none', label=str(c)) for c in cats]
            # If we had any NA, show it explicitly
            if s.isna().any():
                patches.append(Patch(facecolor=(0.8,0.8,0.8), edgecolor='none', label="NA"))
            legend_ax.legend(handles=patches, title=col, loc='upper left',
                             bbox_to_anchor=(0.0, y0 - j*dy), frameon=False,
                             fontsize=8, title_fontsize=9)

    # write values on tiles (optional)
    if getattr(args, "show_values", False):
        rows = g.dendrogram_row.reordered_ind if cluster_rows else slice(None)
        cols = g.dendrogram_col.reordered_ind if cluster_cols else slice(None)
        data_reordered = plot_df.iloc[rows, cols]
        mask_reordered = plot_mask.iloc[rows, cols]
        ny, nx = data_reordered.shape
        for yi in range(ny):
            for xi in range(nx):
                if mask_reordered.iat[yi, xi]:
                    continue
                val = data_reordered.iat[yi, xi]
                ax.text(xi + 0.5, yi + 0.5, f"{val:{getattr(args, 'values_fmt', '.2f')}}",
                        ha='center', va='center', fontsize=getattr(args, "values_fontsize", 6))

    g.fig.subplots_adjust(right=right_pad)
    ext = os.path.splitext(args.output)[1].lower().lstrip(".") or "pdf"
    g.fig.savefig(args.output, format=ext, bbox_inches='tight', dpi=300)
    plt.close(g.fig)
    sys.exit(0)

#--

def _handle_filter_columns(df, args, input_sep, is_header_present, row_idx_col_name):
    index_col_name = None
    if hasattr(args, "index") and args.index:
        index_idx = _parse_column_arg(args.index, df.columns, is_header_present, "index")
        index_col_name = df.columns[index_idx]
    
    keep_columns = []
    if hasattr(args, "keep_columns") and args.keep_columns:
        keep_columns = [col.strip() for col in args.keep_columns.split(",") if col.strip()]
    
    cols_to_filter = []
    for col in df.columns:
        if index_col_name is not None and col == index_col_name:
            continue
        if col in keep_columns:
            continue
        series = df[col].dropna()
        match = False
        if args.is_numeric:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not series.empty and numeric_series.notna().all():
                match = True
        if args.is_integer:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if (not series.empty and numeric_series.notna().all() and
                numeric_series.apply(lambda x: abs(x - round(x)) < 1e-8).all()):
                match = True
        if args.is_same:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if (not series.empty and numeric_series.notna().all() and numeric_series.nunique() == 1):
                match = True
        if args.min_value is not None:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if (not series.empty and numeric_series.notna().all() and (numeric_series >= args.min_value).all()):
                match = True
        if args.max_value is not None:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if (not series.empty and numeric_series.notna().all() and (numeric_series <= args.max_value).all()):
                match = True
        if match:
            cols_to_filter.append(col)
    
    if args.invert:
        cols_to_keep = cols_to_filter[:]
        if index_col_name is not None and index_col_name not in cols_to_keep:
            cols_to_keep.append(index_col_name)
        for k in keep_columns:
            if k not in cols_to_keep and k in df.columns:
                cols_to_keep.append(k)
        filtered_df = df.loc[:, cols_to_keep]
    else:
        filtered_df = df.drop(columns=cols_to_filter)
    
    if index_col_name is not None:
        final_order = [index_col_name]
    else:
        final_order = []
    for k in keep_columns:
        if k in filtered_df.columns and k not in final_order:
            final_order.append(k)
    for col in filtered_df.columns:
        if col not in final_order:
            final_order.append(col)
    
    filtered_df = filtered_df.reindex(columns=final_order)
    
    if index_col_name is not None:
        filtered_df = filtered_df.set_index(index_col_name, drop=False)
    return filtered_df

def _handle_aggregate(df, args, input_sep, is_header_present, row_idx_col_name):
    import sys
    import pandas as pd
    from scipy.stats import entropy as calculate_entropy

    if getattr(args, "melted", False):
        required_cols = {"variable", "value"}
        if not required_cols.issubset(set(df.columns)):
            sys.stderr.write("Error: When using --melted, the input must have 'variable' and 'value' columns.\n")
            sys.exit(1)
        group_cols = []
        if getattr(args, "group", None):
            group_cols = [col.strip() for col in args.group.split(",") if col.strip()]
        if "variable" not in group_cols:
            group_cols.append("variable")
        agg_func = args.agg.lower()
        summary_rows = []
        grouped = df.groupby(group_cols)
        for grp_keys, grp_df in grouped:
            if not isinstance(grp_keys, tuple):
                grp_keys = (grp_keys,)
            group_dict = dict(zip(group_cols, grp_keys))
            series = grp_df["value"]
            if agg_func in ["sum", "mean"]:
                try:
                    series_numeric = pd.to_numeric(series, errors="raise")
                except Exception as e:
                    sys.stderr.write(f"Error: Cannot convert 'value' column to numeric for aggregation: {e}\n")
                    sys.exit(1)
                result = series_numeric.sum() if agg_func == "sum" else series_numeric.mean()
                group_dict.update({f"{agg_func}_value": result})
                summary_rows.append(group_dict)
            elif agg_func == "list":
                group_dict.update({"list_value": ",".join(series.astype(str))})
                summary_rows.append(group_dict)
            elif agg_func == "value_counts":
                normalize = getattr(args, "normalize", False)
                vc = series.value_counts(normalize=normalize).reset_index()
                if normalize:
                    vc.columns = ["value", "normalized"]
                    vc["raw_count"] = series.value_counts(normalize=False).reindex(vc["value"]).values
                else:
                    vc.columns = ["value", "count"]
                for _, row in vc.iterrows():
                    out = group_dict.copy()
                    out["aggregated_column"] = "value"
                    out["value"] = row["value"]
                    if normalize:
                        out.update({"raw_count": row["raw_count"], "normalized": row["normalized"]})
                    else:
                        out["count"] = row["count"]
                    summary_rows.append(out)
            elif agg_func == "entropy":
                counts = series.value_counts()
                ent = calculate_entropy(counts)
                group_dict.update({"entropy": ent})
                summary_rows.append(group_dict)
            else:
                sys.stderr.write(f"Error: Unsupported aggregation function '{agg_func}' for melted data.\n")
                sys.exit(1)
        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    else:
        if not getattr(args, "group", None):
            sys.stderr.write("Error: In wide format, you must supply --group to specify grouping variable(s).\n")
            sys.exit(1)
        if not getattr(args, "cols", None):
            sys.stderr.write("Error: In wide format, you must supply --cols with the comma-separated list of columns to aggregate (or '*' for all non-group columns).\n")
            sys.exit(1)
        group_cols = [col.strip() for col in args.group.split(",") if col.strip()]
        if args.cols.strip() in ["*", "all"]:
            agg_cols = [col for col in df.columns if col not in group_cols]
        else:
            agg_cols = [col.strip() for col in args.cols.split(",") if col.strip()]
        agg_func = args.agg.lower()
        if agg_func in ["sum", "mean"]:
            valid_agg_cols = []
            for col in agg_cols:
                series_numeric = pd.to_numeric(df[col], errors="coerce")
                if series_numeric.notna().sum() > 0:
                    valid_agg_cols.append(col)
                else:
                    if getattr(args, "verbose", False):
                        sys.stderr.write(f"Warning: Skipping non-numeric column '{col}' for aggregator '{agg_func}'.\n")
            if not valid_agg_cols:
                sys.stderr.write("Error: No numeric columns found for aggregation.\n")
                sys.exit(1)
            for col in valid_agg_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            grouped = df.groupby(group_cols)
            summary_df = grouped[valid_agg_cols].agg(agg_func).reset_index()
            rename_dict = {col: f"{agg_func}_{col}" for col in valid_agg_cols}
            summary_df.rename(columns=rename_dict, inplace=True)
            return summary_df

        elif agg_func == "list":
            grouped = df.groupby(group_cols)
            summary_df = grouped[agg_cols].agg(lambda x: ",".join(x.astype(str))).reset_index()
            rename_dict = {col: f"list_{col}" for col in agg_cols}
            summary_df.rename(columns=rename_dict, inplace=True)
            return summary_df

        elif agg_func in ["value_counts", "entropy"]:
            summary_rows = []
            grouped = df.groupby(group_cols)
            for grp_keys, grp_df in grouped:
                if not isinstance(grp_keys, tuple):
                    grp_keys = (grp_keys,)
                group_dict = dict(zip(group_cols, grp_keys))
                for col in agg_cols:
                    series = grp_df[col]
                    if agg_func == "value_counts":
                        normalize = getattr(args, "normalize", False)
                        vc = series.value_counts(normalize=normalize).reset_index()
                        if normalize:
                            vc.columns = ["value", "normalized"]
                            vc["raw_count"] = series.value_counts(normalize=False).reindex(vc["value"]).values
                        else:
                            vc.columns = ["value", "count"]
                        for _, row in vc.iterrows():
                            out = group_dict.copy()
                            out["aggregated_column"] = col
                            out["value"] = row["value"]
                            if normalize:
                                out.update({"raw_count": row["raw_count"], "normalized": row["normalized"]})
                            else:
                                out["count"] = row["count"]
                            summary_rows.append(out)
                    elif agg_func == "entropy":
                        counts = series.value_counts()
                        ent = calculate_entropy(counts)
                        out = group_dict.copy()
                        out["aggregated_column"] = col
                        out["entropy"] = ent
                        summary_rows.append(out)
            summary_df = pd.DataFrame(summary_rows)
            return summary_df

        else:
            sys.stderr.write(f"Error: Unsupported aggregation function '{agg_func}'.\n")
            sys.exit(1)

def _handle_transpose(df, args, input_sep, is_header_present, row_idx_col_name):
    if is_header_present:
        header_row = list(df.columns)
        df = pd.concat([pd.DataFrame([header_row], columns=df.columns), df], ignore_index=True)
    if df.shape[1] < 2:
        sys.stderr.write("Error: Input table must have at least 2 columns for transpose operation (first column will be used as header).\n")
        sys.exit(1)
    new_headers = df.iloc[:, 0].tolist()
    data_to_transpose = df.iloc[:, 1:]
    transposed_df = data_to_transpose.T
    transposed_df.columns = new_headers
    return transposed_df

def _handle_move(df, args, input_sep, is_header_present, row_idx_col_name):
    from_idx = _parse_column_arg(args.column, df.columns, is_header_present, "source column (--column)")
    to_idx = _parse_column_arg(args.dest_column, df.columns, is_header_present, "destination column (--dest_column)")
    if to_idx > df.shape[1]:
        to_idx = df.shape[1]
    _print_verbose(args, f"Moving column '{df.columns[from_idx]}' from index {from_idx} to {to_idx}.")
    col_name = df.columns[from_idx]
    data = df.pop(col_name)
    df.insert(to_idx, col_name, data)
    return df

def _handle_col_insert(df, args, input_sep, is_header_present, row_idx_col_name):
    pos = _parse_column_arg(args.column, df.columns, is_header_present, "column (--column)")
    value = codecs.decode(args.value, 'unicode_escape')
    new_header = args.new_header
    if is_header_present and new_header in df.columns:
        new_header = get_unique_header(new_header, df)
        _print_verbose(args, f"New header exists; using '{new_header}'.")
    df.insert(pos, new_header, value)
    return df

def _handle_col_drop(df, args, input_sep, is_header_present, row_idx_col_name):
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present, "columns (--column)")
    names = [df.columns[i] for i in indices]
    _print_verbose(args, f"Dropping columns: {names}.")
    df = df.drop(columns=names)
    return df

def _handle_grep(df, args, input_sep, is_header_present, row_idx_col_name, state=None):
    col = _parse_column_arg(args.column, df.columns, is_header_present, "column (--column)")
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
    col = _parse_column_arg(args.column, df.columns, is_header_present, "column (--column)")
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
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present, "columns (--column)")
    if not indices:
        raise ValueError("Error: No columns specified for join operation.")
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    names = [df.columns[i] for i in indices]
    _print_verbose(args, f"Joining columns {names} with delimiter {delim!r}.")
    joined = df.iloc[:, indices[0]].astype(str)
    for i in range(1, len(indices)):
        joined += delim + df.iloc[:, indices[i]].astype(str)
    if args.target_column:
        target = _parse_column_arg(args.target_column, df.columns, is_header_present, "target column (--target_column)")
    else:
        target = indices[0]
    new_header = args.new_header
    if is_header_present:
        new_header = get_unique_header(new_header, df)
    # compute how many dropped columns lie before the target position
    before_target = sum(1 for i in indices if i < target)
    drop_names = [df.columns[i] for i in sorted(indices, reverse=True)]
    df = df.drop(columns=drop_names)
    pos = min(max(target - before_target, 0), df.shape[1])
    df.insert(pos, new_header, joined.reset_index(drop=True))
    _print_verbose(args, f"Inserted joined column '{new_header}' at index {pos}.")
    return df

def _handle_tr(df, args, input_sep, is_header_present, row_idx_col_name):
    col = _parse_column_arg(args.column, df.columns, is_header_present, "column (--column)")
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
            raise ValueError("Error: Must specify --to_val when using --from_val.")
        from_val = codecs.decode(args.from_val, 'unicode_escape')
        to_val = codecs.decode(args.to_val, 'unicode_escape')
        _print_verbose(args, f"Translating values in '{original}' from '{from_val}' to '{to_val}' ({'regex' if args.regex else 'literal'}).")
        try:
            translated = df.iloc[:, col].astype(str).str.replace(from_val, to_val, regex=args.regex)
        except re.error as e:
            raise ValueError(f"Error: Invalid regex '{from_val}': {e}")
    else:
        raise ValueError("Error: For replace operation, specify either a dict file (--dict_file) or both --from_val and --to_val.")
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
    cols = _parse_multiple_columns_arg(args.column, df.columns, is_header_present, "column (--column)")
    if len(cols) != 1:
        raise ValueError("Error: Sort operation requires a single column specified by --column.")
    target = df.columns[cols[0]]
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
        sys.stderr.write("Warning: '--noheader' provided. 'cleanup_header' will have no effect.\n")
        _print_verbose(args, "Skipping cleanup_header (no header).")
    else:
        original = list(df.columns)
        df.columns = [_clean_string_for_header_and_data(col) for col in df.columns]
        _print_verbose(args, f"Cleaned header. Before: {original}  After: {list(df.columns)}")
    return df

def _handle_cleanup_values(df, args, input_sep, is_header_present, row_idx_col_name):
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present, "columns (--column)")
    names = [df.columns[i] for i in indices]
    _print_verbose(args, f"Cleaning values in columns: {names}.")
    for i in indices:
        df.iloc[:, i] = df.iloc[:, i].apply(_clean_string_for_header_and_data)
    return df

def _handle_prefix_add(df, args, input_sep, is_header_present, row_idx_col_name):
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present, "columns (--column)")
    prefix = codecs.decode(args.string, 'unicode_escape')
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    names = [df.columns[i] for i in indices]
    _print_verbose(args, f"Adding prefix '{prefix}' with delimiter '{delim}' to columns: {names}.")
    for i in indices:
        df.iloc[:, i] = df.iloc[:, i].astype(str).apply(lambda x: f"{prefix}{delim}{x}")
    return df

def _handle_value_counts(df, args, input_sep, is_header_present, row_idx_col_name, state=None):
    counter = Counter() if state is None else state
    indices = _parse_multiple_columns_arg(args.column, df.columns, is_header_present, "columns (--column)")
    if not indices:
        raise ValueError("Error: No columns specified for value_counts.")
    for i in indices:
        col = df.columns[i]
        counter.update(df[col].dropna().astype(str))
    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    summary = pd.DataFrame(sorted_counts[:args.top_n], columns=['Value', 'Count'])
    sys.stdout.write(f"--- Top {args.top_n} Values ---\n")
    sys.stdout.write(summary.to_string(index=False, header=True) + '\n')
    sys.exit(0)

def _handle_strip(df, args, input_sep, is_header_present, row_idx_col_name):
    col = _parse_column_arg(args.column, df.columns, is_header_present, "column (--column)")
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
    mapping, next_id = ({}, 1) if state is None else state
    col = _parse_column_arg(args.column, df.columns, is_header_present, "column (--column)")
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
    col = _parse_column_arg(args.column, df.columns, is_header_present, "column (--column)")
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

def _handle_view(df, args, input_sep, is_header_present, row_idx_col_name, raw_first_line=None):
    import sys
    import pandas as pd

    _print_verbose(args, f"Viewing data (max rows: {args.max_rows}, max cols: {args.max_cols}).")
    pd.set_option('display.max_rows', args.max_rows)
    pd.set_option('display.max_columns', args.max_cols)
    pd.set_option('display.width', None)
    pd.set_option('display.colheader_justify', 'left')
    
    disp = df.copy()
    if getattr(args, "cleanup_numbers", False):
         disp = _format_numeric_columns(disp)
    
    if row_idx_col_name and row_idx_col_name in disp.columns:
        cols = [row_idx_col_name] + [col for col in disp.columns if col != row_idx_col_name]
        disp = disp[cols]
        _print_verbose(args, f"Moved row-index column '{row_idx_col_name}' to the front.")
    
    if getattr(args, "pretty_print", True):
         sys.stdout.write(disp.to_string(index=True, header=is_header_present) + '\n')
    else:
         disp.to_csv(
             sys.stdout,
             sep=input_sep,
             index=False,
             header=is_header_present,
             encoding='utf-8',
             quoting=csv.QUOTE_NONE,
             escapechar='\\'
         )
    
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.colheader_justify')
    
    sys.exit(0)

def _handle_cut(df, args, input_sep, is_header_present, row_idx_col_name):
    if args.list:
        if not is_header_present:
            raise ValueError("Error: File has no header (--noheader option was provided)")
        if not args.pattern:
            raise ValueError("Error: When using --list, you must provide a comma-separated list of column names or a file with column names.")
        if os.path.exists(args.pattern):
            with open(args.pattern, "r", encoding="utf-8") as f:
                col_list = [line.strip() for line in f if line.strip()]
        else:
            col_list = [x.strip() for x in args.pattern.split(',') if x.strip()]
        missing = [col for col in col_list if col not in df.columns]
        if missing:
            raise ValueError(f"Error: Columns not found: {missing}. Available columns: {list(df.columns)}")
        _print_verbose(args, f"Columns selected: {col_list}.")
        df = df[col_list]
        return df
    else:
        if not args.pattern:
            raise ValueError("Error: A pattern must be provided (either as a positional argument or with -p/--pattern) when not using --list mode.")
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

def _handle_viewheader(df, args, input_sep, is_header_present, row_idx_col_name, raw_first_line):
    import sys

    _print_verbose(args, "Displaying header names with first data row.")

    if is_header_present:
        headers = list(df.columns)
        if not df.empty:
            first_data_row = [str(x) for x in df.iloc[0].tolist()]
        else:
            first_data_row = [""] * len(headers)
    else:
        headers = raw_first_line if raw_first_line else []
        if not df.empty:
            first_data_row = [str(x) for x in df.iloc[0].tolist()]
        else:
            first_data_row = [""] * len(headers)

    if not headers:
        sys.stderr.write("No valid header found.\n")
        sys.exit(1)
    if not first_data_row:
        sys.stderr.write("No valid data row found.\n")
        sys.exit(1)

    max_header_len = max(len(header.strip()) for header in headers)
    max_value_len = 0
    for entry in first_data_row:
        entry = entry.strip()
        display_entry = (entry[:40] + '...[truncated]') if len(entry) > 40 else entry
        max_value_len = max(max_value_len, len(display_entry))

    header_idx_len = len(str(len(headers)))
    header_format = f"{{:<{header_idx_len}}} | {{:>{max_header_len}}} | {{:<{max_value_len}}}"
    sep_line = "-" * (header_idx_len + 1) + "+" + "-" * (max_header_len + 2) + "+" + "-" * (max_value_len + 2)

    print(sep_line)
    for i, (header, entry) in enumerate(zip(headers, first_data_row)):
        header = header.strip()
        entry = entry.strip()
        display_entry = (entry[:40] + '...[truncated]') if len(entry) > 40 else entry
        print(header_format.format(i + 1, header, display_entry))
    print(sep_line)

    sys.exit(0)

def _handle_row_insert(df, args, input_sep, is_header_present, row_idx_col_name):
    insert_pos = args.row_idx - 1
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

# New Plot Handlers
def _handle_ggplot(df, args, input_sep, is_header_present, row_idx_col_name):
    if hasattr(args, "figure_size") and args.figure_size:
        try:
            if ',' in args.figure_size:
                parts = args.figure_size.split(',')
            elif 'x' in args.figure_size.lower():
                parts = args.figure_size.lower().split('x')
            else:
                parts = [args.figure_size]
            fig_dims = tuple(float(x.strip()) for x in parts)
            if len(fig_dims) != 2:
                raise ValueError
        except Exception as e:
            sys.stderr.write("Error: --figure_size must be two numbers separated by a comma or x, e.g. '8,6' or '8x6'.\n")
            sys.exit(1)
    else:
        fig_dims = (8, 6)
    sys.stderr.write(f"DEBUG: Using figure size: {fig_dims[0]} x {fig_dims[1]} inches\n")
    
    if not getattr(args, "dont_replace_dots_in_colnames", False):
        col_mapping = {col: col.replace(".", "_") for col in df.columns}
        df = df.rename(columns=col_mapping)
        if args.x:
            args.x = col_mapping.get(args.x, args.x)
        if args.y:
            args.y = col_mapping.get(args.y, args.y)
        if args.fill:
            args.fill = col_mapping.get(args.fill, args.fill)
        if args.facet:
            facet_cols = [x.strip() for x in args.facet.split("~")]
            sanitized_facet = " ~ ".join(col_mapping.get(col, col) for col in facet_cols)
            args.facet = sanitized_facet
    
    if not args.melted:
        if df.shape[1] > 2:
            if not args.id_vars:
                sys.stderr.write("Error: For wide data with >2 columns, --id_vars must be provided or use --melted.\n")
                sys.exit(1)
            id_vars = [col.strip() for col in args.id_vars.split(',')]
            if args.value_vars:
                value_vars = [col.strip() for col in args.value_vars.split(',')]
            else:
                value_vars = [col for col in df.columns if col not in id_vars]
            df = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                         var_name="variable", value_name="value")
    
    if args.y:
        try:
            df[args.y] = pd.to_numeric(df[args.y], errors='coerce')
        except Exception as e:
            sys.stderr.write(f"Warning: Could not convert y values to numeric: {e}\n")
    
    from plotnine import (ggplot, aes, geom_boxplot, geom_bar, geom_point,
                          geom_histogram, geom_tile, facet_grid, labs, theme, 
                          scale_x_log10, scale_y_log10, scale_x_continuous, scale_y_continuous, guides, guide_legend)
    
    aes_dict = {}
    if args.x:
        aes_dict['x'] = args.x
    if args.y:
        aes_dict['y'] = args.y
    if args.fill:
        aes_dict['fill'] = args.fill
    p = ggplot(df, aes(**aes_dict))
    
    if args.geom == "boxplot":
        if not args.y:
            sys.stderr.write("Error: --y must be provided for boxplot.\n")
            sys.exit(1)
        p += geom_boxplot()
    elif args.geom == "bar":
        if args.y:
            p += geom_bar(stat="identity", position="dodge")
        else:
            p += geom_bar()
    elif args.geom == "point":
        if not args.y:
            sys.stderr.write("Error: --y must be provided for point plot.\n")
            sys.exit(1)
        p += geom_point()
    elif args.geom == "hist":
        if not args.x:
            sys.stderr.write("Error: --x must be provided for histogram.\n")
            sys.exit(1)
        p += geom_histogram()
    elif args.geom == "tile":
        p += geom_tile()
    elif args.geom == "pie":
        sys.stderr.write("Error: For pie charts, please use the 'plot_mpl' subcommand instead.\n")
        sys.exit(1)
    else:
        sys.stderr.write(f"Error: Unsupported geom '{args.geom}' for ggplot.\n")
        sys.exit(1)
    
    if args.facet:
        p += facet_grid(args.facet, scales='free')
    
    label_dict = {}
    if args.title:
        label_dict['title'] = args.title
    if args.xlab:
        label_dict['x'] = args.xlab
    if args.ylab:
        label_dict['y'] = args.ylab
    if label_dict:
        p += labs(**label_dict)
    
    if args.xlim:
        try:
            x_limits = [float(x.strip()) for x in args.xlim.split(',')]
            if len(x_limits) != 2:
                raise ValueError
        except:
            sys.stderr.write("Error: --xlim must be two numbers separated by a comma, e.g. '0,100'.\n")
            sys.exit(1)
        p += scale_x_continuous(limits=x_limits)
    if args.ylim:
        try:
            y_limits = [float(x.strip()) for x in args.ylim.split(',')]
            if len(y_limits) != 2:
                raise ValueError
        except:
            sys.stderr.write("Error: --ylim must be two numbers separated by a comma, e.g. '0,100'.\n")
            sys.exit(1)
        p += scale_y_continuous(limits=y_limits)
    if args.x_scale_log:
        p += scale_x_log10()
    if args.y_scale_log:
        p += scale_y_log10()
    
    p += theme_nizar()
    p += guides(fill=guide_legend(ncol=1))
    
    # choose format from extension
    fmt = "pdf" if args.output.lower().endswith(".pdf") else ("png" if args.output.lower().endswith(".png") else None)
    try:
        p.save(filename=args.output, width=fig_dims[0], height=fig_dims[1], format=fmt)
    except Exception as e:
        sys.stderr.write(f"Error saving figure: {e}\n")
        sys.exit(1)
    sys.exit(0)

def _handle_matplotlib(df, args, input_sep, is_header_present, row_idx_col_name):
    if hasattr(args, "figure_size") and args.figure_size:
        try:
            if ',' in args.figure_size:
                parts = args.figure_size.split(',')
            elif 'x' in args.figure_size.lower():
                parts = args.figure_size.lower().split('x')
            else:
                parts = [args.figure_size]
            fig_dims = tuple(float(x.strip()) for x in parts)
            if len(fig_dims) != 2:
                raise ValueError
        except Exception as e:
            sys.stderr.write("Error: --figure_size must be two numbers separated by a comma or 'x'.\n")
            sys.exit(1)
    else:
        fig_dims = (8, 6)
    sys.stderr.write(f"DEBUG: Using figure size: {fig_dims[0]} x {fig_dims[1]} inches\n")
    
    import matplotlib.pyplot as plt
    try:
        import scienceplots
        plt.style.use(['science'])
    except ImportError:
        pass
    
    mode = args.mode.lower()
    if not args.colnames:
        sys.stderr.write("Error: For matplotlib plotting, --colnames is required.\n")
        sys.exit(1)
    colnames = [col.strip() for col in args.colnames.split(',')]
    if mode == "venn2":
        if len(colnames) != 2:
            sys.stderr.write("Error: For venn2, --colnames must contain exactly 2 names.\n")
            sys.exit(1)
    elif mode == "venn3":
        if len(colnames) != 3:
            sys.stderr.write("Error: For venn3, --colnames must contain exactly 3 names.\n")
            sys.exit(1)
    
    try:
        fig, segment_table = venn_diagram(df, colnames)
    except Exception as e:
        sys.stderr.write(f"Error generating venn diagram: {e}\n")
        sys.exit(1)
    
    if args.title:
        plt.title(args.title)
    ext = os.path.splitext(args.output)[1].lower().lstrip(".") or "pdf"
    fig.savefig(args.output, format=ext, bbox_inches='tight')
    sys.exit(0)

def _handle_melt(df, args, input_sep, is_header_present, row_idx_col_name):
    id_vars = [col.strip() for col in args.id_vars.split(',')]
    if args.value_vars:
        value_vars = [col.strip() for col in args.value_vars.split(',')]
    else:
        value_vars = [col for col in df.columns if col not in id_vars]
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                   var_name=args.var_name, value_name=args.value_name)

def _handle_unmelt(df, args, input_sep, is_header_present, row_idx_col_name):
    try:
        unmelted_df = df.pivot(index=args.index, columns=args.columns, values=args.value)
        unmelted_df = unmelted_df.reset_index()
        unmelted_df.columns = [col if not isinstance(col, tuple) else '_'.join(map(str, col)).strip('_')
                                for col in unmelted_df.columns.values]
    except Exception as e:
        sys.stderr.write(f"Error in unmelt operation: {e}\n")
        sys.exit(1)
    return unmelted_df

def _handle_add_metadata(df, args, input_sep, is_header_present, row_idx_col_name):
    import sys
    import pandas as pd
    import csv
    import codecs

    if args.lowmem:
        sys.stderr.write("Error: 'join_meta' operation does not support low-memory mode (--lowmem).\n")
        sys.exit(1)

    meta_sep_raw = getattr(args, "meta_sep", None) or input_sep
    meta_sep = codecs.decode(meta_sep_raw, 'unicode_escape')

    try:
        meta_df = pd.read_csv(args.meta, sep=meta_sep, dtype=str, quoting=csv.QUOTE_NONE, escapechar='\\')
    except Exception as e:
        sys.stderr.write(f"Error reading metadata file '{args.meta}': {e}\n")
        sys.exit(1)

    _print_verbose(args, f"Metadata file head:\n{meta_df.head().to_string()}")

    df.columns = df.columns.astype(str).str.strip()
    meta_df.columns = meta_df.columns.astype(str).str.strip()

    key_input_idx = _parse_column_arg(args.key_column_in_input, df.columns, is_header_present, "key_column_in_input")
    key_input = df.columns[key_input_idx]
    key_meta_idx = _parse_column_arg(args.key_column_in_meta, meta_df.columns, True, "key_column_in_meta")
    key_meta = meta_df.columns[key_meta_idx]

    df[key_input] = df[key_input].astype(str).str.strip().str.upper()
    meta_df[key_meta] = meta_df[key_meta].astype(str).apply(remove_ansi).str.strip().str.upper()

    _print_verbose(args, f"Merging metadata: joining input column '{key_input}' with metadata column '{key_meta}'.")

    merged_df = df.merge(meta_df, how='left',
                         left_on=key_input,
                         right_on=key_meta,
                         suffixes=("", "_meta"))

    to_drop = []
    for col in merged_df.columns:
        if col.endswith("_meta"):
            base = col[:-5]
            if base in merged_df.columns:
                merged_df[base] = merged_df[col].combine_first(merged_df[base])
                to_drop.append(col)
            else:
                merged_df.rename(columns={col: base}, inplace=True)
    if to_drop:
        merged_df.drop(columns=to_drop, inplace=True)

    return merged_df

# --------------------------
# Custom Functions for Plotting
# --------------------------
def theme_nizar():
    import plotnine as p9
    return p9.theme(
        panel_background=p9.element_rect(fill="white"),
        panel_grid_major=p9.element_line(linetype='dotted', color='grey', size=0.2),
        panel_grid_minor=p9.element_line(linetype='dotted', color='grey', size=0.5),
        panel_border=p9.element_rect(color='grey', fill=None),
        legend_title=p9.element_text(weight="bold"),
        legend_direction="horizontal",
        legend_text=p9.element_text(size=6),
        axis_text_x=p9.element_text(rotation=25, hjust=1, size=6),
        axis_text_y=p9.element_text(size=8),
        legend_position='top',
        strip_text=p9.element_text(size=6)
    )

def venn_diagram(df, colnames):
    num_cols = len(colnames)
    df_num = df[colnames].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df_binary = (df_num > 0).astype(int)
    segment_data = []
    segment_counts = {}
    col_masks = [df_binary[col].astype(bool) for col in colnames]
    for i in range(1, 2**num_cols):
        binary_pattern = bin(i)[2:].zfill(num_cols)
        current_mask = pd.Series(True, index=df_binary.index)
        segment_name_parts = []
        for j, col_name in enumerate(colnames):
            if binary_pattern[j] == '1':
                current_mask &= col_masks[j]
                segment_name_parts.append(col_name)
            else:
                current_mask &= ~col_masks[j]
        count = df_binary[current_mask].shape[0]
        if count > 0:
            segment_key = binary_pattern
            segment_counts[segment_key] = count
            if len(segment_name_parts) == num_cols:
                segment_desc = " & ".join(segment_name_parts)
            elif len(segment_name_parts) == 1:
                segment_desc = segment_name_parts[0]
            else:
                segment_desc = " & ".join(segment_name_parts)
            segment_data.append({
                'Segment': segment_desc,
                'Count': count,
                'Percentage': 0
            })
    union_mask = (df_binary[colnames].sum(axis=1) > 0)
    total_elements_in_union = df_binary[union_mask].shape[0]
    for seg_dict in segment_data:
        seg_dict['Percentage'] = (seg_dict['Count'] / total_elements_in_union * 100) if total_elements_in_union > 0 else 0
    segment_table = pd.DataFrame(segment_data)
    segment_table = segment_table.sort_values(by='Count', ascending=False).reset_index(drop=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    venn_obj = None
    if num_cols == 2:
        venn_subsets = (segment_counts.get('10', 0), segment_counts.get('01', 0), segment_counts.get('11', 0))
        try:
            from matplotlib_venn import venn2
        except ImportError:
            sys.stderr.write("Error: matplotlib_venn is required for venn2 plots.\n")
            sys.exit(1)
        venn_obj = venn2(subsets=venn_subsets, set_labels=colnames, ax=ax)
        label_order_ids = ['10', '01', '11']
    elif num_cols == 3:
        venn_subsets = (segment_counts.get('100', 0), segment_counts.get('010', 0), segment_counts.get('110', 0),
                        segment_counts.get('001', 0), segment_counts.get('101', 0), segment_counts.get('011', 0),
                        segment_counts.get('111', 0))
        try:
            from matplotlib_venn import venn3
        except ImportError:
            sys.stderr.write("Error: matplotlib_venn is required for venn3 plots.\n")
            sys.exit(1)
        venn_obj = venn3(subsets=venn_subsets, set_labels=colnames, ax=ax)
        label_order_ids = ['100', '010', '110', '001', '101', '011', '111']
    for label_id in label_order_ids:
        if venn_obj.get_label_by_id(label_id):
            count = segment_counts.get(label_id, 0)
            percentage = 0.0
            temp_segment_name_parts = []
            for j, col_name in enumerate(colnames):
                if label_id[j] == '1':
                    temp_segment_name_parts.append(col_name)
            if len(temp_segment_name_parts) == num_cols:
                temp_segment_desc = " & ".join(temp_segment_name_parts)
            elif len(temp_segment_name_parts) == 1:
                temp_segment_desc = temp_segment_name_parts[0]
            else:
                temp_segment_desc = " & ".join(temp_segment_name_parts)
            matched_row = segment_table[segment_table['Segment'] == temp_segment_desc]
            if not matched_row.empty:
                percentage = matched_row['Percentage'].iloc[0]
            venn_obj.get_label_by_id(label_id).set_text(f'{count}\n({percentage:.1f}%)')
            venn_obj.get_label_by_id(label_id).set_fontsize(10)
    ax.set_title(f"Venn Diagram of Clonotype Overlap in {', '.join(colnames)}", fontsize=14)
    plt.close(fig)
    return fig, segment_table

# --------------------------
# Dispatch Table
# --------------------------
OPERATION_HANDLERS = {
    "move": _handle_move,
    "add_col": _handle_col_insert,
    "drop_col": _handle_col_drop,
    "filter": _handle_grep,
    "split_col": _handle_split,
    "join_col": _handle_join,
    "replace": _handle_tr,
    "sort": _handle_sort,
    "cleanup_header": _handle_cleanup_header,
    "cleanup_values": _handle_cleanup_values,
    "prefix": _handle_prefix_add,
    "stats": _handle_value_counts,
    "strip": _handle_strip,
    "factorize": _handle_numeric_map,
    "extract": _handle_regex_capture,
    "view": _handle_view,
    "select": _handle_cut,
    "headers": _handle_viewheader,
    "row_insert": _handle_row_insert,
    "row_drop": _handle_row_drop,
    "transpose": _handle_transpose,
    "plot": _handle_ggplot,
    "plot_mpl": _handle_matplotlib,
    "melt": _handle_melt,
    "pivot": _handle_unmelt,
    "aggregate": _handle_aggregate,
    "join_meta": _handle_add_metadata,
    "filter_cols": _handle_filter_columns,
    "pca": _handle_pca,
    "heatmap": _handle_heatmap,
    "kv_unpack": _handle_unpack_column,
    "detect_outliers": _handle_isolation_forest,
}

# --------------------------
# Input/Output Functions
# --------------------------
def _read_input_data(args, input_sep, header_param, is_header_present, use_chunked):
    raw_first_line = []
    input_stream = args.file
    # Compile regex for lines to ignore
    comment_pattern = re.compile(args.ignore_lines) if args.ignore_lines else None

    def _line_iter(fh):
        for line in fh:
            if comment_pattern and comment_pattern.match(line):
                continue
            yield line

    terminal_allow_empty = {"headers", "view", "stats", "extract"}
    if use_chunked:
        try:
            reader = pd.read_csv(_line_iter(input_stream), sep=input_sep, header=header_param, dtype=str,
                                 chunksize=CHUNK_SIZE, iterator=True)
            first_chunk = next(reader)
            if first_chunk.empty and args.operation not in terminal_allow_empty:
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
            # Filter lines according to regex and join
            content = "".join(l for l in input_stream if not (comment_pattern and comment_pattern.match(l)))
            if not content.strip() and args.operation not in terminal_allow_empty:
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

def _write_output_data(data, args, input_sep, is_header_present, header_printed):
    try:
        if args.operation in ["view", "headers", "stats"]:
            return header_printed
        if isinstance(data, pd.DataFrame):
            data.to_csv(
                sys.stdout,
                sep=input_sep,
                index=False,
                header=(is_header_present and not header_printed),
                encoding='utf-8',
                quoting=csv.QUOTE_NONE,
                escapechar='\\'
            )
            return True
        # Fallback for other iterable CSV outputs (rare)
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

def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()
    if not args.operation:
        parser.print_help()
        sys.exit(0)

    if args.lowmem and args.operation == "detect_outliers":
        sys.stderr.write("Error: Isolation Forest is not supported in --lowmem mode.\n")
        sys.exit(1)
    
    if args.operation == "join_meta" and args.lowmem:
        sys.stderr.write("Error: 'join_meta' operation does not support low-memory mode (--lowmem).\n")
        sys.exit(1)
    
    if args.noheader:
        header_param = None
        is_header_present = False
    else:
        header_param = 0
        is_header_present = True

    input_sep = codecs.decode(args.sep, 'unicode_escape')
    
    if args.lowmem and args.operation == "sort":
        sys.stderr.write("Error: 'sort' operation cannot be performed in low-memory mode (--lowmem).\n")
        sys.exit(1)
    
    use_chunked = args.lowmem and args.operation in LOWMEM_OPS

    # allow user-tunable chunksize
    global CHUNK_SIZE
    CHUNK_SIZE = args.chunksize

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
                idx = _parse_column_arg(args.row_index, cols, is_header_present, "row_index")
                row_idx_col = cols[idx]
                df_or_chunks = (item for item in [first_chunk] + list(df_or_chunks))
                _print_verbose(args, f"Row-index column: '{row_idx_col}' (index {idx}).")
            except StopIteration:
                sys.stderr.write("Error: Input is empty; cannot determine row_index column.\n")
                sys.exit(1)
        else:
            if isinstance(df_or_chunks, pd.DataFrame):
                idx = _parse_column_arg(args.row_index, df_or_chunks.columns, is_header_present, "row_index")
                row_idx_col = df_or_chunks.columns[idx]
                _print_verbose(args, f"Row-index column: '{row_idx_col}' (index {idx}).")
            else:
                sys.stderr.write("Error: Cannot determine row_index column in current mode.\n")
                sys.exit(1)

    header_printed = False
    op_state = {}
    handler = OPERATION_HANDLERS.get(args.operation)
    if handler is None:
        sys.stderr.write(f"Error: Unsupported operation '{args.operation}'.\n")
        sys.exit(1)

    if use_chunked:
        for chunk in df_or_chunks:
            if not is_header_present:
                chunk.columns = pd.Index(range(chunk.shape[1]))
            if args.operation in ["filter", "factorize"]:
                processed, op_state[args.operation] = handler(
                    chunk, args, input_sep, is_header_present, row_idx_col,
                    state=op_state.get(args.operation, {})
                )
            elif args.operation in ["view", "extract", "headers"]:
                processed = handler(
                    chunk, args, input_sep, is_header_present, row_idx_col, raw_first_line
                )
            else:
                processed = handler(
                    chunk, args, input_sep, is_header_present, row_idx_col
                )
            if row_idx_col and isinstance(processed, pd.DataFrame) and row_idx_col in processed.columns:
                cols_order = [row_idx_col] + [col for col in processed.columns if col != row_idx_col]
                processed = processed[cols_order]
            header_printed = _write_output_data(processed, args, input_sep, is_header_present, header_printed)
        if args.operation == "filter" and getattr(args, "word_file", None) and getattr(args, "list_missing_words", False):
            with open(args.word_file, 'r', encoding='utf-8') as f:
                word_list = [line.strip() for line in f if line.strip()]
            matched = op_state.get(args.operation, {}).get("matched_words", set())
            missing = set(word_list) - matched
            sys.stderr.write("Words not seen in input: (n=" + str(len(missing)) + ") " + ", ".join(sorted(missing)) + "\n")
    else:
        if args.operation in ["filter", "factorize"]:
            processed_df, state = handler(df_or_chunks, args, input_sep, is_header_present, row_idx_col)
        elif args.operation in ["view", "extract", "headers"]:
            processed_df = handler(df_or_chunks, args, input_sep, is_header_present, row_idx_col, raw_first_line)
        else:
            processed_df = handler(df_or_chunks, args, input_sep, is_header_present, row_idx_col)
        if args.operation == "filter" and getattr(args, "word_file", None) and getattr(args, "list_missing_words", False):
            with open(args.word_file, 'r', encoding='utf-8') as f:
                word_list = [line.strip() for line in f if line.strip()]
            matched = state.get("matched_words", set()) if state else set()
            missing = set(word_list) - matched
            sys.stderr.write("Words not seen in input: (n=" + str(len(missing)) + ") " + ", ".join(sorted(missing)) + "\n")
        _write_output_data(processed_df, args, input_sep, is_header_present, header_printed)

if __name__ == "__main__":
    main()
