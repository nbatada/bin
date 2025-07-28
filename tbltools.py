#!/usr/bin/env python3
import sys
import argparse
import re
import os
import pandas as pd
import codecs  # For handling escape sequences
from io import StringIO  # For piped input handling with pandas
from collections import Counter
import csv  # For CSV formatting

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
    s = s.replace('.', '_')
    s = re.sub(r'[^\w_]', '', s)
    s = re.sub(r'_{2,}', '_', s) # tr -squeeze
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
# Argument Parser
# --------------------------
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
        "--ignore-lines", default="^#",
        help="Ignore lines starting with this pattern (default: '^#')."
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
    
    # TRANSPOSE
    parser_transpose = subparsers.add_parser(
        "transpose",
        help="Transpose the input table. The first column's values are used as the header in the new output table."
    )
    
    # MOVE
    parser_move = subparsers.add_parser("move", help="Move a column. Required: --column and --dest-column.")
    parser_move.add_argument("-n", "--column", required=True, help="Source column (1-indexed or name).")
    parser_move.add_argument("-j", "--dest-column", required=True, help="Destination column (1-indexed or name).")
    
    # COL_INSERT
    parser_col_insert = subparsers.add_parser("col_insert", help="Insert a new column. Required: --column and -v/--value.")
    parser_col_insert.add_argument("-n", "--column", required=True, help="Column position (1-indexed or name) for insertion.")
    parser_col_insert.add_argument("-v", "--value", required=True, help="Value to populate the new column.")
    parser_col_insert.add_argument("--new-header", default="new_column", help="Header name for the new column (default: 'new_column').")
    
    # COL_DROP
    parser_col_drop = subparsers.add_parser("col_drop", help="Drop columns. Required: --column.")
    parser_col_drop.add_argument("-n", "--column", required=True, help="Comma-separated list of column(s) (1-indexed or names) to drop. Use 'all' to drop all columns.")
    
    # GREP
    parser_grep = subparsers.add_parser("grep", help="Filter rows. Required: --column and one of -p/--pattern, --starts-with, --ends-with, or --word-file.")
    grep_group = parser_grep.add_mutually_exclusive_group(required=True)
    parser_grep.add_argument("-n", "--column", required=True, help="Column to apply the grep filter (1-indexed or name).")
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
    
    # SUMMARIZE
    parser_summarize = subparsers.add_parser(
        "summarize",
        help="Group and summarize data using common aggregators (sum, mean, value_counts, entropy)."
    )
    parser_summarize.add_argument("--group", required=True,
                                  help="Comma-separated list of column(s) to group by.")
    parser_summarize.add_argument("--cols",
                                  help="Comma-separated list of columns to aggregate (not needed for melted input).")
    parser_summarize.add_argument("--agg", required=True,
                                  help="Aggregator to apply. Supported: 'sum', 'mean', 'value_counts', 'entropy'.")
    parser_summarize.add_argument("--normalize", action="store_true",
                                  help="For value_counts, normalize the frequencies within each group.")
    parser_summarize.add_argument("--melted", action="store_true",
                                  help="Indicate that the input is in long (melted) format.")
    
    # SPLIT
    parser_split = subparsers.add_parser("split", help="Split a column. Required: --column and -d/--delimiter.")
    parser_split.add_argument("-n", "--column", required=True, help="Column to split (1-indexed or name).")
    parser_split.add_argument("-d", "--delimiter", required=True, help="Delimiter to split the column by. Supports escape sequences.")
    parser_split.add_argument("--new-header-prefix", default="split_col", help="Prefix for the new columns (default: 'split_col').")
    
    # JOIN
    parser_join = subparsers.add_parser("join", help="Join columns. Required: --column. Optionally, --target-column specifies the destination for the joined column.")
    parser_join.add_argument("-n", "--column", required=True, help="Comma-separated list of columns (1-indexed or names) to join.")
    parser_join.add_argument("-d", "--delimiter", default="", help="Delimiter to insert between joined values (default: no delimiter). Supports escape sequences.")
    parser_join.add_argument("--new-header", default="joined_column", help="Header for the resulting joined column (default: 'joined_column').")
    parser_join.add_argument("-j", "--target-column", help="Target column (1-indexed or name) where the joined column will be placed.")
    
    # TR (Translate)
    parser_tr = subparsers.add_parser("tr", help="Translate values. Required: --column and either -d/--dict-file or --from-val with --to-val.")
    parser_tr.add_argument("-n", "--column", required=True, help="Column to translate (1-indexed or name).")
    tr_group = parser_tr.add_mutually_exclusive_group(required=True)
    tr_group.add_argument("-d", "--dict-file", help="Path to a two-column file (key<sep>value) for mapping. Uses the main --sep as separator.")
    tr_group.add_argument("--from-val", help="Value to translate from (for single translation). Supports escape sequences.")
    parser_tr.add_argument("--to-val", help="Value to translate to (for single translation). Supports escape sequences.")
    parser_tr.add_argument("--regex", action="store_true", help="Treat --from-val as a regex pattern (default is literal).")
    parser_tr.add_argument("--new-header", default="_translated", help="Suffix or new header for the translated column (default: '_translated').")
    parser_tr.add_argument("--in-place", action="store_true", help="Replace the original column with the translated values.")
    
    # SORT
    parser_sort = subparsers.add_parser("sort", help="Sort table. Required: --column. (Not compatible with lowmem mode.)")
    parser_sort.add_argument("-n", "--column", required=True, help="Column to sort by (1-indexed or name). Only one column should be specified.")
    parser_sort.add_argument("--desc", action="store_true", help="Sort in descending order (default is ascending).")
    parser_sort.add_argument("-p", "--pattern", help="Regex pattern to extract a numeric key from the column.")
    parser_sort.add_argument("--suffix-map", help="Comma-separated key:value pairs for suffix mapping (used with --pattern).")
    parser_sort.add_argument("--expand-scientific", action="store_true", 
                             help="Use a default mapping for scientific suffixes and override any supplied --suffix-map.")
    parser_sort.add_argument("-ps", "--pattern-string", help="Optional additional regex pattern for a secondary string sort key.")
    
    # CLEANUP HEADER
    parser_cleanup_header = subparsers.add_parser("cleanup_header", help="Clean header names (lowercase, remove special characters, replace spaces with underscores).")
    
    # CLEANUP VALUES
    parser_cleanup_values = subparsers.add_parser("cleanup_values", help="Clean values in specified columns. Required: --column.")
    parser_cleanup_values.add_argument("-n", "--column", required=True, help="Comma-separated list of columns (1-indexed or names) to clean. Use 'all' to clean every column.")
    
    # PREFIX ADD
    parser_prefix_add = subparsers.add_parser("prefix_add", help="Add a prefix to column values. Required: --column and -v/--string.")
    parser_prefix_add.add_argument("-n", "--column", required=True, help="Comma-separated list of columns (1-indexed or names) to prepend with a prefix. Use 'all' for every column.")
    parser_prefix_add.add_argument("-v", "--string", required=True, help="The prefix string to add. Supports escape sequences.")
    parser_prefix_add.add_argument("-d", "--delimiter", default="", help="Delimiter to insert between the prefix and the original value (default: none). Supports escape sequences.")
    
    # VALUE COUNTS
    parser_value_counts = subparsers.add_parser("value_counts", help="Count top occurring values. Required: --column.")
    parser_value_counts.add_argument("-T", "--top-n", type=int, default=5, help="Number of top values to display (default: 5).")
    parser_value_counts.add_argument("-n", "--column", required=True, help="Comma-separated list of columns (1-indexed or names) to count. Use 'all' for every column.")
    
    # STRIP
    parser_strip = subparsers.add_parser("strip", help="Remove a regex pattern from column values. Required: --column and -p/--pattern.")
    parser_strip.add_argument("-n", "--column", required=True, help="Column to strip (1-indexed or name).")
    parser_strip.add_argument("-p", "--pattern", required=True, help="Regex pattern to remove from the column values.")
    parser_strip.add_argument("--new-header", default="_stripped", help="Suffix or new header for the column after stripping (default: '_stripped').")
    parser_strip.add_argument("--in-place", action="store_true", help="Modify the column in place instead of creating a new column.")
    
    # NUMERIC MAP
    parser_numeric_map = subparsers.add_parser("numeric_map", help="Map unique string values to numbers. Required: --column.")
    parser_numeric_map.add_argument("-n", "--column", required=True, help="Column (1-indexed or name) whose unique values are to be mapped to numbers.")
    parser_numeric_map.add_argument("--new-header", help="Header for the new numeric mapping column (default: 'numeric_map_of_ORIGINAL_COLUMN_NAME').")
    
    # REGEX CAPTURE
    parser_regex_capture = subparsers.add_parser("regex_capture", help="Capture substrings using a regex capturing group. Required: --column and -p/--pattern.")
    parser_regex_capture.add_argument("-n", "--column", required=True, help="Column on which to apply the regex (1-indexed or name).")
    parser_regex_capture.add_argument("-p", "--pattern", required=True, help="Regex pattern with at least one capturing group (e.g., '_(S[0-9]+)\\.' ).")
    parser_regex_capture.add_argument("--new-header", default="_captured", help="Suffix or new header for the captured column (default: '_captured').")
    
    # VIEW
    # In the view subcommand section:
    parser_view = subparsers.add_parser("view", help="Display the data in a formatted table.")
    parser_view.add_argument("--max-rows", type=int, default=20, help="Maximum number of rows to display (default: 20).")
    parser_view.add_argument("--max-cols", type=int, default=None, help="Maximum number of columns to display (default: all columns).")
    parser_view.add_argument("--precision-long", action="store_true",
                         help="Display numeric columns with full precision (do not round to 2 decimal places).")

    # CUT (Enhanced)
    parser_cut = subparsers.add_parser("cut", help="Cut/select columns. Provide either a regex pattern or, if --list is specified, a list of column names.")
    parser_cut.add_argument("pattern", nargs="?", default=None,
                        help=("Either a regex pattern for matching column names "
                              "or, when --list is specified, a comma-separated list "
                              "of column names (or a file containing column names)."))
    parser_cut.add_argument("--regex", action="store_true",
                        help="Interpret the supplied pattern as a regex (default for --list is literal matching).")
    parser_cut.add_argument("--list", action="store_true",
                        help="Interpret the pattern as a comma-separated list of column names for selection in the given order.")
    
    # VIEWHEADER
    parser_viewheader = subparsers.add_parser("viewheader", help="Display header names and positions.")
    
    # ROW_INSERT
    parser_row_insert = subparsers.add_parser("row_insert", help="Insert a new row at a specified 1-indexed position. Use -i 0 to insert at the header.")
    parser_row_insert.add_argument("-i", "--row-idx", type=int, default=0, help="Row position for insertion (1-indexed, 0 for header insertion).")
    parser_row_insert.add_argument("-v", "--values", help="Comma-separated list of values for the new row. Supports escape sequences.")
    
    # ROW_DROP
    parser_row_drop = subparsers.add_parser("row_drop", help="Delete row(s) at a specified 1-indexed position. Use -i 0 to drop the header row.")
    parser_row_drop.add_argument("-i", "--row-idx", type=int, required=True, help="Row position to drop (1-indexed, 0 drops the header).")
    
    # New Subparsers for Plotting
    # ggplot subcommand using Plotnine
    parser_ggplot = subparsers.add_parser("ggplot", help="Generate a ggplot using Plotnine and save to a PDF file.")
    parser_ggplot.add_argument("--geom", required=True, choices=["boxplot", "bar", "point", "hist", "tile", "pie"],
                               help="Type of plot to generate. (Note: 'pie' is not supported in ggplot mode; use the matplotlib subcommand instead.)")
    parser_ggplot.add_argument("--x", required=True, help="Column name for x aesthetic.")
    parser_ggplot.add_argument("--y", help="Column name for y aesthetic (required for boxplot and point).")
    parser_ggplot.add_argument("--fill", help="Column name for fill aesthetic (optional).")
    parser_ggplot.add_argument("--facet", help="Facet formula, e.g., 'col1 ~ col2'.")
    parser_ggplot.add_argument("--title", help="Plot title.")
    parser_ggplot.add_argument("--xlab", help="Label for x-axis.")
    parser_ggplot.add_argument("--ylab", help="Label for y-axis.")
    parser_ggplot.add_argument("--xlim", help="x-axis limits as 'min,max'.")
    parser_ggplot.add_argument("--ylim", help="y-axis limits as 'min,max'.")
    parser_ggplot.add_argument("--x_scale_log", action="store_true", help="Use logarithmic scale for x-axis.")
    parser_ggplot.add_argument("--y_scale_log", action="store_true", help="Use logarithmic scale for y-axis.")
    parser_ggplot.add_argument("-o", "--output", required=True, help="Output PDF filename.")
    parser_ggplot.add_argument("--melted", action="store_true", help="Indicate that input data is already melted. (If not provided, data will be auto-detected.)")
    parser_ggplot.add_argument("--id_vars", help="Comma-separated list of columns to use as id_vars when melting (required if not wide).")
    parser_ggplot.add_argument("--value_vars", help="Comma-separated list of columns to melt. If not provided, all columns not in id_vars are melted.")
    parser_ggplot.add_argument("--figure_size", help="Set figure size as width,height in inches (default: 8,6).")
    
    # matplotlib subcommand for venn diagrams
    parser_mpl = subparsers.add_parser("matplotlib", help="Generate a matplotlib-based plot (supports Venn diagrams) and save to a PDF file.")
    parser_mpl.add_argument("--mode", required=True, choices=["venn2", "venn3"],
                           help="Plot mode for matplotlib: 'venn2' or 'venn3'.")
    parser_mpl.add_argument("--colnames", required=True,
                           help="Comma-separated list of header names to use. (2 names for venn2; 3 names for venn3)")
    parser_mpl.add_argument("--title", help="Plot title.")
    parser_mpl.add_argument("--figure_size", help="Set figure size as width,height in inches (default: 8,6).")
    parser_mpl.add_argument("-o", "--output", required=True, help="Output PDF filename.")
    
    # Melt
    parser_melt = subparsers.add_parser("melt", help="Melt the input table into a long format.")
    parser_melt.add_argument("--id_vars", required=True, help="Comma-separated list of column names to use as id_vars.")
    parser_melt.add_argument("--value_vars", help="Comma-separated list of column names to be melted. If not provided, all columns not in id_vars are used.")
    parser_melt.add_argument("--var_name", default="variable", help="Name for the new variable column (default: 'variable').")
    parser_melt.add_argument("--value_name", default="value", help="Name for the new value column (default: 'value').")
    
    # Unmelt
    parser_unmelt = subparsers.add_parser("unmelt", help="Pivot the melted table back to wide format.")
    parser_unmelt.add_argument("--index", required=True, help="Column name to use as the index (row identifiers).")
    parser_unmelt.add_argument("--columns", required=True, help="Column name that contains variable names (to become new columns).")
    parser_unmelt.add_argument("--value", required=True, help="Column name that contains the values.")
    
    # --------------------------
    # NEW: ADD_METADATA
    # --------------------------
    parser_add_metadata = subparsers.add_parser("add_metadata", help="Merge a metadata file into the main table based on key columns.")
    parser_add_metadata.add_argument("--meta", required=True, help="Path to the metadata file (CSV).")
    parser_add_metadata.add_argument("--key_column_in_input", required=True, help="Key column (name or 1-indexed) in the input file to join on.")
    parser_add_metadata.add_argument("--key_column_in_meta", required=True, help="Key column (name or 1-indexed) in the metadata file to join on.")
    
    return parser

# --------------------------
# Operation Handler Functions
# --------------------------
##==
def _handle_summarize(df, args, input_sep, is_header_present, row_idx_col_name):
    """
    Summarize the input table based on group(s), and apply an aggregator function
    to one or more columns. Supported aggregator functions are:
      - For numeric data: "sum" and "mean"
      - For categorical data: "value_counts" and "entropy"
    Optionally, if the --normalize flag is provided with value_counts, the frequencies
    are normalized within each group.
    
    For wide-format input (default), the user must supply:
      --group  (e.g., "sample_id")
      --cols   (e.g., "reads,quality"), but you can also use --cols "*" to use all non-grouping columns.
      
    For melted input (long format), the table must contain at least "variable" and "value".
    Optionally, you can supply --group so that additional id columns are used.
    
    Returns:
       A pandas DataFrame with the summarized output.
    """
    import sys
    import pandas as pd
    from scipy.stats import entropy as calculate_entropy

    # -------------------------------
    # M E L T E D (Long) Format Mode
    # -------------------------------
    if getattr(args, "melted", False):
        # Check for required "variable" and "value" columns:
        required_cols = {"variable", "value"}
        if not required_cols.issubset(set(df.columns)):
            sys.stderr.write("Error: When using --melted, input must have 'variable' and 'value' columns.\n")
            sys.exit(1)
        # If additional grouping is desired, use the --group argument.
        group_cols = []
        if getattr(args, "group", None):
            group_cols = [col.strip() for col in args.group.split(",") if col.strip()]
        # Always include "variable" to keep track of which field is being aggregated.
        if "variable" not in group_cols:
            group_cols.append("variable")
        agg_func = args.agg.lower()
        summary_rows = []
        grouped = df.groupby(group_cols)
        for grp_keys, grp_df in grouped:
            if not isinstance(grp_keys, tuple):
                grp_keys = (grp_keys,)
            group_dict = dict(zip(group_cols, grp_keys))
            # Apply aggregation on the "value" column:
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

    # -------------------------------
    # W I D E  (Normal) Format Mode
    # -------------------------------
    else:
        # In wide mode, both --group and --cols are required.
        if not getattr(args, "group", None):
            sys.stderr.write("Error: In wide format, you must supply --group to specify grouping variable(s).\n")
            sys.exit(1)
        if not getattr(args, "cols", None):
            sys.stderr.write("Error: In wide format, you must supply --cols with the comma-separated list of columns to aggregate (or '*' for all non-group columns).\n")
            sys.exit(1)
        group_cols = [col.strip() for col in args.group.split(",") if col.strip()]
        # If user supplies '*' for --cols, use all columns except those in group_cols.
        if args.cols.strip() == "*" or args.cols.strip() == "all":
            agg_cols = [col for col in df.columns if col not in group_cols]
        else:
            agg_cols = [col.strip() for col in args.cols.split(",") if col.strip()]

        agg_func = args.agg.lower()
        if agg_func in ["sum", "mean"]:
            # For numeric aggregations, filter out any column that cannot be converted to numeric.
            valid_agg_cols = []
            for col in agg_cols:
                # Attempt to convert the entire column to numeric.
                series_numeric = pd.to_numeric(df[col], errors="coerce")
                # If the entire column is NaN after conversion (i.e. not numeric), then skip.
                if series_numeric.notna().sum() > 0:
                    valid_agg_cols.append(col)
                else:
                    # Optionally: print a warning if in verbose mode.
                    if getattr(args, "verbose", False):
                        sys.stderr.write(f"Warning: Skipping non-numeric column '{col}' for aggregator '{agg_func}'.\n")
            if not valid_agg_cols:
                sys.stderr.write("Error: No numeric columns found for aggregation.\n")
                sys.exit(1)
            # Convert valid columns explicitly to numeric.
            for col in valid_agg_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            grouped = df.groupby(group_cols)
            summary_df = grouped[valid_agg_cols].agg(agg_func).reset_index()
            # Rename columns to indicate the applied function.
            rename_dict = {col: f"{agg_func}_{col}" for col in valid_agg_cols}
            summary_df.rename(columns=rename_dict, inplace=True)
            return summary_df

        # For categorical aggregations: "value_counts" and "entropy"
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

##==

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
    to_idx = _parse_column_arg(args.dest_column, df.columns, is_header_present, "destination column (--dest-column)")
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
        target = _parse_column_arg(args.target_column, df.columns, is_header_present, "target column (--target-column)")
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
        counter.update(df[col].astype(str))
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
    mapping, next_id = ({} , 1) if state is None else state
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
    from pandas.api.types import is_numeric_dtype

    _print_verbose(args, f"Viewing data (max rows: {args.max_rows}, max cols: {args.max_cols}).")
    pd.set_option('display.max_rows', args.max_rows)
    pd.set_option('display.max_columns', args.max_cols)
    pd.set_option('display.width', None)
    pd.set_option('display.colheader_justify', 'left')
    
    # Work on a copy of the DataFrame for display formatting.
    disp = df.copy()

    # Unless --precision-long is specified, round numeric columns.
    if not getattr(args, "precision_long", False):
        for col in disp.columns:
            if is_numeric_dtype(disp[col]):
                # Ensure the column is numeric and then round to 2 decimal places.
                disp[col] = pd.to_numeric(disp[col], errors='coerce')
                disp[col] = disp[col].round(2)
                # If all non-null values are effectively integers, convert column to integer type.
                if disp[col].dropna().apply(lambda x: abs(x - round(x)) < 1e-8).all():
                    disp[col] = disp[col].astype("Int64")
    
    # If a row-index column was specified, move it to the beginning.
    if row_idx_col_name and row_idx_col_name in disp.columns:
        cols = [row_idx_col_name] + [col for col in disp.columns if col != row_idx_col_name]
        disp = disp[cols]
        _print_verbose(args, f"Moved row-index column '{row_idx_col_name}' to the front.")
    
    # Use a float_format callback if we want 2-digit precision (the default).
    float_fmt = (lambda x: f"{x:.2f}") if not getattr(args, "precision_long", False) else None
    
    # Print the DataFrame to stdout.
    sys.stdout.write(disp.to_string(index=True, header=is_header_present, float_format=float_fmt) + '\n')
    
    # Reset display options.
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
    _print_verbose(args, "Listing header names with 1-indexed positions.")
    out_lines = []
    if df.empty and df.columns.size:
        for i, col in enumerate(df.columns):
            out_lines.append(f"{i+1}\t{col}")
    elif not df.empty:
        cols = list(df.columns)
        if not is_header_present and raw_first_line:
            for i, val in enumerate(raw_first_line):
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

# New Plot Handlers
#==
def _handle_ggplot(df, args, input_sep, is_header_present, row_idx_col_name):
    # Determine figure size.
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
    
    # --- New Feature: Sanitize Column Names ---
    # By default, replace all '.' in column names with '_' so that plotnine does not misinterpret them.
    # Users can disable this by providing --dont_replace_dots_in_colnames on the command line.
    if not getattr(args, "dont_replace_dots_in_colnames", False):
        # Build a mapping from original names to their sanitized (dot-replaced) version.
        col_mapping = {col: col.replace(".", "_") for col in df.columns}
        df = df.rename(columns=col_mapping)
        # Update plotting aesthetics if they refer to a column that was modified.
        if args.x:
            args.x = col_mapping.get(args.x, args.x)
        if args.y:
            args.y = col_mapping.get(args.y, args.y)
        if args.fill:
            args.fill = col_mapping.get(args.fill, args.fill)
        if args.facet:
            # Assume a simple facet formula like "col1 ~ col2" (whitespaces tolerated)
            facet_cols = [x.strip() for x in args.facet.split("~")]
            sanitized_facet = " ~ ".join(col_mapping.get(col, col) for col in facet_cols)
            args.facet = sanitized_facet
    # ------------------------------------------------

    # For ggplot, if data is not already melted, melt if necessary.
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
    
    # Convert y column to numeric if possible.
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
        sys.stderr.write("Error: For pie charts, please use the 'matplotlib' subcommand instead.\n")
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
    
    try:
        p.save(filename=args.output, width=fig_dims[0], height=fig_dims[1], format="pdf")
    except Exception as e:
        sys.stderr.write(f"Error saving PDF: {e}\n")
        sys.exit(1)
    sys.exit(0)

#==
def _handle_matplotlib(df, args, input_sep, is_header_present, row_idx_col_name):
    # Determine figure size.
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
    
    mode = args.mode.lower()  # Expected to be 'venn2' or 'venn3'
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
    
    # Use the custom venn_diagram function.
    try:
        fig, segment_table = venn_diagram(df, colnames)
    except Exception as e:
        sys.stderr.write(f"Error generating venn diagram: {e}\n")
        sys.exit(1)
    
    if args.title:
        plt.title(args.title)
    fig.savefig(args.output, format="pdf", bbox_inches='tight')
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

# --------------------------
# NEW: ADD_METADATA Handler
# --------------------------
def _handle_add_metadata(df, args, input_sep, is_header_present, row_idx_col_name):
    # This operation does not support low-memory mode because we need to join the full DataFrame.
    if args.lowmem:
        sys.stderr.write("Error: 'add_metadata' operation does not support low-memory mode (--lowmem).\n")
        sys.exit(1)
    try:
        meta_df = pd.read_csv(args.meta, sep=input_sep, dtype=str)
    except Exception as e:
        sys.stderr.write(f"Error reading metadata file '{args.meta}': {e}\n")
        sys.exit(1)
    # Parse key column for input.
    key_input_idx = _parse_column_arg(args.key_column_in_input, df.columns, is_header_present, "key_column_in_input")
    key_input = df.columns[key_input_idx]
    # For metadata, assume it has a header.
    key_meta_idx = _parse_column_arg(args.key_column_in_meta, meta_df.columns, True, "key_column_in_meta")
    key_meta = meta_df.columns[key_meta_idx]
    
    _print_verbose(args, f"Merging metadata: joining input column '{key_input}' with metadata column '{key_meta}'.")
    merged_df = df.merge(meta_df, how='left', left_on=key_input, right_on=key_meta)
    return merged_df

# --------------------------
# Custom Functions for Plotting
# --------------------------
import plotnine as p9
def theme_nizar():
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
    """
    Build a Venn diagram from the input DataFrame using the supplied column names.
    It converts the specified columns to binary indicators (True if value > 0) and then
    computes the counts for each segment.
    """
    num_cols = len(colnames)
    df_binary = (df[colnames].astype(float) > 0).astype(int)
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
    # Create the figure and axis.
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
    "transpose": _handle_transpose,
    "ggplot": _handle_ggplot,
    "matplotlib": _handle_matplotlib,
    "melt": _handle_melt,
    "unmelt": _handle_unmelt,
    "summarize": _handle_summarize,
    "add_metadata": _handle_add_metadata  # NEW operation mapping
}

# --------------------------
# Input/Output Functions
# --------------------------
def _read_input_data(args, input_sep, header_param, is_header_present, use_chunked):
    raw_first_line = []
    input_stream = args.file
    comment_char = None
    if args.ignore_lines:
        if args.ignore_lines.startswith('^'):
            comment_char = args.ignore_lines[1:]
        else:
            comment_char = args.ignore_lines
    if use_chunked:
        try:
            reader = pd.read_csv(input_stream, sep=input_sep, header=header_param, dtype=str,
                                 comment=comment_char,
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
            df = pd.read_csv(csv_io, sep=input_sep, header=header_param, dtype=str, comment=comment_char)
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
        if args.operation in ["view", "viewheader", "value_counts"]:
            return header_printed
        if isinstance(data, pd.DataFrame):
            data.to_csv(
                sys.stdout,
                sep=input_sep,
                index=False,
                header=is_header_present,
                encoding='utf-8',
                quoting=csv.QUOTE_NONE,
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

def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()
    if not args.operation:
        parser.print_help()
        sys.exit(0)
    
    # Disallow low-memory mode for operations that require merging the entire table.
    if args.operation == "add_metadata" and args.lowmem:
        sys.stderr.write("Error: 'add_metadata' operation does not support low-memory mode (--lowmem).\n")
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
    
    lowmem_ops = ["value_counts", "numeric_map", "grep", "tr", "strip", "prefix_add", "cleanup_values", "regex_capture"]
    use_chunked = args.lowmem and args.operation in lowmem_ops

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
            sys.stderr.write("Words not seen in input: " + ", ".join(sorted(matched)) + "\n")
        _write_output_data(processed_df, args, input_sep, is_header_present, header_printed)

if __name__ == "__main__":
    main()
