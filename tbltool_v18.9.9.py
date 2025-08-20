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
from functools import wraps
from scipy.stats import entropy as calculate_entropy
import difflib

__version__ = "8.19.9"


# ---
# VERSION=8.19.9 (WORKS)
# This is a major refactoring release that incorporates numerous bug fixes, UX enhancements, and robustness improvements based on a detailed code review.
# Summary of changes:
#   - CLI: Added --version flag. Error messages now suggest corrections (e.g., "Did you mean...?"). Help text is improved and no longer hardcoded.
#   - Correctness: Fixed decorator mismatches for column operations. `col_capture_regex` now handles multiple capture groups. `col_encode` no longer mutates original data. `row_add` works on empty files.
#   - Features: Added flags to `tbl_join_meta` (--how), `row_query` (--expr), `col_split` (--keep), `col_join` (--keep), and `view` (--max-col-width).
#   - I/O: Exposed more CSV parsing options (e.g., --quotechar, --na-values) for more robust file reading.
#   - Fixed a bug where commands that accept stdin (e.g., 'view') would incorrectly show help instead of executing when receiving piped data.
#   - Re-implemented the main help text formatter to use pandas for a clean, table-like, justified layout for command categories.
#   - Renamed `col_stats` to `col_frequency` for clarity.
#   - Changed the output of `col_frequency` from a long-format table to a wide-format table.
#   - The new format displays the top N values as a list of strings: "Value (Count, Frequency%)" in each column.
# --
# VERSION=8.19.8
# This update fixes a critical bug that caused a crash with piped input and revamps the main help menu for clarity.
# Summary of changes:
#   - Fixed a "name 'kwargs' is not defined" crash affecting all commands when receiving piped data.
#   - Re-implemented the main help menu to display commands and their descriptions in a two-column, samtools-like format for improved readability.
# ---

# --------------------------
# Custom Argument Parser
# --------------------------
class GroupedHelpFormatter(argparse.HelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = 'Usage: '
        return super(GroupedHelpFormatter, self).add_usage(usage, actions, groups, prefix)

    def _format_action(self, action):
        parts = super(GroupedHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n" + parts
        return parts

    def start_section(self, heading):
        super(GroupedHelpFormatter, self).start_section(heading.capitalize())

    def _format_actions_usage(self, actions, groups):
        return super(GroupedHelpFormatter, self)._format_actions_usage(actions, groups)

class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(CustomArgumentParser, self).__init__(*args, **kwargs)
        self.formatter_class = GroupedHelpFormatter
    def format_help(self):
        # This initial part captures the standard usage and description
        formatter = self._get_formatter()
        if self.description:
            formatter.add_text(self.description)
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        
        subparser_actions = [
            action for action in self._actions if isinstance(action, argparse._SubParsersAction)
        ]
        # Fallback for subcommand help (e.g., ./script view -h)
        if not subparser_actions:
            for action_group in self._action_groups:
                if action_group._group_actions:
                    formatter.start_section(action_group.title)
                    formatter.add_arguments(action_group._group_actions)
                    formatter.end_section()
            return formatter.format_help()

        # Start building the final help string, beginning with the usage info
        output_parts = [formatter.format_help().split('\n\n')[0].strip()]

        categories = {
            'Table commands': [], 'Column commands': [], 'Row commands': [], 'Utilities': []
        }
        all_parsers = subparser_actions[0].choices
        
        # Group all commands into their respective categories
        for command, parser in all_parsers.items():
            item = {'Command': command, 'Description': parser.description or ''}
            if command.startswith('tbl_'): categories['Table commands'].append(item)
            elif command.startswith('col_'): categories['Column commands'].append(item)
            elif command.startswith('row_'): categories['Row commands'].append(item)
            else: categories['Utilities'].append(item)

        # Manually format each category with correct justification
        for title, items in categories.items():
            if items:
                output_parts.append(f"\n{title.capitalize()}:")
                output_parts.append("-"*15)
                # Calculate max length for right-justification
                max_len = max(len(item['Command']) for item in items)
                
                # Format each line with right-justified command and left-justified description
                sorted_items = sorted(items, key=lambda x: x['Command'])
                for item in sorted_items:
                    # command_part = f"{item['Command']:>{max_len}}" # right aligns
                    command_part = f"{item['Command']:<{max_len}}"
                    description_part = item['Description']
                    output_parts.append(f"  {command_part}  {description_part}")
        
        output_parts.append(f"\nFor command-specific help, run: {self.prog} <command> -h")
        return "\n".join(output_parts)


    def error(self, message):
        if 'invalid choice' in message:
            m = re.search(r"invalid choice: '([^']*)'", message)
            cmd = m.group(1) if m else "unknown"
            choices = []
            for a in self._actions:
                if isinstance(a, argparse._SubParsersAction):
                    choices = list(a.choices.keys())
                    break
            suggestion = difflib.get_close_matches(cmd, choices, n=1)
            use_color = sys.stderr.isatty() and os.getenv("NO_COLOR") is None
            red = "\033[91m" if use_color else ""
            reset = "\033[0m" if use_color else ""
            line = "-" * (len(cmd) + 20)
            hint = f"\nDid you mean: {suggestion[0]!r}?\n" if suggestion else "\n"
            self.exit(2, f"\n{line}\n{red}Error: Command '{cmd}' not found.{reset}\n{line}{hint}")
        super().error(message)

# --------------------------
# Utility Functions
# --------------------------
def _clean_string(s):
    if not isinstance(s, str): return s
    s = s.lower().replace(' ', '_').replace('.', '_')
    s = re.sub(r'[^\w_]', '', s)
    return re.sub(r'_{2,}', '_', s)

def _get_unique_header(candidate, existing_columns):
    if candidate not in existing_columns: return candidate
    i = 1
    while f"{candidate}_{i}" in existing_columns: i += 1
    return f"{candidate}_{i}"

def _parse_single_col(value, df_columns, is_header_present):
    try:
        col_idx = int(value) - 1
        if not (0 <= col_idx < len(df_columns)): raise IndexError(f"Column index '{value}' is out of bounds.")
        return df_columns[col_idx]
    except ValueError:
        if not is_header_present: raise ValueError("Cannot use column names when --noheader is specified.")
        if value not in df_columns: raise ValueError(f"Column '{value}' not found. Available: {list(df_columns)}")
        return value

def _parse_multi_cols(values, df_columns, is_header_present):
    stripped_values = [val.strip() for val in values.split(',') if val.strip()]
    if "all" in [v.lower() for v in stripped_values]: return list(df_columns)
    return [_parse_single_col(val, df_columns, is_header_present) for val in stripped_values]

def process_columns(required=True, multi=False):
    def decorator(func):
        @wraps(func)
        def wrapper(df, args, **kwargs):
            if not hasattr(args, 'columns') or args.columns is None:
                if required: raise ValueError(f"Operation '{args.operation}' requires a -c/--columns argument.")
                return func(df, args, column_names=[], **kwargs)
            is_header = kwargs.get('is_header_present', True)
            if multi: column_names = _parse_multi_cols(args.columns, df.columns, is_header)
            else: column_names = [_parse_single_col(args.columns, df.columns, is_header)]
            if required and not column_names: raise ValueError("No valid columns were specified.")
            return func(df, args, column_names=column_names, **kwargs)
        return wrapper
    return decorator

def _pretty_print_df(df, is_header_present, show_index_header=False, max_col_width=40, truncate=True):
    if df.empty:
        if is_header_present: print(" | ".join(df.columns))
        print("(empty table)")
        return
    df_str = df.astype(str)
    col_widths = {}
    for col in df_str.columns:
        header_len = len(str(col)) if is_header_present else 0
        max_data_len = df_str[col].str.len().max()
        if pd.isna(max_data_len): max_data_len = 0
        col_widths[col] = min(max(header_len, int(max_data_len)), max_col_width)
    if show_index_header:
        print(" | ".join([f"{i+1:>{col_widths[col]}}" for i, col in enumerate(df.columns)]))
    if is_header_present:
        print(" | ".join([f"{str(col):>{col_widths[col]}}" for col in df.columns]))
        print("-+-".join(["-" * col_widths[col] for col in df.columns]))
    for _, row in df_str.iterrows():
        row_values = []
        for col in df.columns:
            val = row[col]
            if truncate and len(val) > col_widths[col]: val = val[:col_widths[col] - 3] + "..."
            row_values.append(f"{val:>{col_widths[col]}}")
        print(" | ".join(row_values))

# --------------------------
# Operation Handlers
# --------------------------
def _handle_tbl_transpose(df, args, is_header_present, **kwargs):
    if is_header_present: df = pd.concat([pd.DataFrame([df.columns], columns=df.columns), df], ignore_index=True)
    if df.shape[1] < 2: raise ValueError("Transpose requires at least 2 columns.")
    transposed_df = df.iloc[:, 1:].T
    transposed_df.columns = df.iloc[:, 0].tolist()
    return transposed_df

@process_columns(multi=True)
def _handle_tbl_aggregate(df, args, column_names, **kwargs):
    group_cols = _parse_multi_cols(args.group, df.columns, kwargs['is_header_present'])
    agg_func = args.agg.lower()
    if agg_func in ["sum", "mean"]:
        numeric_cols = df[column_names].select_dtypes(include='number').columns.tolist()
        if not numeric_cols: raise ValueError("No numeric columns found for aggregation.")
        summary = df.groupby(group_cols)[numeric_cols].agg(agg_func).reset_index()
        summary.rename(columns={col: f"{agg_func}_{col}" for col in numeric_cols}, inplace=True)
        return summary
    summary_rows = []
    for grp_keys, grp_df in df.groupby(group_cols):
        grp_keys = (grp_keys,) if not isinstance(grp_keys, tuple) else grp_keys
        group_dict = dict(zip(group_cols, grp_keys))
        for col in column_names:
            if agg_func == "value_counts":
                vc = grp_df[col].value_counts(normalize=args.normalize).reset_index()
                vc.columns = ["value", "count"]
                for _, row in vc.iterrows(): summary_rows.append({**group_dict, "aggregated_column": col, "value": row["value"], "count": row["count"]})
            elif agg_func == "entropy":
                summary_rows.append({**group_dict, "aggregated_column": col, "entropy": calculate_entropy(grp_df[col].value_counts())})
    return pd.DataFrame(summary_rows)

@process_columns()
def _handle_tbl_sort(df, args, column_names, **kwargs):
    col_name = column_names[0]
    df_copy = df.copy()
    try:
        if args.pattern:
            df_copy['_sort_key'] = df_copy[col_name].astype(str).str.extract(f'({args.pattern})').iloc[:, 0].astype(float)
            df_copy = df_copy.sort_values(by='_sort_key', ascending=not args.desc, kind='stable').drop(columns=['_sort_key'])
        else:
            df_copy = df_copy.sort_values(by=col_name, ascending=not args.desc, kind='stable')
    except re.error as e: raise ValueError(f"Invalid regex in sort pattern: {e}")
    return df_copy

def _handle_tbl_clean_header(df, args, is_header_present, **kwargs):
    if is_header_present: df.columns = [_clean_string(col) for col in df.columns]
    return df

def _handle_tbl_melt(df, args, **kwargs):
    id_vars = _parse_multi_cols(args.id_vars, df.columns, kwargs['is_header_present'])
    value_vars = _parse_multi_cols(args.value_vars, df.columns, kwargs['is_header_present']) if args.value_vars else None
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars)

def _handle_tbl_unmelt(df, args, **kwargs):
    return df.pivot(index=args.index, columns=args.columns, values=args.value).reset_index()

def _handle_tbl_join_meta(df, args, **kwargs):
    meta_df = pd.read_csv(args.meta, sep=kwargs['input_sep'])
    key_input = _parse_single_col(args.key_column_in_input, df.columns, kwargs['is_header_present'])
    key_meta = _parse_single_col(args.key_column_in_meta, meta_df.columns, True)
    return pd.merge(df, meta_df, how=args.how, left_on=key_input, right_on=key_meta, suffixes=tuple(args.suffixes.split(',')))

@process_columns(required=False)
def _handle_row_query(df, args, column_names, **kwargs):
    if args.expr:
        try: return df.query(args.expr)
        except Exception as e: raise ValueError(f"Invalid query expression: {e}")
    col_name = column_names[0]
    series = pd.to_numeric(df[col_name], errors='coerce')
    op_map = { 'lt': series < args.value, 'gt': series > args.value, 'eq': series == args.value, 'ne': series != args.value, 'le': series <= args.value, 'ge': series >= args.value }
    return df[op_map[args.operator]]


@process_columns()
def _handle_col_move(df, args, column_names, **kwargs):
    col_name = column_names[0]
    dest_col = _parse_single_col(args.dest_column, df.columns, kwargs['is_header_present'])
    dest_idx = df.columns.get_loc(dest_col)
    data = df.pop(col_name)
    df.insert(dest_idx, col_name, data)
    return df

@process_columns()
def _handle_col_add(df, args, column_names, **kwargs):
    pos_col = column_names[0]
    pos_idx = df.columns.get_loc(pos_col)
    value = codecs.decode(args.value, 'unicode_escape')
    new_header = _get_unique_header(args.new_header, df.columns)
    df.insert(pos_idx, new_header, value)
    return df

@process_columns(multi=True)
def _handle_col_drop(df, args, column_names, **kwargs):
    return df.drop(columns=column_names)

@process_columns(multi=True)
def _handle_col_split(df, args, column_names, **kwargs):
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    n = args.maxsplit if args.maxsplit is not None else -1
    for col_name in column_names:
        col_idx = df.columns.get_loc(col_name)
        if args.regex:
            try: split_df = df[col_name].astype(str).str.split(delim, n=n, expand=True, regex=True).fillna('')
            except re.error as e: raise ValueError(f"Invalid regex for split: {e}")
        else: split_df = df[col_name].astype(str).str.split(delim, n=n, expand=True).fillna('')
        new_headers = [_get_unique_header(f"{col_name}_{i+1}", df.columns.tolist() + list(split_df.columns)) for i in range(split_df.shape[1])]
        split_df.columns = new_headers
        df = pd.concat([df.iloc[:, :col_idx+1], split_df, df.iloc[:, col_idx+1:]], axis=1)
        if not args.keep: df = df.drop(columns=[col_name])
    return df

@process_columns(multi=True)
def _handle_col_join(df, args, column_names, **kwargs):
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    joined_series = df[column_names].astype(str).agg(delim.join, axis=1)
    new_header = _get_unique_header(args.new_header, df.columns)
    min_idx = min(df.columns.get_loc(c) for c in column_names)
    df_copy = df if args.keep else df.drop(columns=column_names)
    df_copy.insert(min_idx, new_header, joined_series)
    return df_copy

@process_columns()
def _handle_col_replace(df, args, column_names, **kwargs):
    col_name = column_names[0]
    if args.dict_file:
        mapping = dict(line.strip().split('\t', 1) for line in open(args.dict_file) if '\t' in line)
        translated = df[col_name].astype(str).map(mapping).fillna(df[col_name])
    else:
        from_val = codecs.decode(args.from_val, 'unicode_escape')
        to_val = codecs.decode(args.to_val, 'unicode_escape')
        translated = df[col_name].astype(str).str.replace(from_val, to_val, regex=args.regex)
    if args.in_place: df[col_name] = translated
    else: df.insert(df.columns.get_loc(col_name) + 1, _get_unique_header(f"{col_name}_translated", df.columns), translated)
    return df

@process_columns(multi=True)
def _handle_col_clean_values(df, args, column_names, **kwargs):
    for col in column_names: df[col] = df[col].apply(_clean_string)
    return df

@process_columns(multi=True)
def _handle_col_add_prefix(df, args, column_names, **kwargs):
    prefix = codecs.decode(args.string, 'unicode_escape')
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    for col in column_names: df[col] = prefix + delim + df[col].astype(str)
    return df

@process_columns(multi=True)
def _handle_col_add_suffix(df, args, column_names, **kwargs):
    suffix = codecs.decode(args.string, 'unicode_escape')
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    for col in column_names: df[col] = df[col].astype(str) + delim + suffix
    return df

@process_columns(multi=True)
def _handle_col_frequency(df, args, column_names, **kwargs):
    """Calculates and displays the top N most frequent values in a wide format."""
    results_dict = {}
    total_rows = len(df)

    # Ensure there's data to process
    if total_rows == 0:
        return pd.DataFrame({col: [] for col in column_names})

    for col in column_names:
        # Get top N value counts
        value_counts = df[col].value_counts().nlargest(args.top_n)
        
        col_results = []
        # Format each value into the "Value (Count, Frequency%)" string
        for value, count in value_counts.items():
            frequency = (count / total_rows) * 100
            formatted_string = f"{value} ({count}, {frequency:.2f}%)"
            col_results.append(formatted_string)
            
        # Pad the list with empty strings if there are fewer unique values than top_n
        while len(col_results) < args.top_n:
            col_results.append("")
            
        results_dict[col] = col_results
        
    return pd.DataFrame(results_dict)

@process_columns()
def _handle_col_strip_chars(df, args, column_names, **kwargs):
    col_name = column_names[0]
    try: stripped = df[col_name].astype(str).str.replace(args.pattern, '', regex=True)
    except re.error as e: raise ValueError(f"Invalid regex for strip: {e}")
    if args.in_place: df[col_name] = stripped
    else: df.insert(df.columns.get_loc(col_name) + 1, _get_unique_header(f"{col_name}_stripped", df.columns), stripped)
    return df

@process_columns()
def _handle_col_encode(df, args, column_names, **kwargs):
    col_name = column_names[0]
    codes, _ = pd.factorize(df[col_name])
    encoded = pd.Series(codes, index=df.index)
    new_header = args.new_header or _get_unique_header(f"{col_name}_encoded", df.columns)
    df.insert(df.columns.get_loc(col_name) + 1, new_header, encoded)
    return df

@process_columns(multi=True)
def _handle_col_capture_regex(df, args, column_names, **kwargs):
    try: re.compile(args.pattern)
    except re.error as e: raise ValueError(f"Invalid regex pattern: {e}")
    for col in column_names:
        extracted = df[col].astype(str).str.extractall(args.pattern)
        if extracted.empty: continue
        if isinstance(extracted, pd.DataFrame) and extracted.shape[1] > 1:
            base = f"{col}_capture"
            extracted.columns = [_get_unique_header(f"{base}_{i+1}", df.columns) for i in range(extracted.shape[1])]
        else:
            extracted.columns = [_get_unique_header(f"{col}_captured", df.columns)]
        
        extracted = extracted.reset_index().set_index('level_0')
        df = df.join(extracted.drop(columns='match', errors='ignore'))
    return df

@process_columns(required=False, multi=True)
def _handle_col_select(df, args, column_names, **kwargs):
    """Selects columns by name and/or by data type."""
    type_selected_cols = []

    # Check if a type-based selector was used
    if args.only_numeric or args.only_integer or args.only_string:
        df_copy = df.copy()
        for col in df_copy.columns:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col])
            except (ValueError, TypeError):
                pass  # Leave non-numeric columns as is

        if args.only_numeric:
            type_selected_cols = df_copy.select_dtypes(include='number').columns.tolist()
        elif args.only_integer:
            type_selected_cols = df_copy.select_dtypes(include='integer').columns.tolist()
        elif args.only_string:
            type_selected_cols = df_copy.select_dtypes(include=['object', 'string']).columns.tolist()

    # Ensure at least one selection criterion was provided
    if not column_names and not type_selected_cols:
        raise ValueError("col_select requires at least one column name (-c) or a type selector (--only_numeric, etc.).")

    # Combine name-based and type-based selections, ensuring no duplicates
    final_cols = column_names + [col for col in type_selected_cols if col not in column_names]
    
    return df[final_cols]

@process_columns()
def _handle_row_grep(df, args, column_names, **kwargs):
    col_name = column_names[0]
    series = df[col_name].astype(str)
    try:
        if args.word_file:
            with open(args.word_file, 'r', encoding='utf-8') as f: words = [line.strip() for line in f if line.strip()]
            pattern = "|".join(map(re.escape, words))
            mask = series.str.contains(pattern, regex=True, na=False)
        else:
            mask = series.str.contains(args.pattern, regex=True, na=False)
    except re.error as e: raise ValueError(f"Invalid regex for grep: {e}")
    return df[~mask] if args.invert else df[mask]

def _handle_row_add(df, args, **kwargs):
    values = [codecs.decode(v.strip(), 'unicode_escape') for v in args.values.split(',')]
    if df.empty: df = pd.DataFrame(columns=[f"col_{i+1}" for i in range(len(values))])
    if len(values) != df.shape[1]: raise ValueError(f"Number of values ({len(values)}) must match number of columns ({df.shape[1]}).")
    new_row = pd.DataFrame([values], columns=df.columns)
    insert_pos = max(0, args.row_idx - 1)
    return pd.concat([df.iloc[:insert_pos], new_row, df.iloc[insert_pos:]]).reset_index(drop=True)

def _handle_row_drop(df, args, **kwargs):
    drop_pos = args.row_idx - 1
    if not (0 <= drop_pos < len(df)): raise IndexError(f"Row index {args.row_idx} is out of bounds.")
    return df.drop(df.index[drop_pos]).reset_index(drop=True)

# --------------------------
# Terminal/Utility Handlers (May exit)
# --------------------------
def _handle_view(df, args, is_header_present, **kwargs):
    if args.header_view:
        if df.empty: sys.stdout.write("(empty table with no columns)\n"); sys.exit(0)
        headers = df.columns
        if not is_header_present: headers = [f'c{i+1}' for i in range(df.shape[1])]
        first_row = [str(x) for x in df.iloc[0]]
        max_header_len = max(len(h) for h in headers)
        output = [f"{i+1:<3} | {header:>{max_header_len}} | {(first_row[i][:37] + '...') if len(first_row[i]) > 40 else first_row[i]}" for i, header in enumerate(headers)]
        sys.stdout.write("\n".join(output) + "\n")
        sys.exit(0)
    
    _pretty_print_df(df.head(args.max_rows), is_header_present, args.show_index, args.max_col_width, not args.no_truncate)
    sys.exit(0)

# --------------------------
# Main Setup
# --------------------------
def _setup_arg_parser():
    parser = CustomArgumentParser(description="A command-line tool for manipulating tabular data.", add_help=False)
    io_opts = parser.add_argument_group("Input/Output Options")
    io_opts.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    io_opts.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    io_opts.add_argument("-f", "--file", type=argparse.FileType('r'), default=sys.stdin, help="Input file (default: stdin).")
    io_opts.add_argument("-s", "--sep", default="\t", help="Field separator for input and output.")
    io_opts.add_argument("--noheader", action="store_true", help="Input has no header.")
    io_opts.add_argument("-r", "--row-index", help="Column to use as row identifier.")
    io_opts.add_argument("--quotechar", default='"', help="Character used to quote fields.")
    io_opts.add_argument("--escapechar", default=None, help="Character used to escape separators in fields.")
    io_opts.add_argument("--doublequote", action='store_true', help="Whether to interpret two consecutive quotechars as a single quotechar.")
    io_opts.add_argument("--na-values", help="Comma-separated list of strings to recognize as NA/NaN.")
    io_opts.add_argument("--na-rep", default="", help="String representation for NA/NaN values in output.")
    io_opts.add_argument("--on-bad-lines", choices=['error', 'warn', 'skip'], default='error', help="Action for lines with too many fields.")
    
    subparsers = parser.add_subparsers(dest="operation", title="Available Operations")

    # Table Operations
    p_t_transpose = subparsers.add_parser("tbl_transpose", help="Transpose the table.", description="Transpose the table.")
    p_t_agg = subparsers.add_parser("tbl_aggregate", help="Group and aggregate data.", description="Group and aggregate data.")
    p_t_agg.add_argument("-c", "--columns", required=True, help="Columns to aggregate.")
    p_t_agg.add_argument("--group", required=True, help="Columns to group by.")
    p_t_agg.add_argument("--agg", required=True, choices=['sum', 'mean', 'value_counts', 'entropy'])
    p_t_agg.add_argument("--normalize", action="store_true")
    p_t_sort = subparsers.add_parser("tbl_sort", help="Sort table by a column.", description="Sort table by a column.")
    p_t_sort.add_argument("-c", "--columns", required=True, help="Column to sort by.")
    p_t_sort.add_argument("--desc", action="store_true")
    p_t_sort.add_argument("-p", "--pattern", help="Regex to extract numeric key for sorting.")
    p_t_clean = subparsers.add_parser("tbl_clean_header", help="Clean all header names.", description="Clean all header names.")
    p_t_melt = subparsers.add_parser("tbl_melt", help="Melt table to long format.", description="Melt table to long format.")
    p_t_melt.add_argument("--id_vars", required=True)
    p_t_melt.add_argument("--value_vars")
    p_t_unmelt = subparsers.add_parser("tbl_unmelt", help="Unmelt table to wide format.", description="Unmelt table to wide format.")
    p_t_unmelt.add_argument("--index", required=True)
    p_t_unmelt.add_argument("--columns", required=True)
    p_t_unmelt.add_argument("--value", required=True)
    p_t_join = subparsers.add_parser("tbl_join_meta", help="Merge a metadata file.", description="Merge a metadata file.")
    p_t_join.add_argument("--meta", required=True)
    p_t_join.add_argument("--key_column_in_input", required=True)
    p_t_join.add_argument("--key_column_in_meta", required=True)
    p_t_join.add_argument("--how", choices=['left', 'right', 'outer', 'inner'], default='left')
    p_t_join.add_argument("--suffixes", default="_x,_y")

    # Row Operations
    p_r_query = subparsers.add_parser("row_query", help="Filter rows on a condition.", description="Filter rows on a condition.")
    q_group = p_r_query.add_mutually_exclusive_group(required=True)
    q_group.add_argument("-c", "--columns", help="Column to query for simple operations.")
    q_group.add_argument("--expr", help="Pandas query expression string (e.g., 'col1 > 5 & col2 == \"foo\"').")
    p_r_query.add_argument("--operator", choices=['lt', 'gt', 'eq', 'ne', 'le', 'ge'])
    p_r_query.add_argument("-v", "--value", type=float)
    p_r_grep = subparsers.add_parser("row_grep", help="Filter rows by pattern.", description="Filter rows by pattern.")
    p_r_grep.add_argument("-c", "--columns", required=True, help="Column to search in.")
    grep_group = p_r_grep.add_mutually_exclusive_group(required=True)
    grep_group.add_argument("-p", "--pattern")
    grep_group.add_argument("--word-file")
    p_r_grep.add_argument("-v", "--invert", action="store_true")
    p_r_add = subparsers.add_parser("row_add", help="Insert a new row.", description="Insert a new row.")
    p_r_add.add_argument("-i", "--row-idx", type=int, default=1)
    p_r_add.add_argument("-v", "--values", required=True)
    p_r_drop = subparsers.add_parser("row_drop", help="Delete a row by position.", description="Delete a row by position.")
    p_r_drop.add_argument("-i", "--row-idx", type=int, required=True)

    # Column Operations
    # Find this parser in the "Column Operations" section
    p_c_select = subparsers.add_parser("col_select", help="Select columns by name or data type.", description="Select columns by name or data type.")
    p_c_select.add_argument("-c", "--columns", help="Comma-separated column names to select.")
    type_group = p_c_select.add_mutually_exclusive_group()
    type_group.add_argument("--only_numeric", action="store_true", help="Select all numeric columns.")
    type_group.add_argument("--only_integer", action="store_true", help="Select all integer columns.")
    type_group.add_argument("--only_string", action="store_true", help="Select all string columns.")

    p_c_move = subparsers.add_parser("col_move", help="Move a column.", description="Move a column.")
    p_c_move.add_argument("-c", "--columns", required=True, help="Source column.")
    p_c_move.add_argument("-j", "--dest-column", required=True)
    p_c_add = subparsers.add_parser("col_add", help="Insert a new column.", description="Insert a new column.")
    p_c_add.add_argument("-c", "--columns", required=True, help="Position for insertion.")
    p_c_add.add_argument("-v", "--value", required=True)
    p_c_add.add_argument("--new-header", default="new_column")
    p_c_drop = subparsers.add_parser("col_drop", help="Drop column(s).", description="Drop column(s).")
    p_c_drop.add_argument("-c", "--columns", required=True, help="Columns to drop.")
    p_c_split = subparsers.add_parser("col_split", help="Split a column.", description="Split column(s) by a delimiter.")
    p_c_split.add_argument("-c", "--columns", required=True, help="Column(s) to split.")
    p_c_split.add_argument("-d", "--delimiter", required=True)
    p_c_split.add_argument("--maxsplit", type=int, help="Maximum number of splits.")
    p_c_split.add_argument("--regex", action="store_true")
    p_c_split.add_argument("--keep", action="store_true", help="Keep original column after splitting.")
    p_c_join = subparsers.add_parser("col_join", help="Join columns.", description="Join columns with a delimiter.")
    p_c_join.add_argument("-c", "--columns", required=True)
    p_c_join.add_argument("-d", "--delimiter", default="")
    p_c_join.add_argument("--new-header", default="joined_column")
    p_c_join.add_argument("--keep", action="store_true", help="Keep original columns after joining.")
    p_c_replace = subparsers.add_parser("col_replace", help="Replace values in a column.", description="Replace values in a column.")
    p_c_replace.add_argument("-c", "--columns", required=True)
    tr_group = p_c_replace.add_mutually_exclusive_group(required=True)
    tr_group.add_argument("-d", "--dict-file")
    tr_group.add_argument("--from-val")
    p_c_replace.add_argument("--to-val")
    p_c_replace.add_argument("--regex", action="store_true")
    p_c_replace.add_argument("--in-place", action="store_true")

    p_c_frequency = subparsers.add_parser("col_frequency", help="Get top N frequent values for column(s).", description="Get top N frequent values for column(s).")
    p_c_frequency.add_argument("-c", "--columns", required=True, help="Column(s) to analyze ('all' for all columns).")
    p_c_frequency.add_argument("-T", "--top-n", type=int, default=10, help="Number of top frequent values to show.")

    p_c_strip = subparsers.add_parser("col_strip_chars", help="Remove pattern from values.", description="Remove pattern from values.")
    p_c_strip.add_argument("-c", "--columns", required=True)
    p_c_strip.add_argument("-p", "--pattern", required=True)
    p_c_strip.add_argument("--in-place", action="store_true")
    p_c_encode = subparsers.add_parser("col_encode", help="Encode categorical column to numbers.", description="Encode categorical column to numbers.")
    p_c_encode.add_argument("-c", "--columns", required=True)
    p_c_encode.add_argument("--new-header")
    p_c_capture = subparsers.add_parser("col_capture_regex", help="Capture regex group(s).", description="Capture regex group(s).")
    p_c_capture.add_argument("-c", "--columns", required=True)
    p_c_capture.add_argument("-p", "--pattern", required=True)

    
    # Utility Operations
    p_u_view = subparsers.add_parser("view", help="Display formatted table.", description="Display formatted table.")
    p_u_view.add_argument("--max-rows", type=int, default=20)
    p_u_view.add_argument("--max-col-width", type=int, default=40)
    p_u_view.add_argument("--no-truncate", action="store_true", help="Disable column truncation.")
    p_u_view.add_argument("-H", "--header-view", action="store_true")
    p_u_view.add_argument("--show-index", action="store_true")
    
    return parser, subparsers

OPERATION_HANDLERS = { "tbl_transpose": _handle_tbl_transpose, "tbl_aggregate": _handle_tbl_aggregate, "tbl_sort": _handle_tbl_sort, "tbl_clean_header": _handle_tbl_clean_header, "tbl_melt": _handle_tbl_melt, "tbl_unmelt": _handle_tbl_unmelt, "tbl_join_meta": _handle_tbl_join_meta, "row_query": _handle_row_query, "col_move": _handle_col_move, "col_add": _handle_col_add, "col_drop": _handle_col_drop, "col_split": _handle_col_split, "col_join": _handle_col_join, "col_replace": _handle_col_replace, "col_clean_values": _handle_col_clean_values, "col_add_prefix": _handle_col_add_prefix, "col_add_suffix": _handle_col_add_suffix, "col_frequency": _handle_col_frequency, "col_strip_chars": _handle_col_strip_chars, "col_encode": _handle_col_encode, "col_capture_regex": _handle_col_capture_regex, "col_select": _handle_col_select, "row_grep": _handle_row_grep, "row_add": _handle_row_add, "row_drop": _handle_row_drop, "view": _handle_view }

def _read_input_data(args, sep, header):
    try:
        content = args.file.read()
        if not content.strip(): return pd.DataFrame()
        na_values = args.na_values.split(',') if args.na_values else None
        return pd.read_csv(StringIO(content), sep=sep, header=header, dtype=str, engine='python', quotechar=args.quotechar, escapechar=args.escapechar, doublequote=args.doublequote, na_values=na_values, on_bad_lines=args.on_bad_lines)
    except Exception as e: sys.stderr.write(f"Error reading input data: {e}\n"); sys.exit(1)

def _write_output(df, sep, is_header, na_rep):
    if df is not None and not df.empty:
        try: df.to_csv(sys.stdout, sep=sep, index=False, header=is_header, quoting=csv.QUOTE_MINIMAL, na_rep=na_rep)
        except BrokenPipeError: pass
        except Exception as e: sys.stderr.write(f"Error writing output: {e}\n"); sys.exit(1)

def main():
    parser, subparsers = _setup_arg_parser()
    if len(sys.argv) == 2 and sys.argv[1] in subparsers.choices and sys.stdin.isatty():
        subparsers.choices[sys.argv[1]].print_help()
        sys.exit(0)

    args = parser.parse_args()
    if not hasattr(args, 'operation') or not args.operation: parser.print_help(); sys.exit(0)
    
    is_header_present = not args.noheader
    header_param = 0 if is_header_present else None
    input_sep = codecs.decode(args.sep, 'unicode_escape')
    
    df = _read_input_data(args, input_sep, header_param)
    if df.empty and args.operation not in ["view", "row_add"]: sys.exit(0)
    if not is_header_present and not df.empty: df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
    
    row_idx_col_name = _parse_single_col(args.row_index, df.columns, is_header_present) if args.row_index and not df.empty else None
    handler = OPERATION_HANDLERS.get(args.operation)
    if not handler: sys.stderr.write(f"Error: Unsupported operation '{args.operation}'.\n"); sys.exit(1)
    
    try:
        handler_kwargs = { "is_header_present": is_header_present, "input_sep": input_sep, "row_idx_col_name": row_idx_col_name }
        processed_df = handler(df, args, **handler_kwargs)
        _write_output(processed_df, input_sep, is_header_present, args.na_rep)
    except (ValueError, IndexError, FileNotFoundError, KeyError) as e: sys.stderr.write(f"Error: {e}\n"); sys.exit(1)
    except Exception as e: sys.stderr.write(f"An unexpected error occurred: {type(e).__name__} - {e}\n"); sys.exit(1)

if __name__ == "__main__":
    main()
