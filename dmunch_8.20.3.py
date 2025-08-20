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
import difflib

# Try to import new dependencies and provide helpful error messages
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
except ImportError:
    sys.stderr.write("Error: scikit-learn is not installed. Please run 'pip install scikit-learn'.\n")
    sys.exit(1)

try:
    import statsmodels.formula.api as smf
except ImportError:
    sys.stderr.write("Error: statsmodels is not installed. Please run 'pip install statsmodels'.\n")
    sys.exit(1)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


__version__ = "8.20.3"


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
        formatter = self._get_formatter()
        if self.description:
            formatter.add_text(self.description)
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        
        subparser_actions = [
            action for action in self._actions if isinstance(action, argparse._SubParsersAction)
        ]
        if not subparser_actions:
            for action_group in self._action_groups:
                if action_group._group_actions:
                    formatter.start_section(action_group.title)
                    formatter.add_arguments(action_group._group_actions)
                    formatter.end_section()
            return formatter.format_help()

        output_parts = [formatter.format_help().split('\n\n')[0].strip()]

        categories = {
            'Table Commands': [], 'Column Commands': [], 'Row Commands': [], 'Analysis & Statistics': [], 'Utilities': []
        }
        all_parsers = subparser_actions[0].choices
        
        # Simplified categorization logic using the new stat_ prefix
        for command, parser in all_parsers.items():
            item = {'Command': command, 'Description': parser.description or ''}
            if command.startswith('stat_'):
                categories['Analysis & Statistics'].append(item)
            elif command.startswith('tbl_'):
                categories['Table Commands'].append(item)
            elif command.startswith('col_'):
                categories['Column Commands'].append(item)
            elif command.startswith('row_'):
                categories['Row Commands'].append(item)
            elif command in ['view']:
                categories['Utilities'].append(item)

        all_items = [item for items in categories.values() for item in items]
        global_max_len = max(len(item['Command']) for item in all_items) if all_items else 0

        for title, items in categories.items():
            if items:
                output_parts.append(f"\n{title.capitalize()}:")
                output_parts.append("-" * len(title))
                
                sorted_items = sorted(items, key=lambda x: x['Command'])
                for item in sorted_items:
                    command_part = f"{item['Command']:<{global_max_len}}"
                    description_part = item['Description']
                    output_parts.append(f"  {command_part}  {description_part}")
        
        output_parts.append(f"\nFor command-specific help, run: {self.prog} <command> -h")
        return "\n".join(output_parts)
    
    def _exit_with_error(self, message, hint=""):
        use_color = sys.stderr.isatty() and os.getenv("NO_COLOR") is None
        red = "\033[91m" if use_color else ""
        reset = "\033[0m" if use_color else ""
        line_content = message.splitlines()[0]
        line = "-" * (len(line_content) + 8)

        formatted_message = f"\n{line}\n{red}Error: {message}{reset}\n{line}{hint}"
        self.exit(2, formatted_message)

    def error(self, message):
        if 'invalid choice' in message:
            m = re.search(r"invalid choice: '([^']*)'", message)
            cmd = m.group(1) if m else "unknown"
            choices = [k for a in self._actions if isinstance(a, argparse._SubParsersAction) for k in a.choices.keys()]
            suggestion = difflib.get_close_matches(cmd, choices, n=1)
            hint = f"\nDid you mean: {suggestion[0]!r}?\n" if suggestion else "\n"
            self._exit_with_error(f"Command '{cmd}' not found.", hint=hint)
        elif 'unrecognized arguments' in message:
            m = re.search(r"unrecognized arguments: (.*)", message)
            args = m.group(1) if m else ""
            self._exit_with_error(f"Unrecognized arguments: {args}")
        elif 'the following arguments are required' in message:
            m = re.search(r"required: (.*)", message)
            args = m.group(1) if m else "Unknown"
            self._exit_with_error(f"Missing required arguments: {args}")
        else:
            self._exit_with_error(message.capitalize())
        
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
    except (ValueError, TypeError):
        if not is_header_present: raise ValueError("Cannot use column names when --noheader is specified.")
        if value not in df_columns: raise ValueError(f"Column '{value}' not found. Available: {list(df_columns)}")
        return value

def _parse_multi_cols(values, df_columns, is_header_present):
    if not values: return []
    stripped_values = [val.strip() for val in values.split(',') if val.strip()]
    if "all" in [v.lower() for v in stripped_values]: return list(df_columns)
    return [_parse_single_col(val, df_columns, is_header_present) for val in stripped_values]

def _get_numeric_cols(df, specified_cols):
    numeric_cols = df[specified_cols].select_dtypes(include='number').columns.tolist()
    if not numeric_cols: raise ValueError("Operation requires at least one numeric column.")
    return numeric_cols

def process_columns(required=True, multi=False):
    def decorator(func):
        @wraps(func)
        def wrapper(df, args, **kwargs):
            if not hasattr(args, 'columns') or args.columns is None:
                if required: raise ValueError(f"Operation '{args.operation}' requires a -c/--columns argument.")
                return func(df, args, column_names=[], **kwargs)
            is_header = kwargs.get('is_header_present', True)
            if multi: column_names = _parse_multi_cols(args.columns, df.columns, is_header)
            else: column_names = [_parse_single_col(args.columns, df.columns, is_header)] if args.columns else []
            if required and not column_names: raise ValueError("No valid columns were specified.")
            return func(df, args, column_names=column_names, **kwargs)
        return wrapper
    return decorator

def _pretty_print_df(df, is_header_present, show_index_header=False, max_col_width=40, truncate=True, precision=None):
    if df.empty:
        if is_header_present and list(df.columns): print(" | ".join(df.columns))
        print("(empty table)")
        return
    
    df_to_print = df.copy()

    # --- NEW LOGIC TO ROUND NUMBERS ---
    if precision is not None:
        numeric_cols = df_to_print.select_dtypes(include='number').columns
        df_to_print[numeric_cols] = df_to_print[numeric_cols].round(precision)
    # --- END NEW LOGIC ---

    df_str = df_to_print.astype(str)
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
# Operation Handlers (A-Z)
# --------------------------

@process_columns()
def _handle_col_add(df, args, column_names, **kwargs):
    pos_col = column_names[0]
    pos_idx = df.columns.get_loc(pos_col)
    value = codecs.decode(args.value, 'unicode_escape')
    new_header = _get_unique_header(args.new_header, df.columns)
    df.insert(pos_idx, new_header, value)
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
def _handle_col_bin(df, args, column_names, **kwargs):
    df_copy = df.copy()
    for col in column_names:
        series = pd.to_numeric(df_copy[col], errors='coerce')
        new_col_name = _get_unique_header(f"{col}_binned", df_copy.columns)
        labels = args.labels.split(',') if args.labels else False
        if args.qcut:
            df_copy[new_col_name] = pd.qcut(series, q=args.qcut, labels=labels, duplicates='drop')
        elif args.breaks:
            breaks = [float(b) for b in args.breaks.split(',')]
            df_copy[new_col_name] = pd.cut(series, bins=breaks, labels=labels, right=False)
        else:
            df_copy[new_col_name] = pd.cut(series, bins=args.bins, labels=labels)
        if not args.keep: df_copy = df_copy.drop(columns=[col])
    return df_copy

@process_columns(multi=True)
def _handle_col_capture_regex(df, args, column_names, **kwargs):
    try: re.compile(args.pattern)
    except re.error as e: raise ValueError(f"Invalid regex pattern: {e}")
    for col in column_names:
        extracted = df[col].astype(str).str.extractall(args.pattern) if args.explode else df[col].astype(str).str.extract(args.pattern)
        if extracted.empty: continue
        if extracted.shape[1] > 1:
            base = f"{col}_capture"
            extracted.columns = [_get_unique_header(f"{base}_{i+1}", df.columns) for i in range(extracted.shape[1])]
        else: extracted.columns = [_get_unique_header(f"{col}_captured", df.columns)]
        if args.explode: df = df.drop(columns=extracted.columns, errors='ignore').join(extracted.droplevel(1))
        else: df = pd.concat([df, extracted], axis=1)
    return df

@process_columns(multi=True)
def _handle_col_clean_values(df, args, column_names, **kwargs):
    for col in column_names: df[col] = df[col].apply(_clean_string)
    return df

@process_columns(multi=True)
def _handle_col_drop(df, args, column_names, **kwargs):
    return df.drop(columns=column_names)

@process_columns()
def _handle_col_encode(df, args, column_names, **kwargs):
    col_name = column_names[0]
    codes, _ = pd.factorize(df[col_name])
    new_header = args.new_header or _get_unique_header(f"{col_name}_encoded", df.columns)
    df.insert(df.columns.get_loc(col_name) + 1, new_header, pd.Series(codes, index=df.index))
    return df

@process_columns(required=False)
def _handle_col_eval(df, args, column_names, **kwargs):
    df_copy = df.copy()
    new_header = _get_unique_header(args.new_header, df.columns)
    is_header_present = kwargs.get('is_header_present', True)

    def eval_on_group(sub_df):
        return sub_df.eval(args.expr, engine='python')

    if args.group:
        group_cols = _parse_multi_cols(args.group, df.columns, is_header_present)
        df_copy[new_header] = df_copy.groupby(group_cols).apply(eval_on_group).reset_index(level=group_cols, drop=True)
    else:
        df_copy[new_header] = df.eval(args.expr, engine='python')
    return df_copy

@process_columns(multi=True)
def _handle_col_fill_na(df, args, column_names, **kwargs):
    df_copy = df.copy()
    for col in column_names:
        if args.value is not None:
            fill_val = pd.to_numeric(args.value, errors='ignore')
            df_copy[col] = df_copy[col].fillna(fill_val)
        elif args.method in ['mean', 'median']:
            series = pd.to_numeric(df_copy[col], errors='coerce')
            fill_val = series.mean() if args.method == 'mean' else series.median()
            df_copy[col] = series.fillna(fill_val)
        else: # ffill, bfill
            df_copy[col] = df_copy[col].fillna(method=args.method)
    return df_copy

@process_columns(multi=True)
def _handle_col_join(df, args, column_names, **kwargs):
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    joined_series = df[column_names].astype(str).agg(delim.join, axis=1)
    new_header = _get_unique_header(args.new_header, df.columns)
    min_idx = min(df.columns.get_loc(c) for c in column_names)
    df_to_modify = df.drop(columns=column_names) if not args.keep else df.copy()
    df_to_modify.insert(min_idx, new_header, joined_series)
    return df_to_modify

@process_columns()
def _handle_col_move(df, args, column_names, **kwargs):
    col_to_move = column_names[0]
    dest_col_name = _parse_single_col(args.dest_column, df.columns, kwargs.get('is_header_present', True))
    if col_to_move == dest_col_name: return df
    cols = df.columns.tolist()
    if col_to_move not in cols or dest_col_name not in cols: raise ValueError("Source or destination column not found.")
    cols.remove(col_to_move)
    dest_idx = cols.index(dest_col_name)
    insert_pos = dest_idx + 1 if args.position == 'after' else dest_idx
    cols.insert(insert_pos, col_to_move)
    return df[cols]
    
@process_columns(multi=False)
def _handle_col_rename(df, args, column_names, **kwargs):
    try:
        rename_map = dict(item.split(':', 1) for item in args.map.split(','))
    except ValueError:
        raise ValueError("Invalid format for --map. Use 'old_name1:new_name1,old_name2:new_name2'.")
    return df.rename(columns=rename_map)

@process_columns()
def _handle_col_replace(df, args, column_names, **kwargs):
    if args.from_val is not None and args.to_val is None: raise ValueError("--to-val is required with --from-val.")
    col_name = column_names[0]
    if args.dict_file:
        mapping = dict(line.strip().split('\t', 1) for line in open(args.dict_file) if '\t' in line)
        translated = df[col_name].astype(str).map(mapping).fillna(df[col_name])
    else:
        from_val, to_val = codecs.decode(args.from_val, 'unicode_escape'), codecs.decode(args.to_val, 'unicode_escape')
        translated = df[col_name].astype(str).str.replace(from_val, to_val, regex=args.regex)
    if args.in_place: df[col_name] = translated
    else: df.insert(df.columns.get_loc(col_name) + 1, _get_unique_header(f"{col_name}_translated", df.columns), translated)
    return df

@process_columns(multi=True)
def _handle_col_scale(df, args, column_names, **kwargs):
    df_copy = df.copy()
    is_header_present = kwargs.get('is_header_present', True)
    
    def scale_group(sub_df, method, cols):
        sub_df_copy = sub_df.copy()
        numeric_cols = _get_numeric_cols(sub_df_copy, cols)
        if method == 'zscore':
            sub_df_copy[numeric_cols] = StandardScaler().fit_transform(sub_df_copy[numeric_cols])
        elif method == 'minmax':
            sub_df_copy[numeric_cols] = MinMaxScaler().fit_transform(sub_df_copy[numeric_cols])
        return sub_df_copy

    if args.group:
        group_cols = _parse_multi_cols(args.group, df.columns, is_header_present)
        return df_copy.groupby(group_cols).apply(scale_group, args.method, column_names).reset_index(drop=True)

    numeric_cols = _get_numeric_cols(df_copy, column_names)
    if args.method == 'zscore':
        df_copy[numeric_cols] = StandardScaler().fit_transform(df_copy[numeric_cols])
    elif args.method == 'minmax':
        df_copy[numeric_cols] = MinMaxScaler().fit_transform(df_copy[numeric_cols])
    return df_copy

@process_columns(required=False, multi=True)
def _handle_col_select(df, args, column_names, **kwargs):
    type_selected_cols = []
    if args.only_numeric or args.only_integer or args.only_string:
        df_copy = df.copy()
        for col in df_copy.columns:
            try: df_copy[col] = pd.to_numeric(df_copy[col], errors='ignore')
            except (ValueError, TypeError): pass
        if args.only_numeric: type_selected_cols = df_copy.select_dtypes(include='number').columns.tolist()
        elif args.only_integer: type_selected_cols = df_copy.select_dtypes(include='integer').columns.tolist()
        elif args.only_string: type_selected_cols = df_copy.select_dtypes(include=['object', 'string']).columns.tolist()
    if not column_names and not type_selected_cols: raise ValueError("col_select requires columns (-c) or a type selector.")
    final_cols = column_names + [col for col in type_selected_cols if col not in column_names]
    return df[final_cols]

@process_columns(multi=True)
def _handle_col_split(df, args, column_names, **kwargs):
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    n = args.maxsplit if args.maxsplit is not None else -1
    for col_name in column_names:
        col_idx = df.columns.get_loc(col_name)
        try: split_df = df[col_name].astype(str).str.split(delim, n=n, expand=True, regex=args.regex).fillna('')
        except re.error as e: raise ValueError(f"Invalid regex for split: {e}")
        new_headers = [_get_unique_header(f"{col_name}_{i+1}", df.columns.tolist() + list(split_df.columns)) for i in range(split_df.shape[1])]
        split_df.columns = new_headers
        df = pd.concat([df.iloc[:, :col_idx+1], split_df, df.iloc[:, col_idx+1:]], axis=1)
        if not args.keep: df = df.drop(columns=[col_name])
    return df

@process_columns()
def _handle_col_strip_chars(df, args, column_names, **kwargs):
    col_name = column_names[0]
    try: stripped = df[col_name].astype(str).str.replace(args.pattern, '', regex=True)
    except re.error as e: raise ValueError(f"Invalid regex for strip: {e}")
    if args.in_place: df[col_name] = stripped
    else: df.insert(df.columns.get_loc(col_name) + 1, _get_unique_header(f"{col_name}_stripped", df.columns), stripped)
    return df

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

@process_columns()
def _handle_row_grep(df, args, column_names, **kwargs):
    col_name = column_names[0]
    series = df[col_name].astype(str)
    try:
        if args.word_file:
            with open(args.word_file, 'r', encoding='utf-8') as f: words = [line.strip() for line in f if line.strip()]
            pattern = "|".join(map(re.escape, words))
            mask = series.str.contains(pattern, regex=True, na=False)
        else: mask = series.str.contains(args.pattern, regex=True, na=False)
    except re.error as e: raise ValueError(f"Invalid regex for grep: {e}")
    return df[~mask] if args.invert else df[mask]

@process_columns(required=False)
def _handle_row_query(df, args, column_names, **kwargs):
    if args.expr:
        try: return df.query(args.expr)
        except Exception as e: raise ValueError(f"Invalid query expression: {e}")
    col_name = column_names[0]
    series = pd.to_numeric(df[col_name], errors='coerce')
    op_map = { 'lt': series < args.value, 'gt': series > args.value, 'eq': series == args.value, 'ne': series != args.value, 'le': series <= args.value, 'ge': series >= args.value }
    return df[op_map[args.operator]]

def _handle_row_sample(df, args, **kwargs):
    return df.sample(n=args.n, frac=args.f, replace=args.with_replacement)

def _handle_row_shuffle(df, args, **kwargs):
    return df.sample(frac=1).reset_index(drop=True)

@process_columns(required=False, multi=True)
def _handle_row_unique(df, args, column_names, **kwargs):
    subset = column_names if column_names else None
    return df.drop_duplicates(subset=subset)

@process_columns(multi=True)
def _handle_stat_frequency(df, args, column_names, **kwargs):
    results_dict = {}
    for col in column_names:
        denominator = df[col].notna().sum()
        if denominator == 0:
            results_dict[col] = [""] * args.top_n
            continue
        value_counts = df[col].value_counts().nlargest(args.top_n)
        col_results = [f"{val} ({count}, {(count / denominator) * 100:.2f}%)" for val, count in value_counts.items()]
        col_results.extend([""] * (args.top_n - len(col_results)))
        results_dict[col] = col_results
    return pd.DataFrame(results_dict)

@process_columns(multi=True)
def _handle_stat_outlier_row(df, args, column_names, **kwargs):
    df_copy = df.copy()
    is_header_present = kwargs.get('is_header_present', True)
    numeric_cols = _get_numeric_cols(df_copy, column_names)
    outlier_mask = pd.Series(False, index=df_copy.index)
    for col in numeric_cols:
        series = pd.to_numeric(df_copy[col], errors='coerce')
        if args.method == 'iqr':
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - args.factor * iqr, q3 + args.factor * iqr
            outlier_mask |= (series < lower) | (series > upper)
        elif args.method == 'zscore':
            mean, std = series.mean(), series.std()
            z_scores = ((series - mean) / std).abs()
            outlier_mask |= (z_scores > args.threshold)
    if args.action == 'flag':
        df_copy['is_outlier'] = outlier_mask
        return df_copy
    return df_copy[~outlier_mask]

def _handle_stat_cor(df, args, **kwargs):
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 2: raise ValueError("Correlation requires at least two numeric columns.")
    corr_matrix = df[numeric_cols].corr(method=args.method)
    return corr_matrix.stack().reset_index().rename(columns={'level_0': 'variable_1', 'level_1': 'variable_2', 0: 'correlation'})

def _handle_stat_lm(df, args, **kwargs):
    is_header_present = kwargs.get('is_header_present', True)
    
    def fit_model(sub_df, formula):
        try:
            model = smf.ols(formula, data=sub_df).fit()
            res = model.summary2().tables[1]
            res['r_squared'] = model.rsquared
            res['n_obs'] = model.nobs
            return res.reset_index().rename(columns={'index': 'term'})
        except Exception:
            return pd.DataFrame()

    if args.group:
        group_cols = _parse_multi_cols(args.group, df.columns, is_header_present)
        results = df.groupby(group_cols).apply(fit_model, args.formula).reset_index()
        return results.drop(columns='level_{}'.format(len(group_cols)))
    return fit_model(df, args.formula)

def _handle_stat_pca(df, args, **kwargs):
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 1: raise ValueError("PCA requires at least one numeric column.")
    data = df[numeric_cols].values
    n_components = min(args.n_components, data.shape[1])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)], index=df.index)
    sys.stderr.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}\n")
    if args.keep_all: return pd.concat([df, pc_df], axis=1)
    return pc_df

def _handle_stat_score(df, args, **kwargs):
    df_copy = df.copy()
    with open(args.signatures_file, 'r') as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2: continue
            sig_name, genes_str = parts
            genes = _parse_multi_cols(genes_str, df.columns, kwargs.get('is_header_present', True))
            numeric_genes = _get_numeric_cols(df, genes)
            if not numeric_genes: continue
            if args.method == 'mean':
                df_copy[sig_name] = df_copy[numeric_genes].mean(axis=1)
            elif args.method == 'median':
                df_copy[sig_name] = df_copy[numeric_genes].median(axis=1)
    return df_copy

def _handle_stat_summary(df, args, **kwargs):
    numeric_cols = df.select_dtypes(include='number')
    if numeric_cols.empty: return pd.DataFrame({'message': ['No numeric columns found.']})
    return numeric_cols.describe().transpose().reset_index().rename(columns={'index': 'column'})


@process_columns(multi=True)
def _handle_tbl_aggregate(df, args, column_names, **kwargs):
    group_cols = _parse_multi_cols(args.group, df.columns, kwargs['is_header_present'])
    
    column_names = [col for col in column_names if col not in group_cols]
    if not column_names:
        raise ValueError("No columns to aggregate after excluding group-by columns.")
        
    agg_func = args.agg.lower()
    
    if agg_func in ["sum", "mean"]:
        numeric_cols = _get_numeric_cols(df, column_names)
        summary = df.groupby(group_cols)[numeric_cols].agg(agg_func).reset_index()
        summary.rename(columns={col: f"{agg_func}_{col}" for col in numeric_cols}, inplace=True)
        return summary

    if agg_func == "value_counts":
        summaries = []
        for col in column_names:
            top_n_per_group = df.groupby(group_cols)[col].apply(
                lambda x: x.value_counts(normalize=args.normalize).nlargest(args.top_n)
            ).reset_index(name='count')
            
            value_col_name = [c for c in top_n_per_group.columns if c not in group_cols and c != 'count'][0]
            top_n_per_group.rename(columns={value_col_name: 'value'}, inplace=True)

            top_n_per_group['aggregated_column'] = col
            summaries.append(top_n_per_group)

        long_df = pd.concat(summaries, ignore_index=True)

        if args.output_long:
            return long_df

        long_df['formatted_value'] = long_df['value'].astype(str) + " (" + long_df['count'].round(4).astype(str) + ")"
        
        grouped = long_df.groupby(group_cols + ['aggregated_column'])['formatted_value'].apply(list)
        
        wide_df = grouped.unstack('aggregated_column').reset_index().fillna('')
        
        for col in wide_df.columns:
            if col not in group_cols:
                wide_df[col] = wide_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        return wide_df
        
    raise ValueError(f"Unsupported aggregation function: {agg_func}")

def _handle_tbl_clean_header(df, args, is_header_present, **kwargs):
    if is_header_present: df.columns = [_clean_string(col) for col in df.columns]
    return df

def _handle_tbl_join_meta(df, args, **kwargs):
    meta_header = 0 if not args.meta_noheader else None
    meta_df = pd.read_csv(args.meta, sep=codecs.decode(args.meta_sep, 'unicode_escape'), header=meta_header)
    key_input = _parse_single_col(args.key_column_in_input, df.columns, kwargs['is_header_present'])
    key_meta = _parse_single_col(args.key_column_in_meta, meta_df.columns, not args.meta_noheader)
    return pd.merge(df, meta_df, how=args.how, left_on=key_input, right_on=key_meta, suffixes=tuple(args.suffixes.split(',')))

def _handle_tbl_melt(df, args, **kwargs):
    id_vars = _parse_multi_cols(args.id_vars, df.columns, kwargs['is_header_present'])
    value_vars = _parse_multi_cols(args.value_vars, df.columns, kwargs['is_header_present']) if args.value_vars else None
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars)

@process_columns()
def _handle_tbl_sort(df, args, column_names, **kwargs):
    col_name = column_names[0]
    df_copy = df.copy()
    try:
        if args.pattern:
            sort_key = df_copy[col_name].astype(str).str.extract(f'({args.pattern})').iloc[:, 0].astype(float)
            df_copy['_sort_key'] = sort_key
            df_copy = df_copy.sort_values(by='_sort_key', ascending=not args.desc, kind='stable').drop(columns=['_sort_key'])
        else: df_copy = df_copy.sort_values(by=col_name, ascending=not args.desc, kind='stable')
    except re.error as e: raise ValueError(f"Invalid regex in sort pattern: {e}")
    return df_copy

def _handle_tbl_transpose(df, args, is_header_present, **kwargs):
    if is_header_present: df = pd.concat([pd.DataFrame([df.columns], columns=df.columns), df], ignore_index=True)
    if df.shape[1] < 2: raise ValueError("Transpose requires at least 2 columns.")
    transposed_df = df.iloc[:, 1:].T
    transposed_df.columns = df.iloc[:, 0].tolist()
    return transposed_df

def _handle_tbl_unmelt(df, args, **kwargs):
    return df.pivot(index=args.index, columns=args.columns, values=args.value).reset_index()

# --------------------------
# Terminal/Utility Handlers
# --------------------------
def _handle_view(df, args, is_header_present, **kwargs):
    df_to_view = df.copy()
    if args.row_index and not df_to_view.empty:
        row_idx_col_name = _parse_single_col(args.row_index, df_to_view.columns, is_header_present)
        cols = df_to_view.columns.tolist()
        cols.insert(0, cols.pop(cols.index(row_idx_col_name)))
        df_to_view = df_to_view[cols]
    
    # Pass the new precision argument to the pretty printer
    pretty_print_kwargs = {
        "is_header_present": is_header_present,
        "show_index_header": args.show_index,
        "max_col_width": args.max_col_width,
        "truncate": not args.no_truncate,
        "precision": args.precision
    }

    if args.header_view:
        if df_to_view.empty: sys.stdout.write("(empty table with no columns)\n"); sys.exit(0)
        headers = df_to_view.columns
        if not is_header_present: headers = [f'c{i+1}' for i in range(df_to_view.shape[1])]
        first_row = [str(x) for x in df_to_view.iloc[0]] if not df_to_view.empty else [''] * len(headers)
        max_header_len = max(len(h) for h in headers) if len(headers) > 0 else 0
        output = [f"{i+1:<3} | {header:>{max_header_len}} | {(first_row[i][:37] + '...') if len(first_row[i]) > 40 else first_row[i]}" for i, header in enumerate(headers)]
        sys.stdout.write("\n".join(output) + "\n")
        sys.exit(0)
    
    _pretty_print_df(df_to_view.head(args.max_rows), **pretty_print_kwargs)
    sys.exit(0)
    
# --------------------------
# Main Setup
# --------------------------
def _setup_arg_parser():
    parser = CustomArgumentParser(prog="dmunch", description="A command-line tool for manipulating tabular data.", add_help=False)
    io_opts = parser.add_argument_group("Input/Output Options")
    io_opts.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    io_opts.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    io_opts.add_argument("-f", "--file", type=argparse.FileType('r'), default=sys.stdin, help="Input file (default: stdin).")
    io_opts.add_argument("-s", "--sep", default="\t", help="Field separator for input and output.")
    io_opts.add_argument("--noheader", action="store_true", help="Input has no header.")
    io_opts.add_argument("--quotechar", default='"', help="Character used to quote fields.")
    io_opts.add_argument("--escapechar", default=None, help="Character used to escape separators in fields.")
    io_opts.add_argument("--doublequote", action=argparse.BooleanOptionalAction, default=True, help="Use --no-doublequote to disable standard CSV quote handling.")
    io_opts.add_argument("--na-values", help="Comma-separated list of strings to recognize as NA/NaN.")
    io_opts.add_argument("--na-rep", default="", help="String representation for NA/NaN values in output.")
    io_opts.add_argument("--on-bad-lines", choices=['error', 'warn', 'skip'], default='error', help="Action for lines with too many fields.")
    
    subparsers = parser.add_subparsers(dest="operation", title="Available Operations")

    # Table Commands
    p_t_agg = subparsers.add_parser("tbl_aggregate", help="Group and aggregate data.", description="Group and aggregate data.")
    p_t_agg.add_argument("-c", "--columns", required=True, help="Columns to aggregate.")
    p_t_agg.add_argument("--group", required=True, help="Columns to group by.")
    p_t_agg.add_argument("--agg", required=True, choices=['sum', 'mean', 'value_counts'])
    p_t_agg.add_argument("--normalize", action="store_true")
    p_t_agg.add_argument("--output-long", action="store_true", help="Output in long format instead of the default wide format.")
    p_t_agg.add_argument("-T", "--top-n", type=int, default=10, help="For value_counts, show the top N most frequent values.")
    
    p_t_clean = subparsers.add_parser("tbl_clean_header", help="Clean all header names.", description="Clean all header names.")
    p_t_join = subparsers.add_parser("tbl_join_meta", help="Merge a metadata file.", description="Merge a metadata file.")
    p_t_join.add_argument("--meta", required=True)
    p_t_join.add_argument("--key_column_in_input", required=True)
    p_t_join.add_argument("--key_column_in_meta", required=True)
    p_t_join.add_argument("--how", choices=['left', 'right', 'outer', 'inner'], default='left')
    p_t_join.add_argument("--suffixes", default="_x,_y")
    p_t_join.add_argument("--meta-sep", default="\t")
    p_t_join.add_argument("--meta-noheader", action="store_true")
    p_t_melt = subparsers.add_parser("tbl_melt", help="Melt table to long format.", description="Melt table to long format.")
    p_t_melt.add_argument("--id_vars", required=True)
    p_t_melt.add_argument("--value_vars")
    p_t_sort = subparsers.add_parser("tbl_sort", help="Sort table by a column.", description="Sort table by a column.")
    p_t_sort.add_argument("-c", "--columns", required=True, help="Column to sort by.")
    p_t_sort.add_argument("--desc", action="store_true")
    p_t_sort.add_argument("-p", "--pattern", help="Regex to extract numeric key for sorting.")
    p_t_transpose = subparsers.add_parser("tbl_transpose", help="Transpose the table.", description="Transpose the table.")
    p_t_unmelt = subparsers.add_parser("tbl_unmelt", help="Unmelt table to wide format.", description="Unmelt table to wide format.")
    p_t_unmelt.add_argument("--index", required=True)
    p_t_unmelt.add_argument("--columns", required=True)
    p_t_unmelt.add_argument("--value", required=True)
    
    # Row Commands
    p_r_add = subparsers.add_parser("row_add", help="Insert a new row.", description="Insert a new row.")
    p_r_add.add_argument("-i", "--row-idx", type=int, default=1)
    p_r_add.add_argument("-v", "--values", required=True)
    p_r_drop = subparsers.add_parser("row_drop", help="Delete a row by position.", description="Delete a row by position.")
    p_r_drop.add_argument("-i", "--row-idx", type=int, required=True)
    p_r_grep = subparsers.add_parser("row_grep", help="Filter rows by pattern.", description="Filter rows by pattern.")
    p_r_grep.add_argument("-c", "--columns", required=True)
    grep_group = p_r_grep.add_mutually_exclusive_group(required=True)
    grep_group.add_argument("-p", "--pattern")
    grep_group.add_argument("--word-file")
    p_r_grep.add_argument("-v", "--invert", action="store_true")
    p_r_query = subparsers.add_parser("row_query", help="Filter rows on a condition.", description="Filter rows on a condition.")
    q_group = p_r_query.add_mutually_exclusive_group(required=True)
    q_group.add_argument("-c", "--columns", help="Column to query for simple operations.")
    q_group.add_argument("--expr", help="Pandas query expression string.")
    p_r_query.add_argument("--operator", choices=['lt', 'gt', 'eq', 'ne', 'le', 'ge'])
    p_r_query.add_argument("-v", "--value", type=float)
    p_r_sample = subparsers.add_parser("row_sample", help="Randomly subsample rows.", description="Randomly subsample rows.")
    sample_group = p_r_sample.add_mutually_exclusive_group(required=True)
    sample_group.add_argument("-n", type=int, help="Number of rows to sample.")
    sample_group.add_argument("-f", type=float, help="Fraction of rows to sample.")
    p_r_sample.add_argument("--with-replacement", action="store_true")
    p_r_shuffle = subparsers.add_parser("row_shuffle", help="Randomly shuffle all rows.", description="Randomly shuffle all rows.")
    p_r_unique = subparsers.add_parser("row_unique", help="Remove duplicate rows.", description="Remove duplicate rows.")
    p_r_unique.add_argument("-c", "--columns", help="Comma-separated columns to consider for uniqueness.")

    # Column Commands
    p_c_add = subparsers.add_parser("col_add", help="Insert a new column.", description="Insert a new column.")
    p_c_add.add_argument("-c", "--columns", required=True)
    p_c_add.add_argument("-v", "--value", required=True)
    p_c_add.add_argument("--new-header", default="new_column")
    p_c_prefix = subparsers.add_parser("col_add_prefix", help="Add a prefix to values.", description="Add a prefix to values.")
    p_c_prefix.add_argument("-c", "--columns", required=True)
    p_c_prefix.add_argument("-v", "--string", required=True)
    p_c_prefix.add_argument("-d", "--delimiter", default="")
    p_c_suffix = subparsers.add_parser("col_add_suffix", help="Add a suffix to values.", description="Add a suffix to values.")
    p_c_suffix.add_argument("-c", "--columns", required=True)
    p_c_suffix.add_argument("-v", "--string", required=True)
    p_c_suffix.add_argument("-d", "--delimiter", default="")
    p_c_bin = subparsers.add_parser("col_bin", help="Discretize a numeric column.", description="Discretize a numeric column.")
    p_c_bin.add_argument("-c", "--columns", required=True)
    bin_group = p_c_bin.add_mutually_exclusive_group(required=True)
    bin_group.add_argument("--bins", type=int, help="Number of equal-width bins.")
    bin_group.add_argument("--qcut", type=int, help="Number of quantile-based bins.")
    bin_group.add_argument("--breaks", help="Comma-separated bin edges.")
    p_c_bin.add_argument("--labels", help="Comma-separated labels for bins.")
    p_c_bin.add_argument("--keep", action="store_true", help="Keep the original column.")
    p_c_capture = subparsers.add_parser("col_capture_regex", help="Capture regex group(s).", description="Capture regex group(s).")
    p_c_capture.add_argument("-c", "--columns", required=True)
    p_c_capture.add_argument("-p", "--pattern", required=True)
    p_c_capture.add_argument("--explode", action="store_true")
    p_c_clean = subparsers.add_parser("col_clean_values", help="Clean string values.", description="Clean string values.")
    p_c_clean.add_argument("-c", "--columns", required=True)
    p_c_drop = subparsers.add_parser("col_drop", help="Drop column(s).", description="Drop column(s).")
    p_c_drop.add_argument("-c", "--columns", required=True)
    p_c_encode = subparsers.add_parser("col_encode", help="Encode categorical column.", description="Encode categorical column.")
    p_c_encode.add_argument("-c", "--columns", required=True)
    p_c_encode.add_argument("--new-header")
    p_c_eval = subparsers.add_parser("col_eval", help="Create column from expression.", description="Create column from expression.")
    p_c_eval.add_argument("--expr", required=True, help="Pandas eval() expression string.")
    p_c_eval.add_argument("--new-header", default="new_column")
    p_c_eval.add_argument("--group", help="Column(s) to group by for evaluation.")
    p_c_fill = subparsers.add_parser("col_fill_na", help="Fill missing values.", description="Fill missing values.")
    p_c_fill.add_argument("-c", "--columns", required=True)
    fill_group = p_c_fill.add_mutually_exclusive_group(required=True)
    fill_group.add_argument("-v", "--value", help="Value to fill with.")
    fill_group.add_argument("--method", choices=['ffill', 'bfill', 'mean', 'median'])
    p_c_join = subparsers.add_parser("col_join", help="Join columns.", description="Join columns.")
    p_c_join.add_argument("-c", "--columns", required=True)
    p_c_join.add_argument("-d", "--delimiter", default="")
    p_c_join.add_argument("--new-header", default="joined_column")
    p_c_join.add_argument("--keep", action="store_true")
    p_c_move = subparsers.add_parser("col_move", help="Move a column.", description="Move a column.")
    p_c_move.add_argument("-c", "--columns", required=True)
    p_c_move.add_argument("-j", "--dest-column", required=True)
    p_c_move.add_argument("--position", choices=['before', 'after'], default='before')
    p_c_rename = subparsers.add_parser("col_rename", help="Rename column(s).", description="Rename column(s).")
    p_c_rename.add_argument("--map", required=True, help="Map of old:new names, comma-separated.")
    p_c_replace = subparsers.add_parser("col_replace", help="Replace values in a column.", description="Replace values in a column.")
    p_c_replace.add_argument("-c", "--columns", required=True)
    tr_group = p_c_replace.add_mutually_exclusive_group(required=True)
    tr_group.add_argument("-d", "--dict-file")
    tr_group.add_argument("--from-val")
    p_c_replace.add_argument("--to-val")
    p_c_replace.add_argument("--regex", action="store_true")
    p_c_replace.add_argument("--in-place", action="store_true")
    p_c_scale = subparsers.add_parser("col_scale", help="Scale/normalize numeric columns.", description="Scale/normalize numeric columns.")
    p_c_scale.add_argument("-c", "--columns", required=True)
    p_c_scale.add_argument("--method", choices=['zscore', 'minmax'], required=True)
    p_c_scale.add_argument("--group", help="Column(s) to group by for scaling.")
    p_c_select = subparsers.add_parser("col_select", help="Select columns by name or type.", description="Select columns by name or type.")
    p_c_select.add_argument("-c", "--columns")
    type_group = p_c_select.add_mutually_exclusive_group()
    type_group.add_argument("--only_numeric", action="store_true")
    type_group.add_argument("--only_integer", action="store_true")
    type_group.add_argument("--only_string", action="store_true")
    p_c_split = subparsers.add_parser("col_split", help="Split a column.", description="Split a column.")
    p_c_split.add_argument("-c", "--columns", required=True)
    p_c_split.add_argument("-d", "--delimiter", required=True)
    p_c_split.add_argument("--maxsplit", type=int)
    p_c_split.add_argument("--regex", action="store_true")
    p_c_split.add_argument("--keep", action="store_true")
    p_c_strip = subparsers.add_parser("col_strip_chars", help="Remove pattern from values.", description="Remove pattern from values.")
    p_c_strip.add_argument("-c", "--columns", required=True)
    p_c_strip.add_argument("-p", "--pattern", required=True)
    p_c_strip.add_argument("--in-place", action="store_true")
    
    # Analysis & Statistics
    p_a_cor = subparsers.add_parser("stat_cor", help="Compute pairwise correlation matrix.", description="Compute pairwise correlation matrix.")
    p_a_cor.add_argument("--method", choices=['pearson', 'kendall', 'spearman'], default='pearson')
    p_a_frequency = subparsers.add_parser("stat_frequency", help="Get top N frequent values.", description="Get top N frequent values.")
    p_a_frequency.add_argument("-c", "--columns", required=True)
    p_a_frequency.add_argument("-T", "--top-n", type=int, default=10)
    p_a_lm = subparsers.add_parser("stat_lm", help="Fit a linear model.", description="Fit a linear model.")
    p_a_lm.add_argument("--formula", required=True, help="Model formula (e.g., 'y ~ x1 + x2').")
    p_a_lm.add_argument("--group", help="Column(s) to group by, fitting a model for each group.")
    p_a_outlier = subparsers.add_parser("stat_outlier_row", help="Filter or flag outlier rows.", description="Filter or flag outlier rows.")
    p_a_outlier.add_argument("-c", "--columns", required=True, help="Numeric column(s) to check.")
    p_a_outlier.add_argument("--method", choices=['iqr', 'zscore'], default='iqr')
    p_a_outlier.add_argument("--factor", type=float, default=1.5, help="Factor for IQR method.")
    p_a_outlier.add_argument("--threshold", type=float, default=3.0, help="Threshold for Z-score method.")
    p_a_outlier.add_argument("--action", choices=['filter', 'flag'], default='filter')
    p_a_pca = subparsers.add_parser("stat_pca", help="Perform Principal Component Analysis.", description="Perform Principal Component Analysis.")
    p_a_pca.add_argument("--n-components", type=int, default=2)
    p_a_pca.add_argument("--keep-all", action="store_true", help="Append PCs to the original table.")
    p_a_score = subparsers.add_parser("stat_score", help="Compute signature scores.", description="Compute signature scores (e.g., for gene sets).")
    p_a_score.add_argument("--signatures-file", required=True)
    p_a_score.add_argument("--method", choices=['mean', 'median'], default='mean')
    p_a_summary = subparsers.add_parser("stat_summary", help="Get descriptive statistics.", description="Get descriptive statistics for numeric columns.")

    # Utility Operations
    p_u_view = subparsers.add_parser("view", help="Display formatted table.", description="Display formatted table.")
    p_u_view.add_argument("--max-rows", type=int, default=20)
    p_u_view.add_argument("--max-col-width", type=int, default=40)
    p_u_view.add_argument("--no-truncate", action="store_true")
    p_u_view.add_argument("-H", "--header-view", action="store_true")
    p_u_view.add_argument("--show-index", action="store_true")
    p_u_view.add_argument("-r", "--row-index", help="Column to use as a row identifier.")
    p_u_view.add_argument("-p", "--precision", type=int, help="Number of decimal places to round numeric output to.")
    return parser, subparsers

OPERATION_HANDLERS = {
    # A-Z order for maintainability
    "col_add": _handle_col_add, "col_add_prefix": _handle_col_add_prefix, "col_add_suffix": _handle_col_add_suffix,
    "col_bin": _handle_col_bin, "col_capture_regex": _handle_col_capture_regex, "col_clean_values": _handle_col_clean_values,
    "col_drop": _handle_col_drop, "col_encode": _handle_col_encode, "col_eval": _handle_col_eval, "col_fill_na": _handle_col_fill_na,
    "col_join": _handle_col_join, "col_move": _handle_col_move, "col_rename": _handle_col_rename,
    "col_replace": _handle_col_replace, "col_scale": _handle_col_scale, "col_select": _handle_col_select,
    "col_split": _handle_col_split, "col_strip_chars": _handle_col_strip_chars,
    "row_add": _handle_row_add, "row_drop": _handle_row_drop, "row_grep": _handle_row_grep, 
    "row_query": _handle_row_query, "row_sample": _handle_row_sample,
    "row_shuffle": _handle_row_shuffle, "row_unique": _handle_row_unique,
    "stat_cor": _handle_stat_cor, "stat_frequency": _handle_stat_frequency, "stat_lm": _handle_stat_lm, 
    "stat_outlier_row": _handle_stat_outlier_row, "stat_pca": _handle_stat_pca, "stat_score": _handle_stat_score, 
    "stat_summary": _handle_stat_summary,
    "tbl_aggregate": _handle_tbl_aggregate, "tbl_clean_header": _handle_tbl_clean_header, 
    "tbl_join_meta": _handle_tbl_join_meta, "tbl_melt": _handle_tbl_melt, "tbl_sort": _handle_tbl_sort, 
    "tbl_transpose": _handle_tbl_transpose, "tbl_unmelt": _handle_tbl_unmelt,
    "view": _handle_view
}

def _read_input_data(args, sep, header):
    na_values = args.na_values.split(',') if args.na_values else None
    try:
        # Use a buffer to handle potential empty stdin
        input_content = args.file.read()
        if not input_content.strip(): return pd.DataFrame()
        
        # Convert numeric columns where possible, but keep others as strings
        df = pd.read_csv(
            StringIO(input_content), sep=sep, header=header, engine='python',
            quotechar=args.quotechar, escapechar=args.escapechar,
            doublequote=args.doublequote, na_values=na_values,
            on_bad_lines=args.on_bad_lines
        )
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df
    except Exception as e:
        sys.stderr.write(f"Error reading input data: {e}\n")
        sys.exit(1)

def _write_output(df, sep, is_header, na_rep):
    if df is not None and not df.empty:
        try: df.to_csv(sys.stdout, sep=sep, index=False, header=is_header, quoting=csv.QUOTE_MINIMAL, na_rep=na_rep)
        except BrokenPipeError: 
            sys.stderr.close()
        except Exception as e: sys.stderr.write(f"Error writing output: {e}\n"); sys.exit(1)

def main():
    parser, subparsers = _setup_arg_parser()
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in subparsers.choices and sys.stdin.isatty()):
        parser.print_help() if len(sys.argv) == 1 else subparsers.choices[sys.argv[1]].print_help()
        sys.exit(0)

    args = parser.parse_args()
    if not hasattr(args, 'operation') or not args.operation: parser.print_help(); sys.exit(0)
    
    is_header_present = not args.noheader
    header_param = 0 if is_header_present else None
    input_sep = codecs.decode(args.sep, 'unicode_escape')
    
    df = _read_input_data(args, input_sep, header_param)
    if df.empty and args.operation not in ["view", "row_add"]: sys.exit(0)
    if not is_header_present and not df.empty: df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
    
    handler = OPERATION_HANDLERS.get(args.operation)
    if not handler: sys.stderr.write(f"Error: Unsupported operation '{args.operation}'.\n"); sys.exit(1)
    
    try:
        handler_kwargs = { "is_header_present": is_header_present, "input_sep": input_sep }
        processed_df = handler(df, args, **handler_kwargs)
        _write_output(processed_df, input_sep, is_header_present, args.na_rep)
    except BrokenPipeError:
        sys.exit(0)
    except (ValueError, IndexError, FileNotFoundError, KeyError, re.error) as e:
        sys.stderr.write(f"Error: {e}\n"); sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"An unexpected error occurred: {type(e).__name__} - {e}\n"); sys.exit(1)

if __name__ == "__main__":
    main()
