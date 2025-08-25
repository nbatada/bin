#!/usr/bin/env python3
"""
Tblkit: A powerful command-line tool for tabular data manipulation.

This script provides a suite of commands for filtering, cleaning, transforming,
and analyzing tabular data. It is designed to work with piped input from
standard streams, reading data into memory for processing with pandas.
"""
import sys
import argparse
import re
import os
import numpy as np
import pandas as pd
import codecs
from io import StringIO
import csv
from functools import wraps
import difflib
from typing import List, Any, Optional

__version__ = "8.24.3"


#----
# Utilities
#----
# --- paste start: advanced column resolver ---
import re

_COL_TOKEN_RE = re.compile(r"""
    ^
    (?:
      re:(?P<regex>.+)           # regex by name
      |
      (?P<name>[^,]+)            # literal name (no comma)
      |
      (?P<pos>\d+)               # 1-based position
      |
      (?P<range>\d+-\d+)         # 1-based inclusive range
    )
    $
""", re.X)

def _resolve_columns_advanced(spec, all_columns):
    """Resolve a columns spec like 'id,date,2-5,re:score_.*' into concrete names."""
    if spec is None or str(spec).strip() == "":
        return []
    cols = list(all_columns)
    n = len(cols)
    chosen = []
    for tok in [t.strip() for t in str(spec).split(",") if t.strip()]:
        m = _COL_TOKEN_RE.match(tok)
        if not m:
            if tok in cols and tok not in chosen:
                chosen.append(tok)
            continue
        if m.group("regex"):
            pat = re.compile(m.group("regex"))
            for c in cols:
                if pat.search(c) and c not in chosen:
                    chosen.append(c)
        elif m.group("name"):
            nm = m.group("name")
            if nm in cols and nm not in chosen:
                chosen.append(nm)
        elif m.group("pos"):
            i = int(m.group("pos"))
            if 1 <= i <= n:
                c = cols[i-1]
                if c not in chosen:
                    chosen.append(c)
        elif m.group("range"):
            a, b = m.group("range").split("-", 1)
            i1, i2 = max(1, int(a)), min(n, int(b))
            for i in range(i1, i2 + 1):
                c = cols[i-1]
                if c not in chosen:
                    chosen.append(c)
    return chosen
# --- paste end: advanced column resolver ---

# --------------------------
# Custom Argument Parser
# --------------------------
class ConsistentActionHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """
    Formats option lines like:
      -c,--columns col1,col2,..  comma separated list...
    Shows both short & long flags if available, and aligns columns nicely.
    """
    def _format_action_invocation(self, action: argparse.Action) -> str:
        # Positionals: keep default behavior but normalize metavar
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest.upper())(1)
            return metavar

        # Options: join short + long with comma (no space), then a single metavar (when needed)
        parts = []
        shorts = [s for s in action.option_strings if s.startswith('-') and not s.startswith('--')]
        longs  = [s for s in action.option_strings if s.startswith('--')]
        flags = shorts + longs
        head = ",".join(flags)

        # Need an argument?
        takes_value = not (action.nargs in (0, None))
        if takes_value:
            # Prefer explicit metavar, else a hint attribute, else DEST in upper
            hint = getattr(action, "metavar", None) or getattr(action, "metavar_hint", None) or action.dest.upper()
            return f"{head} {hint}"
        else:
            return head

    # Keep default spacing & the help text in the second column

class GroupedHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
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

class RequiredOptionalHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """
    Formatter that shows 'Required arguments' and 'Optional arguments' sections
    with defaults appended automatically.
    """
    def add_arguments(self, actions):
        # Separate subparser actions from normal options/positionals.
        subparsers = [a for a in actions if isinstance(a, argparse._SubParsersAction)]
        others = [a for a in actions if not isinstance(a, argparse._SubParsersAction)]

        # Hide universal obvious options like -h/--help.
        def _is_help(a):
            return isinstance(a, argparse._HelpAction) or any(
                s in ('-h', '--help') for s in getattr(a, 'option_strings', [])
            )
        others = [a for a in others if not _is_help(a)]

        # Split required vs optional for "others".
        required = [a for a in others if getattr(a, "required", False) or (not a.option_strings and a.nargs != 0)]
        optional = [a for a in others if a not in required]

        if required:
            self.start_section('Required arguments')
            for a in required:
                self.add_argument(a)
            self.end_section()

        if optional:
            self.start_section('Optional arguments')
            for a in optional:
                self.add_argument(a)
            self.end_section()

        # Subcommands: show one header line per command (right-justified name) +
        # one 'args:' line listing each option as: "-s, --long [METAVAR]  help".
        for sub in subparsers:
            heading = getattr(sub, 'title', None) or 'Subcommands'
            try:
                help_map = {ca.dest: (ca.help or '') for ca in getattr(sub, '_choices_actions', [])}
            except Exception:
                help_map = {}

            names = list(sub.choices.keys())
            if not names:
                continue

            name_colw = max((len(n) for n in names), default=8) + 2  # align within this group
            buf = [heading, "-" * len(heading)]

            def _opt_block(a):
                # Build "-s, --long [METAVAR]" part, showing BOTH short & long if present.
                shorts = [s for s in a.option_strings if s.startswith('-') and not s.startswith('--')]
                longs  = [s for s in a.option_strings if s.startswith('--')]
                flags = shorts + longs
                if not flags:
                    return None  # positional
                # Add metavar only when an argument is expected
                needs_arg = not (a.nargs in (0, None))
                mv = a.metavar or a.dest.upper()
                head = ', '.join(flags) + (f" [{mv}]" if needs_arg else "")
                # Help text
                desc = a.help or a.dest.replace('_', ' ')
                return f"{head}  {desc}"

            for name in sorted(names):
                p = sub.choices[name]
                desc = help_map.get(name, "") or (getattr(p, 'description', '') or '')

                # RIGHT-justify the name column; second column is the description.
                buf.append(f"  {name:>{name_colw}}  {desc}")

                # Gather option blocks for this subcommand
                opt_blocks = []
                for a in getattr(p, '_actions', []):
                    if isinstance(a, argparse._SubParsersAction) or isinstance(a, argparse._HelpAction):
                        continue
                    blk = _opt_block(a)
                    if blk:
                        opt_blocks.append(blk)

                if opt_blocks:
                    # One 'args:' line containing all options, separated by " ; "
                    buf.append(f"  {'':>{name_colw}}  args: " + " ; ".join(opt_blocks))

            # Emit once to avoid blank lines between items
            self.start_section('')
            self.add_text("\n".join(buf))
            self.end_section()

class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('formatter_class', GroupedHelpFormatter)
        super(CustomArgumentParser, self).__init__(*args, **kwargs)
        
    def format_help(self):
        formatter = self._get_formatter()
        if self.description:
            formatter.add_text(self.description)
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)

        # If no subparsers, fall back to normal argparse layout.
        subparser_actions = [a for a in self._actions if isinstance(a, argparse._SubParsersAction)]
        if not subparser_actions:
            for action_group in self._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            if self.epilog:
                formatter.add_text(self.epilog)
            return formatter.format_help()

        # We have a top-level subparser (groups: tbl/row/col/stat/view)
        spa = subparser_actions[0]

        # Build: group_name -> list of subcommand parsers
        groups = {}
        all_names = []
        for gname, gparser in spa.choices.items():
            cmds = []
            sub_actions = [a for a in gparser._actions if isinstance(a, argparse._SubParsersAction)]
            if not sub_actions:
                # direct command (e.g., 'view' is a one-off command)
                if gname == 'view':
                    cmds.append(gparser)
            else:
                sub = sub_actions[0]
                for _, cparser in sub.choices.items():
                    cmds.append(cparser)
            if cmds:
                groups[gname] = cmds
                all_names.extend([p.prog.split()[-1] for p in cmds])

        # Column widths, aligned across ALL groups; name column right-justified.
        name_colw = (max((len(n) for n in all_names), default=8) + 2)

        # Compact top-level listing (one line per command), with dashed header per group
        output_lines = []
        for gname in ['tbl', 'row', 'col', 'stat', 'view']:
            if gname not in groups:
                continue

            # help map for each group (not all parsers have .help)
            gparser = spa.choices.get(gname)
            sub_actions = [a for a in gparser._actions if isinstance(a, argparse._SubParsersAction)] if gparser else []
            help_map = {}
            if sub_actions:
                try:
                    help_map = {ca.dest: (ca.help or '') for ca in sub_actions[0]._choices_actions}
                except Exception:
                    help_map = {}

            sorted_cmds = sorted(groups[gname], key=lambda p: p.prog.split()[-1])
            if not sorted_cmds:
                continue

            hdr = f"[{gname}]"
            output_lines.append(hdr)
            output_lines.append("-" * len(hdr))

            # One line per subcommand: RIGHT-justified name, then description (aligned across groups)
            for p in sorted_cmds:
                name = p.prog.split()[-1]
                desc = help_map.get(name, "") or (p.description or "")
                output_lines.append(f"  {name:>{name_colw}}  {desc}")

            output_lines.append("")  # spacing between groups

        # Emit listing before options
        if output_lines:
            formatter.add_text("\n".join(output_lines).rstrip())

        # Then standard argparse option groups (I/O, etc.)
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()

        if self.epilog:
            formatter.add_text(self.epilog)

        return formatter.format_help()
    
    def _exit_with_error(self, message: str, hint: str = ""):
        use_color = sys.stderr.isatty() and os.getenv("NO_COLOR") is None
        red, reset = ("\033[91m", "\033[0m") if use_color else ("", "")
        line = "-" * (len(message.splitlines()[0]) + 8)
        formatted_message = f"\n{line}\n{red}Error: {message}{reset}\n{line}{hint}"
        self.exit(2, formatted_message)

    def error(self, message: str):
        if 'invalid choice' in message:
            m = re.search(r"invalid choice: '([^']*)'", message)
            cmd = m.group(1) if m else "unknown"
            
            choices = []
            for action in self._actions:
                if isinstance(action, argparse._SubParsersAction):
                    choices.extend(action.choices.keys())
            
            suggestion = difflib.get_close_matches(cmd, choices, n=1)
            hint = f"\nDid you mean: {suggestion[0]!r}?\n" if suggestion else "\n"
            self._exit_with_error(f"Command '{cmd}' not found.", hint=hint)
        else: self._exit_with_error(message.capitalize())        

##-====== INTERACTIVE START

# =========================
# Interactive Exploration
# =========================

def _setup_explore_parsers(subparsers):
    """Register the 'explore' group and its 'shell' subcommand."""
    # Top-level group parser (use add_parser, not add_subparser)
    pgrp = subparsers.add_parser(
        "explore",
        help="Interactive exploration shell to discover important columns and build pipelines."
    )
    sub = pgrp.add_subparsers(dest="action", metavar="action")
    sub.required = True  # require an action when 'explore' is used

    pshell = sub.add_parser(
        "shell",
        help="Start an interactive shell on the loaded table."
    )
    pshell.add_argument("--target", default=None,
                        help="Optional target column to score associations.")
    pshell.add_argument("--top", type=int, default=20,
                        help="Default top K for frequency or ranking views (default: 20).")
    pshell.add_argument("--max-rows", type=int, default=20,
                        help="Number of rows to show in previews (default: 20).")
    pshell.add_argument("--save-pipeline", default=None,
                        help="If provided, pipeline export writes here on 'export'.")
    pshell.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors in shell output.")

    # Bind the handler so args.handler is present
    pshell.set_defaults(handler=_handle_explore_shell)


# --- paste start: _handle_explore_shell ---
def _handle_explore_shell(df, args, column_names=None, **kwargs):
    """Entry point for the interactive shell (with banner and auto-sampling)."""
    import sys, os

    n_rows, n_cols = df.shape
    try:
        bytes_est = int(df.memory_usage(deep=True).sum())
    except Exception:
        bytes_est = 0
    size_mb = bytes_est / (1024 * 1024) if bytes_est else 0.0

    # Auto-sample for very large tables to keep the shell responsive.
    sampled = False
    df_view = df
    if n_rows > 10000 or size_mb > 100:  # heuristic: >10k rows or >100MB
        if n_rows > 100:
            df_view = df.sample(n=100, random_state=0)
            sampled = True

    shell = TblkitShell(
        df=df_view,
        target=getattr(args, "target", None),
        top=getattr(args, "top", 20),
        max_rows=getattr(args, "max_rows", 20),
        save_path=getattr(args, "save_pipeline", None),
        color=(not getattr(args, "no_color", False)),
    )
    shell.prompt = "tblkit> "
    base_intro = "tblkit interactive shell. Type 'help' to list commands, 'help <command>' for details, and 'exit' to quit."
    if sampled:
        shell.intro = (
            f"{base_intro}\n"
            f"Loaded table: {n_rows} rows, {n_cols} columns (~{size_mb:.1f} MB). "
            f"Showing a random sample of 100 rows for interactivity."
        )
    else:
        shell.intro = (
            f"{base_intro}\n"
            f"Loaded table: {n_rows} rows, {n_cols} columns (~{size_mb:.1f} MB)."
        )

    # Ensure the shell reads from a TTY even when data came via a pipe
    piped_in = False
    try:
        piped_in = (getattr(args, "file", None) is sys.stdin) and (not sys.stdin.isatty())
    except Exception:
        piped_in = True
    if piped_in:
        try:
            tty_in = open("/dev/tty", "r", encoding="utf-8", errors="replace")
            shell.use_rawinput = False
            shell.stdin = tty_in
        except Exception:
            try:
                fd = os.dup(0)
                shell.use_rawinput = False
                shell.stdin = os.fdopen(fd, "r", encoding="utf-8", errors="replace")
            except Exception:
                sys.stderr.write("Interactive mode requires a TTY (use: tblkit -f FILE explore shell).\n")
                return df

    shell.cmdloop()
    return df
# --- paste end: _handle_explore_shell ---


# -------------------------
# Shell implementation
# -------------------------
import sys, re, math, textwrap
import pandas as pd
import numpy as np
from cmd import Cmd

class TblkitShell(Cmd):
    intro = (
        "tblkit interactive shell. Type 'help' to list commands, "
        "'help <command>' for details, and 'exit' to quit."
    )
    prompt = "tblkit> "

    def __init__(self, df: pd.DataFrame, target=None, top=20, max_rows=20, save_path=None, color=True):
        super().__init__()
        self._df = df
        self._target = target
        self._top = int(top)
        self._max_rows = int(max_rows)
        self._save_path = save_path
        self._color = color
        self._pipeline = []  # collects tblkit command segments (without 'tblkit' prefix)

        # precompute simple metadata
        self._meta = self._profile(df)

    # ---------- utilities ----------
    def _c(self, s, code):
        if not self._color: return s
        return f"\033[{code}m{s}\033[0m"

    def _bold(self, s):   return self._c(s, "1")
    def _green(self, s):  return self._c(s, "32")
    def _cyan(self, s):   return self._c(s, "36")
    def _yellow(self, s): return self._c(s, "33")
    def _red(self, s):    return self._c(s, "31")

    def _profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer high-level properties per column."""
        rows = []
        n = len(df)
        for c in df.columns:
            s = df[c]
            nonnull = s.notna().sum()
            nulls = n - nonnull
            nunique = s.nunique(dropna=True)

            if pd.api.types.is_numeric_dtype(s):          kind = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(s): kind = "datetime"
            elif pd.api.types.is_bool_dtype(s):           kind = "bool"
            else:
                kind = "categorical" if nunique <= max(50, int(0.05 * n)) else "text"

            is_string = pd.api.types.is_string_dtype(s) or s.dtype == object
            idlike = bool(is_string and nunique >= int(0.9 * n))

            rows.append({
                "column": c, "kind": kind,
                "nonnull": int(nonnull), "nulls": int(nulls),
                "null_rate": (nulls / n) if n else 0.0,
                "unique": int(nunique), "unique_rate": (nunique / n) if n else 0.0,
                "id_like": idlike,
                "example": repr(s.dropna().iloc[0])[:60] if nonnull else "",
            })
        m = pd.DataFrame(rows)
        return m.sort_values(["kind","null_rate","unique"], ascending=[True, True, False], ignore_index=True)

    def _print_df(self, df: pd.DataFrame, max_rows=None):
        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 120)
        pd.set_option("display.max_colwidth", 100)
        if max_rows is not None:
            print(df.head(max_rows).to_string(index=False))
        else:
            print(df.to_string(index=False))

    def _require_target(self):
        if not self._target:
            print(self._red("No target set. Use:  target <column>"))
            return False
        if self._target not in self._df.columns:
            print(self._red(f"Target '{self._target}' not found in columns."))
            return False
        return True

    # ---------- help ----------
    def do_help(self, arg):
        if arg:
            return super().do_help(arg)
        lines = [
            self._bold("Available commands"),
            "",
            self._bold("Navigation / search"),
            "  columns                  - list columns with type, missingness, uniqueness",
            "  header                   - show header names with 1-based positions",
            "  search <regex>           - find columns whose names match the pattern",
            "  peek [cols] [n]          - preview n rows (default n from --max-rows)",
            "  freq [col] [k]           - top-k frequent values; defaults to several string columns",
            "  hist [col] [bins]        - ASCII histogram(s) for numeric columns (default 20 bins)",
            "  quant [col] [low high]   - show quantiles; default 0.01..0.99",
            "",
            self._bold("Target-aware scoring"),
            "  target <col>             - set the target column",
            "  score                    - rank columns by association with target",
            "  corr [k]                 - top-|r| correlations among numeric columns (pairs)",
            "",
            self._bold("Quality"),
            "  missing                  - columns sorted by missing rate",
            "  unique                   - columns sorted by uniqueness",
            "  idlike                   - likely identifiers (strings, ~unique)",
            "  strings | integers | numeric  - list columns by dtype",
            "",
            self._bold("Pipeline builder"),
            "  (Run commands below to append steps; then 'pipe show' or 'export'.)",
            "  select <cols>            - add a select step (names/positions/ranges/regex via re:)",
            "  encode <cols>            - add one-hot encode",
            "  fillna <cols> <value>    - add fillna",
            "  bin <col> <bins|qcut:k>  - add binning (e.g., '4' or 'qcut:4')",
            "  aggregate <by> <col> <funcs>  - add group aggregation (e.g., 'mean,sum')",
            "  rename old:new;...       - add column renames",
            "  pretty [rows] [first:N|last:N|COLSPEC]  - nicely print with column subset",
            "  pipe show | pipe clear   - show or clear pipeline",
            "  export [path]            - write pipeline to file (or --save-pipeline)",
            "",
            self._bold("Notes"),
            "  On very large inputs, the shell auto-samples ~100 rows for responsiveness.",
            "",
            self._bold("System"),
            "  set top <k>              - set default top K",
            "  set maxrows <n>          - set default preview row count",
            "  exit                     - leave the shell",
            "",
            "Type 'help <command>' for details.",
        ]
        print("\n".join(lines))

    # ---------- commands ----------
    def do_columns(self, arg):
        """List columns with type, missingness, uniqueness, and an example value."""
        self._print_df(self._meta[["column","kind","null_rate","unique","id_like","example"]], max_rows=None)

    def do_search(self, arg):
        """search <regex> : list matching columns."""
        if not arg.strip():
            print("Usage: search <regex>")
            return
        pat = re.compile(arg.strip(), flags=re.I)
        matches = self._meta[self._meta["column"].str.contains(pat)]
        if matches.empty:
            print("No matches.")
        else:
            self._print_df(matches[["column","kind","null_rate","unique","example"]], max_rows=None)

    def do_missing(self, arg):
        """List columns by decreasing missing rate."""
        m = self._meta.sort_values("null_rate", ascending=False)
        self._print_df(m[["column","kind","null_rate","unique"]], max_rows=None)

    def do_unique(self, arg):
        """List columns by uniqueness (descending)."""
        m = self._meta.sort_values("unique", ascending=False)
        self._print_df(m[["column","kind","unique","unique_rate"]], max_rows=None)

    def do_idlike(self, arg):
        """List columns flagged as likely identifiers."""
        m = self._meta[self._meta["id_like"]]
        if m.empty:
            print("No likely identifier columns.")
        else:
            self._print_df(m[["column","kind","unique","unique_rate"]], max_rows=None)

    def do_peek(self, arg):
        """peek [cols] [n] : preview."""
        parts = [a for a in arg.split() if a]
        cols, n = None, self._max_rows
        if parts:
            # if last token is an integer, treat as n
            if parts[-1].isdigit():
                n = int(parts[-1]); parts = parts[:-1]
            if parts:
                cols = [c.strip() for c in " ".join(parts).split(",")]
                for c in cols:
                    if c not in self._df.columns:
                        print(self._red(f"Unknown column: {c}"))
                        return
        view = self._df if not cols else self._df[cols]
        self._print_df(view, max_rows=n)

    def do_freq(self, arg):
        """freq [col] [k] : top-k value counts. If no column, show several string columns."""
        parts = [p for p in arg.split() if p]
        k = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else self._top

        def show_one(col):
            s = self._df[col]
            tbl = (s.value_counts(dropna=False)
                     .head(k)
                     .rename_axis(col)
                     .reset_index(name="count"))
            self._print_df(tbl, max_rows=None)

        if parts:
            col = parts[0]
            if col not in self._df.columns:
                print(self._red(f"Unknown column: {col}")); return
            show_one(col)
        else:
            cat_cols = [c for c in self._df.columns if not pd.api.types.is_numeric_dtype(self._df[c])]
            for c in cat_cols[:min(5, len(cat_cols))]:
                print(self._bold(f"[{c}] top {k}")); show_one(c); print()

    def do_groupfreq(self, arg):
        """groupfreq <by> [col] [k] : top-k value counts per group."""
        parts = [p for p in arg.split() if p]
        if not parts:
            print("Usage: groupfreq <by> [col] [k]"); return
        by = parts[0]
        if by not in self._df.columns:
            print(self._red(f"Unknown group column: {by}")); return
        k = self._top
        col = None
        if len(parts) >= 2 and parts[1] in self._df.columns:
            col = parts[1]
            if len(parts) >= 3 and parts[2].isdigit():
                k = int(parts[2])
        elif len(parts) >= 2 and parts[1].isdigit():
            k = int(parts[1])

        def topk(s):
            return s.value_counts(dropna=False).head(k)

        if col:
            g = self._df.groupby(by)[col].apply(topk).reset_index(name="count")
            self._print_df(g, max_rows=None)
        else:
            cat_cols = [c for c in self._df.columns if c != by and not pd.api.types.is_numeric_dtype(self._df[c])]
            for c in cat_cols[:min(5, len(cat_cols))]:
                print(self._bold(f"[{c}] by {by} (top {k})"))
                g = self._df.groupby(by)[c].apply(topk).reset_index(name="count")
                self._print_df(g, max_rows=None); print()

    def _ascii_hist(self, s: pd.Series, bins: int = 20, width: int = 40) -> None:
        """Print a simple ASCII histogram for a numeric Series."""
        x = pd.to_numeric(s, errors="coerce").dropna()
        if x.empty:
            print("(no numeric data)"); return
        counts, edges = np.histogram(x, bins=bins)
        peak = counts.max() if counts.size else 1
        for i in range(len(counts)):
            left, right = edges[i], edges[i+1]
            bar_len = int(width * (counts[i] / peak)) if peak else 0
            bar = "#" * bar_len
            print(f"[{left:,.3g}, {right:,.3g}) {counts[i]:>6} | {bar}")

    def do_hist(self, arg):
        """hist [col] [bins] : ASCII histogram(s) for numeric column(s)."""
        parts = [p for p in arg.split() if p]
        bins = 20
        col = None
        if parts:
            if parts[-1].isdigit():
                bins = int(parts[-1]); parts = parts[:-1]
            if parts:
                col = parts[0]
                if col not in self._df.columns:
                    print(self._red(f"Unknown column: {col}")); return
        if col:
            print(self._bold(f"Histogram: {col}")); self._ascii_hist(self._df[col], bins=bins)
        else:
            num_cols = list(self._df.select_dtypes(include=[np.number]).columns)
            if not num_cols:
                print("(no numeric columns)"); return
            for c in num_cols[:min(5, len(num_cols))]:
                print(self._bold(f"Histogram: {c}")); self._ascii_hist(self._df[c], bins=bins); print()

    def do_quant(self, arg):
        """quant [col] [low high] : show quantiles (default 1%..99%)."""
        parts = [p for p in arg.split() if p]
        qlow, qhigh = 0.01, 0.99
        col = None
        if parts:
            # detect 2 floats at end
            try:
                if len(parts) >= 3:
                    qlow = float(parts[-2]); qhigh = float(parts[-1]); parts = parts[:-2]
            except Exception:
                pass
            if parts:
                col = parts[0]
                if col not in self._df.columns:
                    print(self._red(f"Unknown column: {col}")); return
        def show(s, name):
            x = pd.to_numeric(s, errors="coerce").dropna()
            if x.empty:
                print(f"{name}: (no numeric data)"); return
            lo, hi = x.quantile(qlow), x.quantile(qhigh)
            print(f"{name}: q[{qlow:.2f}]={lo:.6g}, q[{qhigh:.2f}]={hi:.6g}")
        if col:
            show(self._df[col], col)
        else:
            num_cols = list(self._df.select_dtypes(include=[np.number]).columns)
            if not num_cols:
                print("(no numeric columns)"); return
            for c in num_cols[:min(5, len(num_cols))]:
                show(self._df[c], c)

    def do_rename(self, arg):
        """rename old1:new1;old2:new2  : add a rename step to the pipeline."""
        spec = arg.strip()
        if not spec:
            print("Usage: rename old1:new1;old2:new2"); return
        self._pipeline.append(f'col rename --map "{spec}"')
        print(self._green("Added: ")); print(self._pipeline[-1])

    def do_target(self, arg):
        """target <col> : set target column."""
        col = arg.strip()
        if not col:
            print("Usage: target <col>"); return
        if col not in self._df.columns:
            print(self._red(f"Unknown column: {col}")); return
        self._target = col
        print(f"Target set to {self._bold(col)}.")

    def do_header(self, arg):
        """header : show header names with 1-based positions."""
        items = [f"{c} ({i+1})" for i, c in enumerate(self._df.columns)]
        print(", ".join(items))

    def do_strings(self, arg):
        """List string/object columns."""
        cols = [c for c in self._df.columns if (pd.api.types.is_string_dtype(self._df[c]) or self._df[c].dtype == object)]
        print(", ".join(cols) if cols else "(none)")


    def do_integers(self, arg):
        """List integer columns."""
        cols = [c for c in self._df.select_dtypes(include=["int", "Int64"]).columns]
        print(", ".join(cols) if cols else "(none)")

    def do_numeric(self, arg):
        """List numeric columns."""
        cols = [c for c in self._df.select_dtypes(include=[np.number]).columns]
        print(", ".join(cols) if cols else "(none)")

    def do_pretty(self, arg):
        """pretty [rows] [first:N|last:N|COLSPEC] : print nicely with optional column slice."""
        parts = [p for p in arg.split() if p]
        nrows = self._max_rows
        colsel = None
        if parts:
            # rows if first token is int
            if parts[0].isdigit():
                nrows = int(parts[0]); parts = parts[1:]
        if parts:
            token = parts[0]
            cols = list(self._df.columns)
            if token.startswith("first:"):
                k = max(0, int(token.split(":",1)[1] or "0"))
                colsel = cols[:k]
            elif token.startswith("last:"):
                k = max(0, int(token.split(":",1)[1] or "0"))
                colsel = cols[-k:] if k > 0 else []
            else:
                colsel = _resolve_columns_advanced(token, cols)

        view = self._df if not colsel else self._df[colsel]
        self._print_df(view, max_rows=nrows)

    def do_score(self, arg):
        """Rank columns by association with the target column (must be set)."""
        if not self._require_target(): return
        y = self._df[self._target]
        scores = []
        for c in self._df.columns:
            if c == self._target: continue
            s = self._df[c]
            try:
                if pd.api.types.is_numeric_dtype(y) and pd.api.types.is_numeric_dtype(s):
                    r = y.corr(s)
                    score = abs(r) if pd.notna(r) else 0.0
                    metric = "abs_pearson"
                elif pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_numeric_dtype(s):
                    # one-way ANOVA F surrogate using group means; robust on strings
                    g = y.groupby(s).apply(lambda v: v.dropna().values)
                    groups = [np.asarray(v, dtype=float) for v in g if len(v)>1]
                    score = _anova_f_surrogate(groups) if len(groups) > 1 else 0.0
                    metric = "anova_F"
                else:
                    # Cramér’s V (approximate) for categorical-categorical
                    score = _cramers_v(self._df[[self._target, c]].dropna())
                    metric = "cramers_v"
            except Exception:
                score, metric = 0.0, "na"
            scores.append({"column": c, "score": float(score), "metric": metric})
        out = pd.DataFrame(scores).sort_values("score", ascending=False).head(self._top)
        self._print_df(out, max_rows=None)

    def do_corr(self, arg):
        """corr [k] : show the top-|r| numeric pairs."""
        k = int(arg.strip()) if arg.strip().isdigit() else self._top
        num = self._df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            print("Fewer than two numeric columns.")
            return
        corr = num.corr()
        rows = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                r = corr.iat[i, j]
                if pd.notna(r):
                    rows.append((cols[i], cols[j], float(r), abs(float(r))))
        if not rows:
            print("No finite correlations found.")
            return
        res = pd.DataFrame(rows, columns=["col1","col2","r","abs_r"]).sort_values("abs_r", ascending=False).head(k)[["col1","col2","r"]]
        self._print_df(res, max_rows=None)

    def do_pca(self, arg):
        """pca [n] : show explained variance ratio for n components (numeric columns)."""
        try:
            n = int(arg.strip()) if arg.strip() else 2
        except Exception:
            n = 2
        X = self._df.select_dtypes(include=[np.number]).copy()
        if X.empty:
            print("No numeric columns.")
            return
        X = X.fillna(X.median(numeric_only=True))
        # simple SVD PCA (no sklearn dependency)
        Xc = X - X.mean()
        U, S, Vt = np.linalg.svd(Xc.values, full_matrices=False)
        var = (S**2) / (len(X)-1) if len(X) > 1 else S**2
        total = var.sum() if var.size else 1.0
        ratio = (var / total)[:n]
        tbl = pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(ratio))],
                            "explained_variance_ratio": ratio})
        self._print_df(tbl, max_rows=None)

    # ----- Pipeline builder (record steps, do not mutate df) -----
    def do_select(self, arg):
        """select <cols>  : add a selection step to the pipeline."""
        cols = arg.strip()
        if not cols:
            print("Usage: select a,b,c"); return
        self._pipeline.append(f'col select --column-names "{cols}"')
        print(self._green("Added: ")) ; print(self._pipeline[-1])

    def do_encode(self, arg):
        """encode <cols>  : add one-hot encode step."""
        cols = arg.strip()
        if not cols:
            print("Usage: encode a,b"); return
        self._pipeline.append(f'col encode --columns "{cols}" --method onehot')
        print(self._green("Added: ")) ; print(self._pipeline[-1])

    def do_fillna(self, arg):
        """fillna <cols> <value> : add fillna step."""
        parts = arg.split()
        if len(parts) < 2:
            print("Usage: fillna <cols> <value>"); return
        cols = parts[0]; value = " ".join(parts[1:])
        self._pipeline.append(f'col fillna --columns "{cols}" --value "{value}"')
        print(self._green("Added: ")) ; print(self._pipeline[-1])

    def do_bin(self, arg):
        """bin <col> <bins|qcut:k> : add binning (equal-width or quantile)."""
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: bin <col> <bins|qcut:k>"); return
        col, spec = parts
        if spec.startswith("qcut:"):
            k = spec.split(":",1)[1]
            self._pipeline.append(f'col bin --columns "{col}" --qcut --bins {k}')
        else:
            self._pipeline.append(f'col bin --columns "{col}" --bins {spec}')
        print(self._green("Added: ")) ; print(self._pipeline[-1])

    def do_aggregate(self, arg):
        """aggregate <by> <col> <funcs> : add group aggregation."""
        parts = arg.split()
        if len(parts) < 3:
            print("Usage: aggregate <by> <col> <funcs>"); return
        by, col, funcs = parts[0], parts[1], " ".join(parts[2:])
        self._pipeline.append(f'tbl aggregate --group "{by}" --columns "{col}" --funcs "{funcs}"')
        print(self._green("Added: ")) ; print(self._pipeline[-1])

    def do_pipe(self, arg):
        """pipe show|clear : show or clear the pipeline."""
        sub = arg.strip().lower()
        if sub == "clear":
            self._pipeline.clear()
            print("Pipeline cleared.")
            return
        # default: show
        if not self._pipeline:
            print("(pipeline is empty)")
            return
        s = " | \\\n  ".join(f"tblkit {step}" for step in self._pipeline)
        print(self._bold("Pipeline:")); print(s)

    def do_export(self, arg):
        """export [path] : write pipeline to file (or to --save-pipeline)."""
        if not self._pipeline:
            print("(pipeline is empty)"); return
        path = arg.strip() or self._save_path
        if not path:
            print("Provide a path: export pipeline.txt"); return
        s = " | \\\n  ".join(f"tblkit {step}" for step in self._pipeline)
        with open(path, "w", encoding="utf-8") as f:
            f.write(s + "\n")
        print(self._green(f"Wrote pipeline to {path}"))

    def do_set(self, arg):
        """set top <k> | set maxrows <n>"""
        parts = arg.split()
        if len(parts) != 2:
            print("Usage: set top <k>  |  set maxrows <n>"); return
        key, val = parts[0].lower(), parts[1]
        if key == "top" and val.isdigit():
            self._top = int(val); print(f"top = {self._top}")
        elif key == "maxrows" and val.isdigit():
            self._max_rows = int(val); print(f"max_rows = {self._max_rows}")
        else:
            print("Usage: set top <k>  |  set maxrows <n>")

    def do_exit(self, arg):
        """Exit the shell."""
        print("Bye.")
        return True

    def do_quit(self, arg):
        """Exit the shell."""
        return self.do_exit(arg)


# ---------- small statistics helpers (no external dependencies) ----------
def _anova_f_surrogate(groups):
    """One-way ANOVA F surrogate (between/within variance ratio)."""
    # groups: list of 1-D arrays
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k: return 0.0
    all_vals = np.concatenate(groups)
    grand = np.mean(all_vals) if len(all_vals) else 0.0
    ssb = sum(len(g) * (np.mean(g) - grand)**2 for g in groups)
    ssw = sum(((g - np.mean(g))**2).sum() for g in groups)
    msb = ssb / (k - 1) if k > 1 else 0.0
    msw = ssw / (n - k) if (n - k) > 0 else 1.0
    F = msb / msw if msw != 0 else 0.0
    return float(F)

def _cramers_v(df2cols: pd.DataFrame):
    """Cramér’s V for two categorical columns (approximate)."""
    if df2cols.shape[1] != 2 or df2cols.empty: return 0.0
    a, b = df2cols.columns
    tab = pd.crosstab(df2cols[a], df2cols[b])
    chi2 = _chi2_stat(tab.values)
    n = tab.values.sum()
    if n == 0: return 0.0
    r, c = tab.shape
    denom = n * (min(r, c) - 1) if min(r, c) > 1 else n
    v = math.sqrt(chi2 / denom) if denom else 0.0
    return float(v)

def _chi2_stat(obs: np.ndarray):
    """Pearson chi-squared statistic for a contingency table."""
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    exp = (row_sums @ col_sums) / total if total else np.zeros_like(obs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = ((obs - exp)**2 / exp)
        stat[~np.isfinite(stat)] = 0.0
    return float(stat.sum())


##====== INTERACTIVE END

# --------------------------
# Utility Functions
# --------------------------
def _get_unique_header(candidate: str, existing_columns: List[str]) -> str:
    if candidate not in existing_columns: return candidate
    i = 1
    while f"{candidate}_{i}" in existing_columns: i += 1
    return f"{candidate}_{i}"

def _parse_single_col(value: str, df_columns: pd.Index, is_header_present: bool) -> str:
    try:
        col_idx = int(value) - 1
        if not (0 <= col_idx < len(df_columns)): raise IndexError(f"Column index '{value}' is out of bounds.")
        return df_columns[col_idx]
    except (ValueError, TypeError):
        if not is_header_present: raise ValueError("Cannot use column names when --noheader is specified.")
        if value not in df_columns: raise ValueError(f"Column '{value}' not found. Available: {list(df_columns)}")
        return value

def _parse_multi_cols(values: str, df_columns: pd.Index, is_header_present: bool) -> List[str]:
    if not values: return []
    stripped_values = [val.strip() for val in values.split(',') if val.strip()]
    if "all" in [v.lower() for v in stripped_values]: return list(df_columns)
    return [_parse_single_col(val, df_columns, is_header_present) for val in stripped_values]

def _get_numeric_cols(df: pd.DataFrame, specified_cols: List[str]) -> List[str]:
    numeric_cols = df[specified_cols].select_dtypes(include='number').columns.tolist()
    if not numeric_cols: raise ValueError("Operation requires at least one numeric column.")
    return numeric_cols

# --------------------------
# View helpers: cleaning & formatting
# --------------------------
def _derive_short_alias(dest: str, used: set) -> str | None:
    """Pick a short alias like '-c' from dest; avoid collisions with 'used'."""
    for ch in dest:
        if ch.isalpha():
            cand = "-" + ch.lower()
            if cand not in used:
                return cand
    for ch in "abcdefghijklmnopqrstuvwxyz":
        cand = "-" + ch
        if cand not in used:
            return cand
    return None

def _add_missing_short_options_on_parser(parser: argparse.ArgumentParser):
    """For each long-only option, auto-add a short alias (so -h shows both forms)."""
    used = {s for s in getattr(parser, "_option_string_actions", {}) if len(s) == 2 and s.startswith("-") and not s.startswith("--")}
    for a in parser._actions:
        if isinstance(a, argparse._HelpAction):   # keep help as-is
            continue
        if not getattr(a, "option_strings", None):
            continue
        if isinstance(a, argparse._SubParsersAction):
            continue
        shorts = [s for s in a.option_strings if len(s) == 2 and s.startswith("-") and not s.startswith("--")]
        longs  = [s for s in a.option_strings if s.startswith("--")]
        if longs and not shorts:
            short = _derive_short_alias(a.dest, used)
            if short:
                a.option_strings.insert(0, short)
                used.add(short)
                parser._option_string_actions[short] = a
                if hasattr(parser, "_optionals") and hasattr(parser._optionals, "_option_string_actions"):
                    parser._optionals._option_string_actions[short] = a

def _iter_all_parsers(root: argparse.ArgumentParser):
    """Yield root and all child parsers (recursively)."""
    seen = set()
    stack = [root]
    while stack:
        p = stack.pop()
        if id(p) in seen:
            continue
        seen.add(id(p))
        yield p
        for act in getattr(p, "_actions", []):
            if isinstance(act, argparse._SubParsersAction):
                for sub in act.choices.values():
                    stack.append(sub)

def _ensure_short_flags_everywhere(root: argparse.ArgumentParser):
    """Auto-add short flags to every long-only option across the whole tree."""
    for p in _iter_all_parsers(root):
        _add_missing_short_options_on_parser(p)

def _apply_formatter_everywhere(root: argparse.ArgumentParser, fmt=ConsistentActionHelpFormatter):
    """Force our formatter everywhere (affects Options: section look & alignment)."""
    for p in _iter_all_parsers(root):
        p.formatter_class = fmt

def _set_metavar_hints_everywhere(root: argparse.ArgumentParser):
    """
    Set friendlier metavars for common option names. You can extend this map as needed.
    """
    HINTS = {
        "index": "column_name",
        "columns": "col1,col2,..",
        "value": "column_name",
        "values": "col1,col2,..",
        "agg": "func",
        "include_columns": "col1,col2,..",
        "sep": "char",
        "na_rep": "STRING",
        "numeric_na": "nan|0",
    }
    for p in _iter_all_parsers(root):
        for a in getattr(p, "_actions", []):
            if getattr(a, "option_strings", None) and a.dest in HINTS and not getattr(a, "metavar", None):
                a.metavar = HINTS[a.dest]


def _is_numeric_like(value: Any) -> bool:
    """True if value is a string that safely parses to a finite float (no letters)."""
    if not isinstance(value, str):
        return False
    s = value.strip()
    # Fast reject when letters present (prevents '3.14e-2' from being rejected; keep E/e allowed)
    if re.search(r"[A-DF-Za-df-z]", s):  # allow e/E in scientific notation
        return False
    try:
        x = float(s)
        return x == x and x not in (float('inf'), float('-inf'))
    except Exception:
        return False

_punct_re = re.compile(r'[^0-9A-Za-z_]+')
def _cleanup_string(s: Any) -> Any:
    """Normalize strings to snake_case ASCII, stripping punctuation and whitespace."""
    if s is None or (isinstance(s, float) and pd.isna(s)): return s
    if not isinstance(s, str): s = str(s)
    import unicodedata
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii', 'ignore')
    s = s.strip().lower()
    s = s.replace(' ', '_').replace('-', '_')
    s = re.sub(r'[\\.()/]+', '_', s)
    s = _punct_re.sub('_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s

def _coerce_whole_floats_to_ints(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Floats that are whole numbers -> nullable Int64
    for c in out.select_dtypes(include='number').columns:
        if pd.api.types.is_float_dtype(out[c]):
            s = out[c]
            mask = s.notna()
            if mask.any() and ((s[mask] % 1).abs() < 1e-12).all():
                out[c] = s.astype('Int64')
    # Strings like "123.0" -> "123"
    for c in out.select_dtypes(include=['object', 'string']).columns:
        out[c] = out[c].map(lambda x: re.sub(r'(?<=\d)\.0+$', '', x) if isinstance(x, str) and re.fullmatch(r'\s*-?\d+\.0+\s*', x) else x)
    return out

def _stringify_floats_sig(df: pd.DataFrame, sig: int) -> pd.DataFrame:
    """Return a copy where numeric columns are rendered as strings with N significant digits."""
    out = df.copy()
    fmt = lambda v: (np.nan if pd.isna(v) else format(float(v), f'.{sig}g'))
    for c in out.select_dtypes(include='number').columns:
        out[c] = out[c].map(fmt)
    return out

def _clean_table(df: pd.DataFrame, na_rep: str = "NA", numeric_na: str = "nan", exclude: List[str] = None) -> pd.DataFrame:
    out = df.copy()
    exc = set(exclude or [])

    # Clean headers (skip excluded columns)
    out.columns = [c if c in exc else _cleanup_string(str(c)) for c in out.columns]

    # Clean string/object cells (skip excluded columns),
    # BUT preserve numeric-like strings (floats/ints as text) as-is.
    for c in out.select_dtypes(include=['object', 'string']).columns:
        if c in exc:
            continue
        out[c] = out[c].map(
            lambda v: na_rep if pd.isna(v)
            else (v if _is_numeric_like(v) else _cleanup_string(v))
        )

    # Numeric NA policy (skip excluded columns)
    num_cols = [c for c in out.select_dtypes(include='number').columns if c not in exc]
    if numeric_na == '0':
        out[num_cols] = out[num_cols].fillna(0)
    # else: leave dtype as-is (prevents forced floats with ".0")

    return out

def with_columns(required: bool = True, multi: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
            if not hasattr(args, 'columns') or args.columns is None:
                if required: raise ValueError(f"Operation '{args.action}' requires a -c/--columns argument.")
                return func(df, args, column_names=[], **kwargs)
            is_header = kwargs.get('is_header_present', True)
            if multi: column_names = _parse_multi_cols(args.columns, df.columns, is_header)
            else: column_names = [_parse_single_col(args.columns, df.columns, is_header)] if args.columns else []
            if required and not column_names: raise ValueError("No valid columns were specified.")
            return func(df, args, column_names=column_names, **kwargs)
        return wrapper
    return decorator

def _pretty_print_df(df: pd.DataFrame, is_header_present: bool, **kwargs):
    if df.empty:
        if is_header_present and list(df.columns): print(" | ".join(df.columns))
        print("(empty table)")
        return
    
    df_to_print = df.copy()
    precision = kwargs.get('precision')
    if precision is not None:
        numeric_cols = df_to_print.select_dtypes(include='number').columns
        df_to_print[numeric_cols] = df_to_print[numeric_cols].round(precision)

    max_col_width = kwargs.get('max_col_width', 40)
    truncate = kwargs.get('truncate', True)
    
    df_str = df_to_print.astype(str)
    col_widths = {}
    for col in df_str.columns:
        header_len = len(str(col)) if is_header_present else 0
        max_data_len = df_str[col].str.len().max()
        if pd.isna(max_data_len): max_data_len = 0
        col_widths[col] = min(max(header_len, int(max_data_len)), max_col_width)

    if kwargs.get('show_index_header'):
        print(" | ".join([f"{str(i+1):>{col_widths[col]}}" for i, col in enumerate(df.columns)]))
    
    if is_header_present:
        print(" | ".join([f"{str(col):<{col_widths[col]}}" for col in df.columns]))
        print("-+-".join(["-" * col_widths[col] for col in df.columns]))

    for _, row in df_str.iterrows():
        row_values = []
        for col in df.columns:
            val = row[col]
            width = col_widths[col]
            if truncate and len(val) > width:
                val = val[:width - 3] + "..."
            row_values.append(f"{val:<{width}}")
        print(" | ".join(row_values))

# --------------------------
# Operation Handlers
# --------------------------

@with_columns(multi=False, required=False)
def _handle_col_rename(df, args, column_names=None, **kwargs):
    """
    Rename one or more columns using a mapping string like:
      "old1:new1;old2:new2"
    or commas as separators:
      "old1:new1,old2:new2"
    """
    import re

    # Prefer an explicit mapping in args.map; fall back to column_names string.
    mapping_spec = getattr(args, "map", None) or getattr(args, "column_names", None)
    if not mapping_spec:
        return df

    # Allow ';' or ',' as pair separators.
    pairs = re.split(r"[;,]", str(mapping_spec))
    rename_map = {}
    for p in pairs:
        p = p.strip()
        if not p:
            continue
        if ":" not in p:
            # If a bare name is supplied, skip or map to itself.
            continue
        old, new = p.split(":", 1)
        old, new = old.strip(), new.strip()
        if old:
            rename_map[old] = new

    if not rename_map:
        return df

    return df.rename(columns=rename_map, inplace=False)

def _handle_tbl_add_header(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Sets the header for a headerless table."""
    is_header_present = kwargs.get('is_header_present', True)
    if is_header_present and not args.quiet:
        sys.stderr.write("Warning: Input table already has a header. The new header will overwrite it.\n")

    if args.names:
        values = [codecs.decode(v.strip(), 'unicode_escape') for v in args.names.split(',')]
        if not df.empty and len(values) != df.shape[1]:
            raise ValueError(f"Number of header values ({len(values)}) must match number of columns ({df.shape[1]}).")
    else:
        if df.empty:
            raise ValueError("Cannot auto-generate headers for an empty input. Please provide names with --names.")
        values = [f"c{i+1}" for i in range(df.shape[1])]

    df.columns = values
    return df

@with_columns(multi=True)
def _handle_col_add(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Adds a new column, or a prefix/suffix to existing columns."""
    if args.new_header: # Add a new column
        pos_col = column_names[0]
        pos_idx = df.columns.get_loc(pos_col)
        value = codecs.decode(args.value, 'unicode_escape') if args.value else ''
        new_header = _get_unique_header(args.new_header, df.columns)
        df.insert(pos_idx, new_header, value)
    elif args.prefix is not None:
        prefix = codecs.decode(args.prefix, 'unicode_escape')
        delim = codecs.decode(args.delimiter, 'unicode_escape')
        for col in column_names: df[col] = prefix + delim + df[col].astype(str)
    elif args.suffix is not None:
        suffix = codecs.decode(args.suffix, 'unicode_escape')
        delim = codecs.decode(args.delimiter, 'unicode_escape')
        for col in column_names: df[col] = df[col].astype(str) + delim + suffix
    return df

@with_columns(multi=True)
def _handle_col_bin(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Bins numeric columns into discrete intervals."""
    df_copy = df.copy()
    for col in column_names:
        series = pd.to_numeric(df_copy[col], errors='coerce')
        new_col_name = _get_unique_header(f"{col}_binned", df_copy.columns)
        labels = args.labels.split(',') if args.labels else False
        
        if args.qcut:
            bins = args.qcut
            if labels and len(labels) != bins: raise ValueError(f"Number of labels ({len(labels)}) must match number of bins ({bins}).")
            df_copy[new_col_name] = pd.qcut(series, q=bins, labels=labels, duplicates='drop')
        elif args.breaks:
            breaks = [float(b) for b in args.breaks.split(',')]
            if labels and len(labels) != len(breaks) - 1: raise ValueError(f"Number of labels ({len(labels)}) must be one less than the number of breaks ({len(breaks)}).")
            df_copy[new_col_name] = pd.cut(series, bins=breaks, labels=labels, right=False)
        else:
            bins = args.bins
            if labels and len(labels) != bins: raise ValueError(f"Number of labels ({len(labels)}) must match number of bins ({bins}).")
            df_copy[new_col_name] = pd.cut(series, bins=bins, labels=labels)
        
        if not args.keep: df_copy = df_copy.drop(columns=[col])
    return df_copy

@with_columns(multi=True)
def _handle_col_extract(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Captures groups from a regex pattern into new columns."""
    try:
        compiled_pattern = re.compile(args.pattern)
        num_groups = compiled_pattern.groups
        if num_groups == 0:
            raise ValueError("Pattern for col extract must contain at least one capture group, e.g., '(\\d+)'.")
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")

    df_out = df.copy()

    def make_regex_extractor(cell):
        match = compiled_pattern.search(str(cell))
        if not match:
            return (pd.NA,) * num_groups
        return tuple(g if g is not None else pd.NA for g in match.groups())

    for col in column_names:
        extracted_tuples = df_out[col].apply(make_regex_extractor)
        extracted_df = pd.DataFrame(extracted_tuples.tolist(), index=df_out.index)
        if num_groups > 1:
            base = f"{col}_capture"
            new_names = [_get_unique_header(f"{base}_{i+1}", df_out.columns) for i in range(num_groups)]
        else:
            new_names = [_get_unique_header(f"{col}_captured", df_out.columns)]
        
        extracted_df.columns = new_names
        df_out = df_out.join(extracted_df)
    return df_out

@with_columns(multi=True)
def _handle_col_clean_values(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Cleans string values in columns to be machine-readable."""
    for col in column_names: df[col] = df[col].apply(_cleanup_string)
    return df

@with_columns(multi=True)
def _handle_col_drop(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Drops one or more columns from the table."""
    return df.drop(columns=column_names)

@with_columns()
def _handle_col_encode(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Converts a categorical column into a new integer-encoded column."""
    col_name = column_names[0]
    codes, _ = pd.factorize(df[col_name])
    new_header = args.output or _get_unique_header(f"{col_name}_encoded", df.columns)
    if args.in_place:
        df[col_name] = codes
    else:
        df.insert(df.columns.get_loc(col_name) + 1, new_header, pd.Series(codes, index=df.index))
    return df

@with_columns(required=False)
def _handle_col_eval(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Creates a new column by evaluating a pandas expression."""
    df_copy = df.copy()
    new_header = _get_unique_header(args.output, df.columns)
    is_header_present = kwargs.get('is_header_present', True)

    def eval_on_group(sub_df):
        try:
            return sub_df.eval(args.expr, engine='numexpr')
        except Exception:
            return sub_df.eval(args.expr, engine='python')

    if args.group:
        group_cols = _parse_multi_cols(args.group, df.columns, is_header_present)
        df_copy[new_header] = df_copy.groupby(group_cols).apply(eval_on_group).reset_index(level=group_cols, drop=True)
    else:
        try:
            df_copy[new_header] = df.eval(args.expr, engine='numexpr')
        except Exception:
            df_copy[new_header] = df.eval(args.expr, engine='python')
    return df_copy

@with_columns(multi=True)
def _handle_col_fillna(df, args, column_names=None, **kwargs):
    """
    Fill NA in selected columns with a user-provided value, avoiding deprecated
    pd.to_numeric(errors='ignore') behavior.
    """
    is_hdr = kwargs.get("is_header_present", ".*")
    cols = list(column_names) if column_names is not None else _parse_multi_cols(
        getattr(args, "columns", None) or getattr(args, "column_names", None),
        df.columns, is_hdr
    )
    if not cols:
        return df

    raw_val = getattr(args, "value", None)

    # Try numeric conversion; fall back to the original value on failure.
    try:
        fill_val = pd.to_numeric(raw_val)
    except Exception:
        fill_val = raw_val

    for c in cols:
        df[c] = df[c].fillna(fill_val)
    return df

@with_columns(multi=True)
def _handle_col_join(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Joins the values of multiple columns into a single new column."""
    delim = codecs.decode(args.delimiter, 'unicode_escape')
    joined_series = df[column_names].astype(str).agg(delim.join, axis=1)
    new_header = _get_unique_header(args.output, df.columns)
    min_idx = min(df.columns.get_loc(c) for c in column_names)
    df_to_modify = df.drop(columns=column_names) if not args.keep else df.copy()
    df_to_modify.insert(min_idx, new_header, joined_series)
    return df_to_modify

@with_columns()
def _handle_col_move(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Moves a column to a new position relative to another column."""
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

@with_columns()
def _handle_col_replace(df, args, column_names=None, **kwargs):
    """
    Replace values or substrings in one or more columns.

    Args (from args):
      pattern / repl        : substring replacement; use regex if args.regex==True (args.fixed==False)
      from_val / to_val     : exact value replacement (non-regex)
      columns/column_names  : target columns (string like "a,b" or list)
      in_place (bool)       : overwrite original column (default: False)
      output (str|None)     : if provided and not in_place, write to this column
      regex (bool)          : use regex for pattern
      fixed (bool)          : treat pattern as literal (overrides regex)
    """
    import re
    is_hdr = kwargs.get("is_header_present", ".*")

    # Safe defaults for missing CLI flags
    in_place = bool(getattr(args, "in_place", False))
    output = getattr(args, "output", None)
    use_regex = bool(getattr(args, "regex", False))
    fixed = bool(getattr(args, "fixed", False))
    pattern = getattr(args, "pattern", None)
    repl = getattr(args, "repl", None)
    from_val = getattr(args, "from_val", None)
    to_val = getattr(args, "to_val", None)

    # Resolve columns
    if column_names is None:
        cols_spec = getattr(args, "columns", None) or getattr(args, "column_names", None)
        cols = _parse_multi_cols(cols_spec, df.columns, is_hdr) if cols_spec is not None else []
    else:
        cols = list(column_names)
    if not cols:
        return df

    def _apply_replace(series):
        # Value-level exact replace
        if from_val is not None or to_val is not None:
            return series.replace(from_val, to_val, regex=False)

        # Substring replace path (pattern + repl)
        if pattern is None or repl is None:
            return series  # nothing to do

        # Literal or regex
        rx = pattern if not fixed and use_regex else re.escape(str(pattern))
        # Work on text representation; keep dtype best-effort
        s_text = series.astype(str)
        out = s_text.str.replace(rx, str(repl), regex=True)
        try:
            # If original dtype was numeric and replacement did not change values, restore
            if pd.api.types.is_numeric_dtype(series.dtype):
                as_num = pd.to_numeric(out, errors="coerce")
                return as_num.where(as_num.notna(), out)
        except Exception:
            pass
        return out

    for c in cols:
        result = _apply_replace(df[c])
        if in_place or not output:
            df[c] = result
        else:
            df[output] = result
    return df

@with_columns(multi=True)
def _handle_col_scale(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Scales numeric columns using z-score or min-max scaling."""
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
    except ImportError:
        raise ImportError("scikit-learn is not installed. Please run 'pip install scikit-learn'.")

    df_copy = df.copy()
    
    def scale_group(sub_df, method, cols):
        sub_df_copy = sub_df.copy()
        numeric_cols = sub_df_copy[cols].select_dtypes(include='number').columns.tolist()
        if not numeric_cols: return sub_df_copy
        
        valid_rows = sub_df_copy[numeric_cols].dropna()
        if valid_rows.empty: return sub_df_copy

        scaler = StandardScaler() if method == 'zscore' else MinMaxScaler()
        sub_df_copy.loc[valid_rows.index, numeric_cols] = scaler.fit_transform(valid_rows)
        return sub_df_copy

    if args.group:
        group_cols = _parse_multi_cols(args.group, df.columns, kwargs.get('is_header_present', True))
        return df_copy.groupby(group_cols, group_keys=False).apply(scale_group, args.method, column_names)

    numeric_cols = _get_numeric_cols(df_copy, column_names)
    scaler = StandardScaler() if args.method == 'zscore' else MinMaxScaler()
    df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    return df_copy

@with_columns(required=False, multi=True)
# --- paste start: _handle_col_select ---
def _handle_col_select(df, args, column_names=None, **kwargs):
    """
    Select columns by name, position, range, or regex.
    Examples:
      -c "id,date,2-5,re:score_.*"
    """
    spec = getattr(args, "column_names", None) or getattr(args, "columns", None)
    cols = _resolve_columns_advanced(spec, df.columns)
    if not cols:
        return df.iloc[0:0]  # empty selection if nothing matched
    return df[cols]
# --- paste end: _handle_col_select ---


@with_columns(multi=True)
def _handle_col_split(df, args, column_names=None, **kwargs):
    """
    Split a column into multiple columns.

    Args (from args):
      delimiter (str) | pattern (regex str)
      into (str): comma-separated names for new columns (optional)
      expand (bool): return multiple columns if True
      maxsplit (int): maximum splits; default unlimited
      in_place (bool): keep source column and add new columns; default False
      output (str|None): optional prefix or single output name (if not expanding)
    """
    is_hdr = kwargs.get("is_header_present", ".*")
    cols = list(column_names) if column_names is not None else _parse_multi_cols(
        getattr(args, "columns", None) or getattr(args, "column_names", None),
        df.columns, is_hdr
    )
    if not cols:
        return df

    delimiter = getattr(args, "delimiter", None)
    pattern = getattr(args, "pattern", None)
    expand = bool(getattr(args, "expand", True))
    into = getattr(args, "into", None)
    in_place = bool(getattr(args, "in_place", False))
    output = getattr(args, "output", None)
    n = getattr(args, "maxsplit", None)
    n = -1 if n is None else int(n)

    if not delimiter and not pattern:
        # default: split on whitespace
        delimiter = r"\s+"
        pattern = delimiter

    for c in cols:
        s = df[c].astype(str)
        if pattern:
            parts = s.str.split(pattern, n=n, expand=expand, regex=True)
        else:
            parts = s.str.split(delimiter, n=n, expand=expand)

        if expand:
            if isinstance(parts, pd.DataFrame):
                # Determine target column names
                if into:
                    names = [x.strip() for x in str(into).split(",")]
                    # pad or trim to match width
                    width = parts.shape[1]
                    if len(names) < width:
                        names += [f"{c}_part{j}" for j in range(len(names), width)]
                    elif len(names) > width:
                        names = names[:width]
                else:
                    names = [f"{c}_part{j}" for j in range(parts.shape[1])]
                parts.columns = names
                for name in names:
                    df[name] = parts[name]
            else:
                # Not expanding → single series; respect output if given
                tgt = output or f"{c}_split"
                df[tgt] = parts
        else:
            # expand=False gives Series of lists; write to output
            tgt = output or f"{c}_split"
            df[tgt] = parts

        # If requested to modify in place by replacing source, drop source column
        if in_place:
            # Keep new columns; remove the original
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    return df

@with_columns()
def _handle_col_strip(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Removes characters from values in a column based on a regex pattern."""
    col_name = column_names[0]
    try:
        stripped = df[col_name].astype(str).str.replace(args.pattern, '', regex=not args.fixed)
    except re.error as e:
        raise ValueError(f"Invalid regex for strip: {e}")
    if args.in_place or not args.output:
        df[col_name] = stripped
    else:
        df.insert(df.columns.get_loc(col_name) + 1, _get_unique_header(args.output, df.columns), stripped)
    return df

def _handle_row_add(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Adds a new row with specified values to the table."""
    values = [codecs.decode(v.strip(), 'unicode_escape') for v in args.values.split(',')]
    if df.empty: df = pd.DataFrame(columns=[f"col_{i+1}" for i in range(len(values))])
    if len(values) != df.shape[1]: raise ValueError(f"Number of values ({len(values)}) must match number of columns ({df.shape[1]}).")
    new_row = pd.DataFrame([values], columns=df.columns)
    insert_pos = max(0, args.row_index - 1)
    return pd.concat([df.iloc[:insert_pos], new_row, df.iloc[insert_pos:]]).reset_index(drop=True)

def _handle_row_drop(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Drops a row from the table by its position."""
    drop_pos = args.row_index - 1
    if not (0 <= drop_pos < len(df)): raise IndexError(f"Row index {args.row_index} is out of bounds.")
    return df.drop(df.index[drop_pos]).reset_index(drop=True)

def _handle_row_filter(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Filters rows based on a pandas query expression or a simple numeric comparison."""
    if args.expr:
        try:
            expr_to_run = f"not ({args.expr})" if args.invert else args.expr
            try:
                return df.query(expr_to_run, engine='numexpr')
            except Exception:
                return df.query(expr_to_run, engine='python')
        except Exception as e:
            raise ValueError(f"Invalid query expression: {e}")
    
    col_name = _parse_single_col(args.columns, df.columns, kwargs.get('is_header_present', True))
    series = df[col_name].astype(str)
    try:
        if args.word_file:
            with open(args.word_file, 'r', encoding='utf-8') as f: words = [line.strip() for line in f if line.strip()]
            pattern = "|".join(map(re.escape, words))
            mask = series.str.contains(pattern, regex=True, na=False)
        else:
            use_regex = not args.fixed
            mask = series.str.contains(args.pattern, regex=use_regex, na=False)
    except re.error as e: raise ValueError(f"Invalid regex for filter: {e}")
    return df[~mask] if args.invert else df[mask]

def _handle_row_sample(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Randomly samples rows from the table."""
    return df.sample(n=args.n, frac=args.f, replace=args.with_replacement, random_state=args.seed)

def _handle_row_shuffle(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Randomly shuffles all rows in the table."""
    return df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

@with_columns(required=False, multi=True)
def _handle_row_unique(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Removes duplicate rows, or keeps only duplicate rows if --invert is used."""
    subset = column_names if column_names else None
    if args.invert:
        return df[df.duplicated(subset=subset, keep=False)]
    else:
        return df.drop_duplicates(subset=subset)

def _handle_row_head(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Returns the first N rows of the table."""
    return df.head(args.n)

def _handle_row_tail(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Returns the last N rows of the table."""
    return df.tail(args.n)

@with_columns(multi=True)
def _handle_stat_frequency(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Calculates value counts and frequencies for specified columns."""
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

@with_columns(multi=True)
def _handle_stat_outlier_row(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Filters or flags rows containing outliers."""
    df_copy = df.copy()
    numeric_cols = _get_numeric_cols(df_copy, column_names)
    outlier_mask = pd.Series(False, index=df_copy.index)
    for col in numeric_cols:
        series = pd.to_numeric(df_copy[col], errors='coerce')
        if args.method == 'iqr':
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0: continue
            lower, upper = q1 - args.factor * iqr, q3 + args.factor * iqr
            outlier_mask |= (series < lower) | (series > upper)
        elif args.method == 'zscore':
            mean, std = series.mean(), series.std()
            if std == 0: continue
            z_scores = ((series - mean) / std).abs()
            outlier_mask |= (z_scores > args.threshold)
    if args.action == 'flag':
        df_copy['is_outlier'] = outlier_mask
        return df_copy
    return df_copy[~outlier_mask]

def _handle_stat_cor(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Computes a pairwise correlation matrix for numeric columns."""
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 2:
        raise ValueError("Correlation requires at least two numeric columns.")
    
    corr_matrix = df[numeric_cols].corr(method=args.method)

    if args.melt:
        return corr_matrix.stack().reset_index().rename(columns={'level_0': 'variable_1', 'level_1': 'variable_2', 0: 'correlation'})
    else:
        return corr_matrix.reset_index().rename(columns={'index': 'variable'})

def _handle_stat_lm(df, args, column_names=None, **kwargs):
    """
    Fit a linear model:
      - If args.formula is provided (e.g., "y ~ x1 + x2"), use statsmodels formula API.
      - Otherwise use args.y and args.X (comma-separated predictors).
    Returns a small coefficient table (term, coef, stderr, pvalue).
    """
    import warnings
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ValueWarning

    formula = getattr(args, "formula", None)
    y_name  = getattr(args, "y", None)
    X_spec  = getattr(args, "X", None)

    # Fit with warnings silenced (tiny demo data often triggers ValueWarning).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels")

        if formula:
            model = smf.ols(formula=formula, data=df)
            results = model.fit()
        else:
            if not y_name or not X_spec:
                # Nothing to do if user did not provide y and X.
                return df
            X_cols = [c.strip() for c in str(X_spec).split(",") if c.strip()]
            X = sm.add_constant(df[X_cols], has_constant="add")
            y = df[y_name]
            model = sm.OLS(y, X, missing="drop")
            results = model.fit()

    # Return a tidy table rather than a long textual summary.
    out = (
        pd.DataFrame(
            {
                "term":   results.params.index,
                "coef":   results.params.values,
                "stderr": results.bse.values,
                "pvalue": results.pvalues.values,
            }
        )
        .reset_index(drop=True)
    )
    return out

def _handle_stat_pca(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Performs Principal Component Analysis on numeric columns."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("scikit-learn is not installed. Please run 'pip install scikit-learn'.")
    
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < 1: raise ValueError("PCA requires at least one numeric column.")
    data = df[numeric_cols].dropna()
    if data.empty:
        if not args.quiet:
            sys.stderr.write("Warning: No valid data for PCA after dropping NaNs.\n")
        return pd.DataFrame(columns=[f'PC{i+1}' for i in range(args.n_components)])

    n_components = min(args.n_components, data.shape[1])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)], index=data.index)
    
    if not args.quiet:
        sys.stderr.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}\n")
    if args.keep_all: return pd.concat([df, pc_df], axis=1).drop(columns=df.columns.difference(df.columns))
    return pc_df

def _handle_stat_score(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Computes signature scores based on a list of genes or features."""
    df_copy = df.copy()
    
    normalized_df = None
    if args.method == 'normalized_mean':
        numeric_cols = df_copy.select_dtypes(include='number').columns
        p99 = df_copy[numeric_cols].quantile(0.99)
        p99[p99 == 0] = 1.0
        
        normalized_df = df_copy[numeric_cols].div(p99, axis=1)
        normalized_df = normalized_df.clip(upper=1.0)

    with open(args.signatures_file, 'r') as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2: continue
            
            sig_name, genes_str = parts
            genes = _parse_multi_cols(genes_str, df.columns, kwargs.get('is_header_present', True))
            numeric_genes = df_copy[genes].select_dtypes(include='number').columns.tolist()

            if not numeric_genes: continue

            if args.method == 'mean':
                df_copy[sig_name] = df_copy[numeric_genes].mean(axis=1)
            elif args.method == 'median':
                df_copy[sig_name] = df_copy[numeric_genes].median(axis=1)
            elif args.method == 'normalized_mean':
                df_copy[sig_name] = normalized_df[numeric_genes].sum(axis=1) / len(numeric_genes)

    return df_copy

def _handle_stat_summary(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Computes descriptive statistics for all numeric columns."""
    numeric_cols = df.select_dtypes(include='number')
    if numeric_cols.empty: return pd.DataFrame({'message': ['No numeric columns found.']})
    return numeric_cols.describe().transpose().reset_index().rename(columns={'index': 'column'})

@with_columns(multi=True)
def _handle_tbl_aggregate(df, args, column_names=None, **kwargs):
    """
    Group and aggregate.

    Args (from args):
      group (str|list): grouping columns spec
      columns/column_names: columns to aggregate (optional; defaults to numeric)
      funcs/agg (str): function name or comma-separated list (e.g., "mean,sum")
    """
    is_hdr = kwargs.get("is_header_present", ".*")

    group_spec = getattr(args, "group", None)
    if not group_spec:
        return df
    group_cols = _parse_multi_cols(group_spec, df.columns, is_hdr)

    cols_spec = getattr(args, "columns", None) or getattr(args, "column_names", None)
    if cols_spec:
        agg_cols = _parse_multi_cols(cols_spec, df.columns, is_hdr)
    else:
        agg_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)]

    funcs_spec = getattr(args, "funcs", None) or getattr(args, "agg", None) or "mean"
    funcs = [f.strip() for f in str(funcs_spec).split(",") if f.strip()]

    agg_dict = {c: funcs for c in agg_cols} if len(funcs) > 1 else {c: funcs[0] for c in agg_cols}

    out = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    return out

def _handle_tbl_clean_header(df: pd.DataFrame, args: argparse.Namespace, is_header_present: bool, **kwargs) -> pd.DataFrame:
    """Cleans all column headers to be machine-readable."""
    if is_header_present: df.columns = [_cleanup_string(col) for col in df.columns]
    return df

def _handle_tbl_join(df, args, column_names=None, **kwargs):
    """
    Merge the in-memory left DataFrame `df` with a right CSV file.

    Expected args:
      how                : 'inner'|'left'|'right'|'outer'
      on                 : join key column name (must exist on both sides)
      right              : path to the right-hand CSV
      suffixes           : string like "_x,_y"
      right_noheader     : bool; treat right file as headerless
      right_sep          : delimiter for right file (default ',')
      right_encoding     : encoding for right file (default 'utf-8')
      right_quotechar    : quote character; if None, quoting is disabled
      right_escapechar   : escape character or None
      right_doublequote  : bool for CSV doublequote handling
      na_values          : values to treat as NA when reading right file
    """
    import pandas as pd
    import csv as _csv

    # --- read the right table robustly with/without quoting ---
    if getattr(args, "right", None) is None:
        raise ValueError("tbl join: --right path is required")

    sep = getattr(args, "right_sep", ",") or ","
    enc = getattr(args, "right_encoding", "utf-8") or "utf-8"
    qchar = getattr(args, "right_quotechar", None)
    esc   = getattr(args, "right_escapechar", None)
    dquote = bool(getattr(args, "right_doublequote", True))
    header = None if bool(getattr(args, "right_noheader", False)) else "infer"
    na_vals = getattr(args, "na_values", None)

    if qchar:
        quoting = _csv.QUOTE_MINIMAL
        read_kw = dict(sep=sep, encoding=enc, quotechar=qchar, escapechar=esc,
                       doublequote=dquote, quoting=quoting, header=header,
                       na_values=na_vals)
    else:
        # Disable quoting entirely to avoid "quotechar must be set…" errors.
        quoting = _csv.QUOTE_NONE
        read_kw = dict(sep=sep, encoding=enc, escapechar=esc,
                       doublequote=dquote, quoting=quoting, header=header,
                       na_values=na_vals)
    right_df = pd.read_csv(getattr(args, "right"), **read_kw)

    # If headerless, pandas will create integer column names; user-supplied `on`
    # must then refer to those names. We do not auto-infer here.

    # --- perform the merge ---
    how = getattr(args, "how", "inner") or "inner"
    on  = getattr(args, "on", None)
    if not on:
        raise ValueError("tbl join: --on <column> is required")

    # suffixes: accept "_x,_y" or default to ("_x","_y")
    sfx = getattr(args, "suffixes", None)
    if sfx and isinstance(sfx, str) and "," in sfx:
        left_sfx, right_sfx = [t.strip() for t in sfx.split(",", 1)]
        suffixes = (left_sfx or "_x", right_sfx or "_y")
    else:
        suffixes = ("_x", "_y")

    out = df.merge(right_df, how=how, on=on, suffixes=suffixes)
    return out

def _handle_tbl_melt(df, args, column_names=None, **kwargs):
    """
    Melt a wide table to long format.

    Args (from args):
      id_vars (str|list)
      value_vars (str|list)
      var_name (str)
      value_name (str)
    """
    is_hdr = kwargs.get("is_header_present", ".*")
    id_spec = getattr(args, "id_vars", None)
    val_spec = getattr(args, "value_vars", None)
    var_name = getattr(args, "var_name", None) or "variable"
    value_name = getattr(args, "value_name", None) or "value"

    id_vars = _parse_multi_cols(id_spec, df.columns, is_hdr) if id_spec else None
    value_vars = _parse_multi_cols(val_spec, df.columns, is_hdr) if val_spec else None

    out = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    return out

@with_columns(required=False, multi=True)
def _handle_col_prune(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Reports on or removes uninformative/redundant columns."""
    if args.method == 'correlated':
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) < 2:
            if not args.quiet:
                sys.stderr.write("Warning: 'correlated' method requires at least 2 numeric columns.\n")
            return df if args.execute else pd.DataFrame()

        corr_matrix = df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        report_data = []

        for column in upper_tri.columns:
            highly_correlated_cols = upper_tri.index[upper_tri[column] >= args.threshold].tolist()
            for correlated_col in highly_correlated_cols:
                if correlated_col in to_drop or column in to_drop: continue
                
                var_column = df[column].var()
                var_correlated = df[correlated_col].var()
                
                drop_candidate = correlated_col if var_column >= var_correlated else column
                keep_candidate = column if var_column >= var_correlated else correlated_col

                to_drop.add(drop_candidate)
                report_data.append({
                    'column_to_prune': drop_candidate,
                    'correlated_with': keep_candidate,
                    'correlation': corr_matrix.loc[drop_candidate, keep_candidate]
                })
        
        if not args.execute:
            return pd.DataFrame(report_data, columns=['column_to_prune', 'correlated_with', 'correlation'])
        final_drop_list = sorted(list(to_drop))
    else:
        prune_candidates = {}
        if args.method == 'zero_variance':
            for col in df.columns:
                if df[col].nunique(dropna=False) <= 1: prune_candidates[col] = "nunique <= 1"
        elif args.method == 'high_nan':
            for col in df.columns:
                if len(df) > 0:
                    nan_ratio = df[col].isna().sum() / len(df)
                    if nan_ratio >= args.threshold: prune_candidates[col] = f"nan_ratio={nan_ratio:.2f}"
        elif args.method == 'low_variance':
            numeric_cols = df.select_dtypes(include='number').columns
            variances = df[numeric_cols].var()
            for col, var in variances[variances < args.threshold].items():
                prune_candidates[col] = f"variance={var:.4f}"
        elif args.method == 'high_cardinality':
            cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
            for col in cat_cols:
                if len(df) > 0:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio >= args.threshold: prune_candidates[col] = f"unique_ratio={unique_ratio:.2f}"

        if not args.execute:
            if not prune_candidates: return pd.DataFrame(columns=['column_to_prune', 'reason'])
            report_df = pd.DataFrame.from_dict(prune_candidates, orient='index', columns=['reason'])
            return report_df.reset_index().rename(columns={'index': 'column_to_prune'})
        final_drop_list = sorted(list(prune_candidates.keys()))

    if not final_drop_list:
        if not args.quiet: sys.stderr.write("No columns to prune based on the specified criteria.\n")
        return df
    
    if not args.quiet: sys.stderr.write(f"Pruning {len(final_drop_list)} columns: {', '.join(final_drop_list)}\n")
    return df.drop(columns=final_drop_list)

@with_columns()
def _handle_tbl_sort(df: pd.DataFrame, args: argparse.Namespace, column_names: List[str], **kwargs) -> pd.DataFrame:
    """Sorts the table by one or more columns."""
    col_name = column_names[0]
    df_copy = df.copy()
    is_ascending = args.order == 'asc'
    try:
        if args.pattern:
            sort_key = pd.to_numeric(df_copy[col_name].astype(str).str.extract(f'({args.pattern})').iloc[:, 0], errors='coerce')
            df_copy['_sort_key'] = sort_key
            df_copy = df_copy.sort_values(by='_sort_key', ascending=is_ascending, kind='stable', na_position='last').drop(columns=['_sort_key'])
        else:
            df_copy = df_copy.sort_values(by=col_name, ascending=is_ascending, kind='stable', na_position='last')
    except re.error as e:
        raise ValueError(f"Invalid regex in sort pattern: {e}")
    return df_copy

def _handle_tbl_transpose(df: pd.DataFrame, args: argparse.Namespace, is_header_present: bool, **kwargs) -> pd.DataFrame:
    """Transposes the entire table, swapping rows and columns."""
    if is_header_present: df = pd.concat([pd.DataFrame([df.columns], columns=df.columns), df], ignore_index=True)
    if df.shape[1] < 2: raise ValueError("Transpose requires at least 2 columns.")
    transposed_df = df.iloc[:, 1:].T
    
    original_first_col = df.iloc[:, 0].tolist()
    new_cols = []
    for col_name in original_first_col:
        unique_name = _get_unique_header(str(col_name), new_cols)
        new_cols.append(unique_name)
    
    transposed_df.columns = new_cols
    return transposed_df

def _handle_tbl_pivot(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Converts a table from long format to wide format."""
    is_header_present = kwargs.get('is_header_present', True)
    index_col = _parse_single_col(args.index, df.columns, is_header_present)
    columns_col = _parse_single_col(args.columns, df.columns, is_header_present)
    value_col = _parse_single_col(args.value, df.columns, is_header_present)

    unmelted_df = df.pivot_table(index=index_col, columns=columns_col, values=value_col, aggfunc=args.agg).reset_index()
    unmelted_df.columns.name = None
    return unmelted_df

def _handle_view(df: pd.DataFrame, args: argparse.Namespace, is_header_present: bool, **kwargs) -> Optional[pd.DataFrame]:
    """View command behavior:
    - default (no flags): echo table as-is
    - --pretty-print: formatted view with separators and alignment
    - --precision N: float columns shown with N significant digits
    - --clean-integer-columns: show whole-number floats as integers
    - --clean-table: sanitize headers & strings, fill missing values
    """
    df_to_view = df.copy()

    # Re-order to show a row id column first if requested
    if args.row_index and not df_to_view.empty:
        row_idx_col_name = _parse_single_col(args.row_index, df_to_view.columns, is_header_present)
        cols = df_to_view.columns.tolist()
        cols.insert(0, cols.pop(cols.index(row_idx_col_name)))
        df_to_view = df_to_view[cols]

    # Convenience cleaning (supports exclusions)
    if getattr(args, "clean_table", False):
        exclude = []
        if getattr(args, "clean_exclude", None):
            exclude = _parse_multi_cols(args.clean_exclude, df_to_view.columns, is_header_present)
        df_to_view = _clean_table(df_to_view, na_rep=args.na_rep, numeric_na=args.numeric_na, exclude=exclude)

    if getattr(args, "clean_integer_columns", False):
        df_to_view = _coerce_whole_floats_to_ints(df_to_view)

    input_sep = kwargs.get('input_sep', '\t')
    output_sep = kwargs.get('output_sep', input_sep)

    # If precision is requested, format numeric columns with significant digits (as strings)
    if args.precision is not None:
        df_to_view = _stringify_floats_sig(df_to_view, args.precision)

    # Header-only fast path (still pretty)
    if getattr(args, "header_view", False):
        _pretty_print_df(
            df_to_view.head(1),
            is_header_present=is_header_present,
            show_index_header=args.show_index,
            max_col_width=args.max_col_width,
            truncate=not args.no_truncate,
            precision=None
        )
        sys.exit(0)

    # Pretty print path
    if args.pretty_print:
        _pretty_print_df(
            df_to_view.head(args.max_rows),
            is_header_present=is_header_present,
            show_index_header=args.show_index,
            max_col_width=args.max_col_width,
            truncate=not args.no_truncate,
            precision=None  # already stringified if --precision
        )
        sys.exit(0)

    # Default: raw echo (or cleaned/precision-adjusted echo) — write once, then return None
    _write_output(
        df_to_view,
        sep=output_sep,
        is_header=is_header_present,
        na_rep=args.na_rep,
        quotechar=args.quotechar,
        escapechar=args.escapechar,
        doublequote=args.doublequote
    )
    return None

# --- paste start: _handle_view_frequency ---
def _handle_view_frequency(df, args, column_names=None, **kwargs):
    """
    Show top-N value counts per column. Defaults to non-numeric columns.
    Options mirror previous 'stat frequency':
      --columns / --column-names : advanced spec (names/positions/ranges/regex)
      -n / --n / --top-n         : top K (default 20)
    """
    import pandas as pd
    top = getattr(args, "top_n", None) or getattr(args, "n", None) or 20
    spec = getattr(args, "column_names", None) or getattr(args, "columns", None)

    if spec:
        cols = _resolve_columns_advanced(spec, df.columns)
    else:
        cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    frames = []
    for c in cols:
        vc = df[c].value_counts(dropna=False).head(int(top)).rename_axis(c).reset_index(name="count")
        vc.insert(0, "column", c)
        frames.append(vc)

    return pd.concat(frames, ignore_index=True) if frames else df.head(0)
# --- paste end: _handle_view_frequency ---


@with_columns(multi=True)
def _handle_col_cast(df, args, column_names=None, **kwargs):
    """
    Cast selected columns to a target dtype, without using deprecated 'ignore'.
    If conversion fails for a cell, preserve the original value.
    """
    is_hdr = kwargs.get("is_header_present", ".*")
    cols = list(column_names) if column_names is not None else _parse_multi_cols(
        getattr(args, "columns", None) or getattr(args, "column_names", None),
        df.columns, is_hdr
    )
    if not cols:
        return df

    target = (getattr(args, "to", None) or getattr(args, "dtype", None) or "").lower()

    for c in cols:
        if target in {"int", "integer"}:
            s = pd.to_numeric(df[c], errors="coerce", downcast="integer")
            # keep original where conversion failed
            df[c] = s.where(s.notna(), df[c])
        elif target in {"float", "double"}:
            s = pd.to_numeric(df[c], errors="coerce")
            df[c] = s.where(s.notna(), df[c])
        elif target in {"str", "string", "object"}:
            df[c] = df[c].astype("string")
        elif target in {"bool", "boolean"}:
            # Try strict boolean coercion; keep original where invalid
            s = df[c].astype(str).str.lower().map({"true": True, "false": False})
            df[c] = s.where(s.notna(), df[c])
        else:
            # Best-effort astype to a given pandas/numpy dtype string
            try:
                df[c] = df[c].astype(target)
            except Exception:
                # Leave column unchanged on failure
                pass
    return df

def _handle_tbl_cat(df: pd.DataFrame, args: argparse.Namespace, **kwargs) -> pd.DataFrame:
    """Concatenates multiple files vertically."""
    # The first file is already read into `df`.
    # We need to read the subsequent files from positional arguments.
    all_dfs = [df]
    if args.files:
        for file_path in args.files:
            try:
                # Re-use global read options
                with open(file_path, 'r', encoding=args.encoding) as f:
                    next_df = _read_input_data(args, kwargs['input_sep'], 0 if not args.noheader else None)
                    all_dfs.append(next_df)
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")
            except Exception as e:
                raise IOError(f"Error reading file {file_path}: {e}")
    
    return pd.concat(all_dfs, ignore_index=True, sort=False)

# --------------------------
# Argparse Setup Helpers
# --------------------------
def add_columns_arg(p, required=True, help_text="Column(s) to operate on."):
    p.add_argument("-c", "--columns", required=required, help=help_text)

def add_pattern_args(p, required=True, help_text="Regular expression pattern."):
    p.add_argument("-p", "--pattern", required=required, help=help_text)
    p.add_argument("--fixed", action="store_true", help="Treat pattern as a literal string (no regex).")

def add_group_arg(p, required=False, help_text="Column(s) to group by."):
    p.add_argument("-g", "--group", required=required, help=help_text)
    
def add_output_arg(p, default_name="new_column", help_text="Name for the new output column."):
    p.add_argument("-o", "--output", default=default_name, help=help_text)
    p.add_argument("--in-place", action="store_true", help="Modify the column in-place instead of creating a new one.")

def _setup_tbl_parsers(subparsers):
    p = subparsers.add_parser("add-header", help="Add a header to a headerless table.")
    p.add_argument("-n", "--names", help="Optional comma-separated list of header names.")
    p.set_defaults(handler=_handle_tbl_add_header)
    p.usage_example = "tblkit tbl add-header -n id,name,val"

    p = subparsers.add_parser("aggregate", help="Group and aggregate data.")
    add_columns_arg(p, help_text="Column(s) to aggregate.")
    add_group_arg(p, required=True)
    p.add_argument("--agg", required=True, choices=['sum', 'mean', 'value_counts'])
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--output-long", action="store_true", help="Output in long format instead of the default wide format.")
    p.add_argument("-T", "--top-n", type=int, default=10, help="For value_counts, show top N values.")
    p.set_defaults(handler=_handle_tbl_aggregate)
    p.usage_example = "tblkit tbl aggregate -g region -c sales --agg mean"
    
    p = subparsers.add_parser("clean-header", help="Clean all header names.")
    p.set_defaults(handler=_handle_tbl_clean_header)
    p.usage_example = "tblkit tbl clean-header"

    p = subparsers.add_parser("join", help="Merge with a secondary table.")
    p.add_argument("--right", required=True, help="Path to the right table file.")
    key_group = p.add_mutually_exclusive_group(required=True)
    key_group.add_argument("--on", help="Column names to join on (must be in both tables).")
    key_group.add_argument("--left-on", help="Column from the left table to use as key.")
    p.add_argument("--right-on", help="Column from the right table to use as key (requires --left-on).")
    p.add_argument("--how", choices=['left', 'right', 'outer', 'inner'], default='left')
    p.add_argument("--suffixes", default="_x,_y")
    p.add_argument("--right-sep", default="\t", help="Separator for the right table.")
    p.add_argument("--right-noheader", action="store_true", help="Right table has no header.")
    p.set_defaults(handler=_handle_tbl_join)
    p.usage_example = "tblkit tbl join --right meta.tsv --on id"

    p = subparsers.add_parser("melt", help="Melt table to long format.")
    p.add_argument("--id-vars", required=True)
    p.add_argument("--value-vars")
    p.set_defaults(handler=_handle_tbl_melt)
    p.usage_example = "tblkit tbl melt --id-vars gene --value-vars 's1,s2'"

    p = subparsers.add_parser("sort", help="Sort table by a column.")
    add_columns_arg(p)
    p.add_argument("--order", choices=['asc', 'desc'], default='asc')
    p.add_argument("-p", "--pattern", help="Regex to extract numeric key for sorting.")
    p.set_defaults(handler=_handle_tbl_sort)
    p.usage_example = "tblkit tbl sort -c score --order desc"

    p = subparsers.add_parser("transpose", help="Transpose the table.")
    p.set_defaults(handler=_handle_tbl_transpose)
    p.usage_example = "tblkit tbl transpose"

    # pivot: long->wide
    p = subparsers.add_parser("pivot", help="Pivot a long table to wide format.")
    p.add_argument("-i", "--index",   required=True,
                   help="Row index key (use comma to provide multiple keys).")
    p.add_argument("-c", "--columns", required=True,
                   help="Name of the column whose VALUES become new column headers (exactly one column).")
    p.add_argument("-v", "--value",   required=True,
                   help="Name of the column that provides the cell values.")
    p.add_argument("-a", "--agg",     default="first",
                   choices=["first","sum","mean","max","min","count"],
                   help="How to resolve duplicate index/column pairs (default: %(default)s).")
    p.add_argument("-I", "--include-columns",
                   help="Comma-separated extra columns to carry beside the index (each must be constant within an index group).")
    p.set_defaults(handler=_handle_tbl_pivot)
    p.usage_example = "tblkit tbl pivot -i sample_id -c gene -v expr"

    p = subparsers.add_parser("cat", help="Concatenate tables vertically.")
    p.add_argument("files", nargs='*', help="Paths to files to concatenate. First file can be from stdin.")
    p.set_defaults(handler=_handle_tbl_cat)
    p.usage_example = "tblkit tbl cat file1.tsv file2.tsv"

def _setup_row_parsers(subparsers):
    p = subparsers.add_parser("add", help="Insert a new row.")
    p.add_argument("-i", "--row-index", type=int, default=1)
    p.add_argument("-v", "--values", required=True)
    p.set_defaults(handler=_handle_row_add)
    p.usage_example = "tblkit row add -i 2 -v 'val1,val2,val3'"

    p = subparsers.add_parser("drop", help="Delete a row by position.")
    p.add_argument("-i", "--row-index", type=int, required=True)
    p.set_defaults(handler=_handle_row_drop)
    p.usage_example = "tblkit row drop -i 3"

    p = subparsers.add_parser("filter", help="Filter rows on a condition.")
    q_group = p.add_mutually_exclusive_group(required=True)
    q_group.add_argument("--expr", help="Pandas query expression string.")
    q_group.add_argument("-p", "--pattern", help="Regex/text pattern for simple filtering.")
    p.add_argument("-c", "--columns", help="Column for simple pattern filtering.")
    p.add_argument("--word-file", help="File with list of words to match.")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--fixed", action="store_true", help="Treat pattern as literal string.")
    p.set_defaults(handler=_handle_row_filter)
    p.usage_example = "tblkit row filter -c msg -p 'ERR'"

    p = subparsers.add_parser("sample", help="Randomly subsample rows.")
    sample_group = p.add_mutually_exclusive_group(required=True)
    sample_group.add_argument("-n", type=int)
    sample_group.add_argument("-f", type=float)
    p.add_argument("--with-replacement", action="store_true")
    p.add_argument("--seed", type=int)
    p.set_defaults(handler=_handle_row_sample)
    p.usage_example = "tblkit row sample -n 100 --seed 42"
    
    p = subparsers.add_parser("shuffle", help="Randomly shuffle all rows.")
    p.add_argument("--seed", type=int)
    p.set_defaults(handler=_handle_row_shuffle)
    p.usage_example = "tblkit row shuffle --seed 123"

    p = subparsers.add_parser("unique", help="Remove duplicate rows.")
    add_columns_arg(p, required=False)
    p.add_argument("--invert", action="store_true", help="Keep only duplicate rows.")
    p.set_defaults(handler=_handle_row_unique)
    p.usage_example = "tblkit row unique -c user_id"

    p = subparsers.add_parser("head", help="Return the first N rows.")
    p.add_argument("-n", type=int, default=10, help="Number of rows to return.")
    p.set_defaults(handler=_handle_row_head)
    p.usage_example = "tblkit row head -n 5"
    
    p = subparsers.add_parser("tail", help="Return the last N rows.")
    p.add_argument("-n", type=int, default=10, help="Number of rows to return.")
    p.set_defaults(handler=_handle_row_tail)
    p.usage_example = "tblkit row tail -n 5"

def _setup_col_parsers(subparsers):
    p = subparsers.add_parser("rename", help="Rename one or more columns.")
    p.add_argument("--map", required=True, help="Comma-separated map of old:new names.")
    p.set_defaults(handler=_handle_col_rename)
    p.usage_example = "tblkit col rename --map 'c1:id,c2:name'"

    p = subparsers.add_parser("add", help="Add a new column or add a prefix/suffix.")
    add_columns_arg(p)
    action_group = p.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--new-header", help="Name for a new column to be added.")
    action_group.add_argument("--prefix", help="String to add as a prefix.")
    action_group.add_argument("--suffix", help="String to add as a suffix.")
    p.add_argument("-v", "--value", help="Static value for the new column.")
    p.add_argument("-d", "--delimiter", default="", help="Delimiter to use with prefix/suffix.")
    p.set_defaults(handler=_handle_col_add)
    p.usage_example = "tblkit col add -c 1 --new-header status -v active"
    
    p = subparsers.add_parser("bin", help="Discretize a numeric column.")
    add_columns_arg(p)
    bin_group = p.add_mutually_exclusive_group(required=True)
    bin_group.add_argument("--bins", type=int)
    bin_group.add_argument("--qcut", type=int)
    bin_group.add_argument("--breaks")
    p.add_argument("--labels")
    p.add_argument("--keep", action="store_true")
    p.set_defaults(handler=_handle_col_bin)
    p.usage_example = "tblkit col bin -c score --bins 5"

    p = subparsers.add_parser("extract", help="Extract regex capture group(s).")
    add_columns_arg(p)
    add_pattern_args(p)
    p.set_defaults(handler=_handle_col_extract)
    p.usage_example = "tblkit col extract -c url -p 'id=(\\d+)'"
    
    p = subparsers.add_parser("clean-values", help="Clean string values.")
    add_columns_arg(p)
    p.set_defaults(handler=_handle_col_clean_values)
    p.usage_example = "tblkit col clean-values -c category"
    
    p = subparsers.add_parser("drop", help="Drop column(s).")
    add_columns_arg(p)
    p.set_defaults(handler=_handle_col_drop)
    p.usage_example = "tblkit col drop -c 'notes,extra'"

    p = subparsers.add_parser("encode", help="Encode categorical column.")
    add_columns_arg(p)
    add_output_arg(p, default_name=None)
    p.set_defaults(handler=_handle_col_encode)
    p.usage_example = "tblkit col encode -c country"
    
    p = subparsers.add_parser("eval", help="Create column from expression.")
    p.add_argument("--expr", required=True)
    add_output_arg(p)
    add_group_arg(p)
    p.set_defaults(handler=_handle_col_eval)
    p.usage_example = "tblkit col eval --expr 'c3 = c1 * c2'"
    
    p = subparsers.add_parser("fillna", help="Fill missing values.")
    add_columns_arg(p)
    fill_group = p.add_mutually_exclusive_group(required=True)
    fill_group.add_argument("-v", "--value")
    fill_group.add_argument("--method", choices=['ffill', 'bfill', 'mean', 'median'])
    p.set_defaults(handler=_handle_col_fillna)
    p.usage_example = "tblkit col fillna -c score --method mean"

    p = subparsers.add_parser("join", help="Join values from columns.")
    add_columns_arg(p)
    p.add_argument("-d", "--delimiter", default="")
    add_output_arg(p, default_name="joined_column")
    p.add_argument("--keep", action="store_true")
    p.set_defaults(handler=_handle_col_join)
    p.usage_example = "tblkit col join -c 'first,last' -o full_name"

    p = subparsers.add_parser("move", help="Move a column.")
    add_columns_arg(p, help_text="Column to move.")
    p.add_argument("-j", "--dest-column", required=True, help="Destination column.")
    p.add_argument("--position", choices=['before', 'after'], default='before')
    p.set_defaults(handler=_handle_col_move)
    p.usage_example = "tblkit col move -c id -j name --position before"

    p = subparsers.add_parser("replace", help="Replace values in a column.")
    add_columns_arg(p)
    tr_group = p.add_mutually_exclusive_group(required=True)
    tr_group.add_argument("-d", "--dict-file")
    tr_group.add_argument("--from-val")
    p.add_argument("--to-val")
    p.add_argument("--fixed", action="store_true")
    add_output_arg(p, default_name=None)
    p.set_defaults(handler=_handle_col_replace)
    p.usage_example = "tblkit col replace -c country --from USA --to 'United States'"
    
    p = subparsers.add_parser("scale", help="Scale/normalize numeric columns.")
    add_columns_arg(p)
    p.add_argument("--method", choices=['zscore', 'minmax'], required=True)
    add_group_arg(p)
    p.set_defaults(handler=_handle_col_scale)
    p.usage_example = "tblkit col scale -c value --method zscore"

    p = subparsers.add_parser("select", help="Select columns by name or type.")
    add_columns_arg(p, required=False)
    p.add_argument("--type", choices=['any', 'numeric', 'integer', 'string'], default='any', help="Select columns of a specific type.")
    p.add_argument("--invert", action="store_true")
    p.set_defaults(handler=_handle_col_select)
    p.usage_example = "tblkit col select -c 'id,name,date'"
    
    p = subparsers.add_parser("split", help="Split a column.")
    add_columns_arg(p)
    p.add_argument("-d", "--delimiter", required=True)
    p.add_argument("--maxsplit", type=int)
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--keep-na", action="store_true")
    p.set_defaults(handler=_handle_col_split)
    p.usage_example = "tblkit col split -c full_name -d ' '"

    p = subparsers.add_parser("strip", help="Remove pattern from values.")
    add_columns_arg(p)
    add_pattern_args(p)
    add_output_arg(p, default_name=None)
    p.set_defaults(handler=_handle_col_strip)
    p.usage_example = "tblkit col strip -c price -p '[$,]'"
    
    p = subparsers.add_parser("prune", help="Report or drop uninformative columns.")
    p.add_argument("--method", choices=['zero_variance', 'low_variance', 'high_nan', 'high_cardinality', 'correlated'], default='zero_variance')
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(handler=_handle_col_prune)
    p.usage_example = "tblkit col prune --method zero_variance --execute"

    p = subparsers.add_parser("cast", help="Change the data type of a column.")
    add_columns_arg(p)
    p.add_argument("--to", required=True, choices=['int', 'float', 'string', 'bool', 'datetime'])
    p.add_argument("--format", help="Format string for datetime conversion.")
    p.add_argument("--errors", choices=['raise', 'coerce', 'ignore'], default='raise')
    p.set_defaults(handler=_handle_col_cast)
    p.usage_example = "tblkit col cast -c date --to datetime"

def _setup_stat_parsers(subparsers):
    p = subparsers.add_parser("cor", help="Compute pairwise correlation matrix.")
    p.add_argument("--method", choices=['pearson', 'kendall', 'spearman'], default='pearson')
    p.add_argument("--melt", action="store_true")
    p.set_defaults(handler=_handle_stat_cor)
    p.usage_example = "tblkit stat cor --method spearman"
    
    #p = subparsers.add_parser("frequency", help="Get top N frequent values.")
    #add_columns_arg(p)
    #p.add_argument("-T", "--top-n", type=int, default=10)
    #p.set_defaults(handler=_handle_stat_frequency)
    #p.usage_example = "tblkit stat frequency -c country -T 20"

    p = subparsers.add_parser("lm", help="Fit a linear model.")
    p.add_argument("--formula", required=True)
    add_group_arg(p)
    p.set_defaults(handler=_handle_stat_lm)
    p.usage_example = "tblkit stat lm --formula 'y ~ x1 + x2'"

    p = subparsers.add_parser("outlier-row", help="Filter or flag outlier rows.")
    add_columns_arg(p)
    p.add_argument("--method", choices=['iqr', 'zscore'], default='iqr')
    p.add_argument("--factor", type=float, default=1.5)
    p.add_argument("--threshold", type=float, default=3.0)
    p.add_argument("--action", choices=['filter', 'flag'], default='filter')
    p.set_defaults(handler=_handle_stat_outlier_row)
    p.usage_example = "tblkit stat outlier-row -c value"

    p = subparsers.add_parser("pca", help="Perform Principal Component Analysis.")
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--keep-all", action="store_true")
    p.set_defaults(handler=_handle_stat_pca)
    p.usage_example = "tblkit stat pca --n-components 3"

    p = subparsers.add_parser("score", help="Compute signature scores.")
    p.add_argument("--signatures-file", required=True)
    p.add_argument("--method", choices=['mean', 'median', 'normalized_mean'], default='mean')
    p.set_defaults(handler=_handle_stat_score)
    p.usage_example = "tblkit stat score --signatures-file sigs.txt"

    p = subparsers.add_parser("summary", help="Get descriptive statistics.")
    p.set_defaults(handler=_handle_stat_summary)
    p.usage_example = "tblkit stat summary"

def _setup_view_parser(p_view):
    """
    Configure the 'view' command. Keeps the existing direct 'view' handler
    and adds a 'view frequency' subcommand for top-N value counts.
    """

    # If your code defines _handle_view for the direct 'view' command, keep it.
    if "_handle_view" in globals():
        p_view.set_defaults(handler=_handle_view)

    # Add a subparser group under 'view' for extra actions (e.g., frequency).
    view_actions = p_view.add_subparsers(dest="action", title="View Actions")
    view_actions.required = False  # 'view' alone remains valid

    # view frequency
    pvf = view_actions.add_parser(
        "frequency",
        help="Top-N value counts (EDA). Defaults to non-numeric columns unless --columns is given.",
    )
    pvf.add_argument(
        "--columns", "--column-names",
        dest="columns",
        help="Column spec (names, positions, ranges, regex via re:), e.g. 'id,2-5,re:^cat_'",
    )
    pvf.add_argument(
        "-n", "--n", "--top-n",
        dest="top_n",
        type=int,
        default=20,
        help="Top K (default: 20)",
    )
    pvf.set_defaults(handler=_handle_view_frequency)


def _setup_arg_parser():
    import argparse

    parser = CustomArgumentParser(
        prog="tblkit",
        description="A command-line tool for manipulating tabular data.",
        add_help=False,
    )

    # I/O options
    io_opts = parser.add_argument_group("Input/Output Options")
    io_opts.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    io_opts.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    io_opts.add_argument("-f", "--file", type=argparse.FileType('r'), default=sys.stdin, help="Input file (default: stdin).")
    io_opts.add_argument("-s", "--sep", default="\t", help="Field separator for input.")
    io_opts.add_argument("--output-sep", help="Field separator for output (default: same as input).")
    io_opts.add_argument("--encoding", default="utf-8", help="Input file encoding.")
    io_opts.add_argument("--noheader", action="store_true", help="Input has no header.")
    io_opts.add_argument("--comment", help="Character indicating a comment line.")
    io_opts.add_argument("--quotechar", default='"')
    io_opts.add_argument("--escapechar")
    io_opts.add_argument("--doublequote", action=argparse.BooleanOptionalAction, default=True)
    io_opts.add_argument("--na-values", help="Comma-separated list of strings to treat as NA.")
    io_opts.add_argument("--na-rep", default="NA", help="Output representation for NA values.")
    io_opts.add_argument("--on-bad-lines", choices=['error', 'warn', 'skip'], default='error')
    io_opts.add_argument("--stream", action="store_true", help="Enable streaming mode for supported operations.")
    io_opts.add_argument("--quiet", action="store_true", help="Suppress informational messages.")

    # Top-level groups
    subparsers = parser.add_subparsers(
        dest="group",
        title="Command Groups",
        metavar="{tbl,row,col,stat,view,explore}",
    )
    subparsers.required = True

    # Group: tbl / row / col / stat / view
    p_tbl = subparsers.add_parser(
        "tbl", help="Table-level operations",
        description="Commands that operate on the entire table structure.",
        formatter_class=RequiredOptionalHelpFormatter,
    )
    p_row = subparsers.add_parser(
        "row", help="Row-level operations",
        description="Commands for filtering, selecting, and ordering rows.",
        formatter_class=RequiredOptionalHelpFormatter,
    )
    p_col = subparsers.add_parser(
        "col", help="Column-level operations",
        description="Commands for adding, removing, or transforming columns.",
        formatter_class=RequiredOptionalHelpFormatter,
    )
    p_stat = subparsers.add_parser(
        "stat", help="Statistical analysis",
        description="Commands for statistical analysis and summarization.",
        formatter_class=RequiredOptionalHelpFormatter,
    )
    p_view = subparsers.add_parser(
        "view", help="Formatted table preview",
        description="Command to display or pretty-print a formatted preview of the table.",
    )

    # Actions within groups
    tbl_actions  = p_tbl.add_subparsers(dest="action", title="Table Actions")
    row_actions  = p_row.add_subparsers(dest="action", title="Row Actions")
    col_actions  = p_col.add_subparsers(dest="action", title="Column Actions")
    stat_actions = p_stat.add_subparsers(dest="action", title="Analysis Actions")

    _setup_tbl_parsers(tbl_actions)
    _setup_row_parsers(row_actions)
    _setup_col_parsers(col_actions)
    _setup_stat_parsers(stat_actions)
    _setup_view_parser(p_view)     # view is a direct command

    # Explore group (interactive shell)
    _setup_explore_parsers(subparsers)

    return parser

# --------------------------
# Main Execution
# --------------------------
def _handle_header_view_fast(file_handle, sep, comment_char):
    """Optimized function to read only the start of a file for header view."""
    header, first_row = None, None
    sep = codecs.decode(sep, 'unicode_escape')

    for line in file_handle:
        line = line.strip()
        if not line or (comment_char and line.startswith(comment_char)):
            continue
        if header is None:
            header = [h.strip() for h in line.split(sep)]
            continue
        if first_row is None:
            first_row = [c.strip() for c in line.split(sep)]
            break 
    
    if header is None:
        sys.stdout.write("(empty or commented-out input)\n")
        return

    first_row = first_row or [''] * len(header)
    max_header_len = max(len(h) for h in header) if header else 0
    output = [
        f"{i+1:<3} | {h:>{max_header_len}} | {(cell[:37] + '...') if len(cell) > 40 else cell}"
        for i, (h, cell) in enumerate(zip(header, first_row))
    ]
    sys.stdout.write("\n".join(output) + "\n")

def _read_input_data(args: argparse.Namespace, sep: str, header: Optional[int]) -> pd.DataFrame:
    """Reads and parses the input data stream into a pandas DataFrame."""
    na_values = args.na_values.split(',') if args.na_values else None
    engine = 'c' if len(sep) == 1 and not args.comment else 'python'

    try:
        file_to_read = args.file
        if file_to_read is sys.stdin:
            file_to_read = open(sys.stdin.fileno(), mode='r', encoding=args.encoding, errors='replace')

        # Check if file is empty to avoid pandas error
        if file_to_read.seekable():
            if not file_to_read.read(1): return pd.DataFrame()
            file_to_read.seek(0)
            stream_to_read = file_to_read
        else: # Non-seekable stream (like pipe)
            content = file_to_read.read()
            if not content.strip(): return pd.DataFrame()
            stream_to_read = StringIO(content)

        return pd.read_csv(
            stream_to_read, sep=sep, header=header, engine=engine,
            comment=args.comment, quotechar=args.quotechar, escapechar=args.escapechar,
            doublequote=args.doublequote, na_values=na_values, on_bad_lines=args.on_bad_lines
        ).convert_dtypes()
    except Exception as e:
        sys.stderr.write(f"Error reading input data: {e}\n")
        sys.exit(1)

def _write_output(df: pd.DataFrame, sep: str, is_header: bool, na_rep: str, **kwargs):
    """Writes the DataFrame to stdout using the csv module for robustness."""
    if df is None: return
    try:
        writer = csv.writer(
            sys.stdout, delimiter=sep, quotechar=kwargs.get('quotechar', '"'),
            escapechar=kwargs.get('escapechar'), doublequote=kwargs.get('doublequote', True),
            lineterminator='\n', quoting=csv.QUOTE_MINIMAL
        )
        if is_header and not df.empty:
            writer.writerow([str(c) for c in df.columns])
        for row in df.itertuples(index=False, name=None):
            writer.writerow([na_rep if pd.isna(item) else item for item in row])
    except BrokenPipeError:
        try: sys.stdout.close()
        except: pass
    except Exception as e:
        sys.stderr.write(f"Error writing output: {e}\n")
        sys.exit(1)

def _print_group_help(top_parser: argparse.ArgumentParser, group_name: str) -> None:
    """
    Two-line-per-command help for a single group (tbl/row/col/stat/view),
    RIGHT-justified name column, showing short|long flags in args synopsis.
    """
    formatter = top_parser._get_formatter()
    spas = [a for a in top_parser._actions if isinstance(a, argparse._SubParsersAction)]
    if not spas:
        top_parser.print_help(); return
    spa = spas[0]
    gparser = spa.choices.get(group_name)
    if not gparser:
        top_parser.print_help(); return

    action_spas = [a for a in gparser._actions if isinstance(a, argparse._SubParsersAction)]
    if not action_spas:
        gparser.print_help(); return
    sub = action_spas[0]

    try:
        help_map = {ca.dest: (ca.help or '') for ca in sub._choices_actions}
    except Exception:
        help_map = {}

    names = list(sub.choices.keys())
    name_colw = max((len(n) for n in names), default=8) + 2
    buf = []
    hdr = gparser.description or f"{group_name} commands"
    buf.append(hdr)
    buf.append("-" * len(hdr))

    def _opt_repr(a):
        shorts = [s for s in a.option_strings if s.startswith('-') and not s.startswith('--')]
        longs = [s for s in a.option_strings if s.startswith('--')]
        both = shorts + longs
        if not both:
            return (a.metavar or a.dest.upper())
        arg = '' if a.nargs in (0, None) else ' ' + (a.metavar or a.dest.upper())
        return ('|'.join(both)) + arg

    for name in sorted(names):
        p = sub.choices[name]
        desc = help_map.get(name, "") or (p.description or "")

        req, opt = [], []
        for a in p._actions:
            if isinstance(a, argparse._SubParsersAction) or isinstance(a, argparse._HelpAction):
                continue
            if getattr(a, 'option_strings', None) and any(s in ('-h', '--help') for s in a.option_strings):
                continue
            if not a.option_strings:
                req.append(a.metavar or a.dest.upper())
            elif getattr(a, 'required', False):
                req.append(_opt_repr(a))
            else:
                opt.append(_opt_repr(a))

        synopsis_req = ' '.join(req)
        synopsis_opt = ' '.join(opt[:6]) + (' …' if len(opt) > 6 else '')

        buf.append(f"  {name:>{name_colw}}  {desc}")
        if synopsis_req or synopsis_opt:
            buf.append(f"  {'':>{name_colw}}  args: {synopsis_req}{('  ['+synopsis_opt+']') if synopsis_opt else ''}")

    formatter.start_section('')
    formatter.add_text("\n".join(buf))
    formatter.end_section()
    sys.stdout.write(formatter.format_help())

def main():
    """Main entry point for the script."""
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError):
        pass

    parser = _setup_arg_parser()
    _apply_formatter_everywhere(parser, ConsistentActionHelpFormatter)
    _set_metavar_hints_everywhere(parser)
    _ensure_short_flags_everywhere(parser)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Fast-path for `view -H`
    if len(sys.argv) > 1 and sys.argv[1] == 'view' and ('-H' in sys.argv or '--header-view' in sys.argv):
        temp_parser = argparse.ArgumentParser(add_help=False)
        temp_parser.add_argument('-f', '--file', type=argparse.FileType('r'), default=sys.stdin)
        temp_parser.add_argument('-s', '--sep', default='\t')
        temp_parser.add_argument('--comment')
        temp_args, _ = temp_parser.parse_known_args()
        _handle_header_view_fast(temp_args.file, temp_args.sep, temp_args.comment)
        sys.exit(0)

    args = parser.parse_args()

    # If no handler was bound, show focused help for the requested group.
    if not hasattr(args, 'handler') or not args.handler:
        _print_group_help(parser, args.group)
        sys.exit(0)

    # Common IO settings
    input_sep = codecs.decode(args.sep, 'unicode_escape')
    output_sep = codecs.decode(args.output_sep, 'unicode_escape') if args.output_sep else input_sep
    is_header_present = not args.noheader
    header_param = 0 if is_header_present else None

    # Read input once
    df = _read_input_data(args, input_sep, header_param)

    # Special-case: interactive explorer shell (no stdout table output)
    if getattr(args, 'group', None) == 'explore' and getattr(args, 'action', None) == 'shell':
        # Provide standard kwargs; handler will ignore writing and run the shell.
        handler_kwargs = {
            "is_header_present": is_header_present,
            "input_sep": input_sep,
            "output_sep": output_sep,
            "quotechar": args.quotechar,
            "escapechar": args.escapechar,
            "doublequote": args.doublequote,
        }
        # Ensure columns exist even if headerless input
        if not is_header_present and not df.empty:
            df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
        args.handler(df, args, **handler_kwargs)  # starts interactive shell (prompt: "tblkit> ")
        sys.exit(0)

    # Regular (non-interactive) pipeline
    is_additive_op = (getattr(args, 'group', None) == 'row' and getattr(args, 'action', None) == 'add')
    if df.empty and not is_additive_op:
        if len(df.columns) > 0:
            _write_output(
                df, output_sep, is_header_present, args.na_rep,
                quotechar=args.quotechar, escapechar=args.escapechar, doublequote=args.doublequote
            )
        sys.exit(0)

    if not is_header_present and not df.empty:
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]

    try:
        handler_kwargs = {
            "is_header_present": is_header_present,
            "input_sep": input_sep,
            "output_sep": output_sep,
            "quotechar": args.quotechar,
            "escapechar": args.escapechar,
            "doublequote": args.doublequote,
        }
        processed_df = args.handler(df, args, **handler_kwargs)

        output_header_state = is_header_present
        if getattr(args, 'group', None) == 'tbl' and getattr(args, 'action', None) == 'add-header':
            output_header_state = True

        if isinstance(processed_df, pd.DataFrame):
            _write_output(
                processed_df, output_sep, output_header_state, args.na_rep,
                quotechar=args.quotechar, escapechar=args.escapechar, doublequote=args.doublequote
            )

    except (ValueError, IndexError, FileNotFoundError, KeyError, re.error, ImportError) as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"An unexpected error occurred: {type(e).__name__} - {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
