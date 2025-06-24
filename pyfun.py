#!/usr/bin/env python3
import sys
import argparse
import ast
import difflib
import os
import re

# ---------------------------
# Helper functions

def get_functions_from_code(code):
    """
    Attempts to extract functions from code.
    First tries using AST; if that fails (e.g. because of a syntax error),
    then falls back to a regex-based extraction (which may only return the header).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback using regex extraction (only header is returned)
        pattern = re.compile(
            r'(^\s*def\s+([A-Za-z_]\w*)\s*\(.*?\))\s*:',
            re.DOTALL | re.MULTILINE
        )
        functions = {}
        for match in pattern.finditer(code):
            func_name = match.group(2)
            functions[func_name] = match.group(0)
        return functions

    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            src = ast.get_source_segment(code, node)
            functions[node.name] = src
    return functions

def remove_comments(source):
    lines = source.splitlines()
    cleaned_lines = []
    in_multiline_comment = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            if in_multiline_comment:
                in_multiline_comment = False
                continue
            else:
                if (stripped_line.count('"""') % 2 != 0 or 
                    stripped_line.count("'''") % 2 != 0):
                    in_multiline_comment = True
                    continue
        if in_multiline_comment:
            continue
        if '#' in line:
            line = line.split('#', 1)[0]
        if line.strip():
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def _align_assignments(code_segment):
    """
    Aligns assignment operators (including compound assignments) for lines that are simple one‑line assignments.
    Lines that begin with "def" or "class" are ignored.
    """
    assignment_pattern = re.compile(
        r'^(?!\s*(?:def|class)\b)(\s*)(.+?)\s*(\*\*=|//=|>>=|<<=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|=)\s*(.+)$'
    )
    lines = code_segment.splitlines()
    aligned_output = []
    group = []  # Accumulates tuples (indent, lhs, operator, rhs).

    def process_group():
        nonlocal group, aligned_output
        if not group:
            return
        max_lhs = max(len(lhs) for indent, lhs, op, rhs in group)
        for indent, lhs, op, rhs in group:
            aligned_output.append(f"{indent}{lhs.ljust(max_lhs)} {op} {rhs}")
        group.clear()

    for line in lines:
        match = assignment_pattern.match(line)
        if match:
            group.append(match.groups())
        else:
            process_group()
            aligned_output.append(line)
    process_group()
    return "\n".join(aligned_output)

def get_code_from_input(args):
    if args.file:
        with open(args.file, 'r') as f:
            return f.read()
    else:
        return sys.stdin.read()

# ---------------------------
# New helper functions for function header transformation

def split_arguments(param_str):
    """
    Splits a function's parameter string into individual arguments,
    taking care of nested delimiters and quoted strings.
    """
    args_list = []
    current = ""
    depth = 0
    in_string = False
    string_char = ""
    escape = False
    for ch in param_str:
        if escape:
            current += ch
            escape = False
            continue
        if ch == "\\":
            current += ch
            escape = True
            continue
        if in_string:
            current += ch
            if ch == string_char:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            current += ch
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args_list.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args_list.append(current.strip())
    return args_list

def transform_func_header(header_str, unfold=True):
    """
    Transforms a function header.
    If unfold==True, rewrites a (possibly single‑line) function definition header
    so that each parameter appears on its own line.
    If unfold==False, folds a multi‑line header into one line.
    Expects header_str to include the trailing colon.
    """
    header_body = header_str.strip()
    if header_body.endswith(':'):
        header_body = header_body[:-1].rstrip()
    i1 = header_body.find('(')
    i2 = header_body.rfind(')')
    if i1 == -1 or i2 == -1 or i2 < i1:
        return header_str  # Not a standard header.
    prefix = header_body[:i1].rstrip()  # e.g. "def func"
    params_str = header_body[i1+1:i2].strip()
    indent_match = re.match(r'^(\s*)', header_str)
    base_indent = indent_match.group(1) if indent_match else ""
    
    if unfold:
        if not params_str:
            return f"{prefix}():"
        params = split_arguments(params_str)
        new_lines = [f"{prefix}("]
        inner_indent = base_indent + "    "  # 4-space indent.
        for param in params:
            if param:
                new_lines.append(f"{inner_indent}{param},")
        new_lines.append(f"{base_indent}):")
        return "\n".join(new_lines)
    else:
        # Fold: join parameters with a comma and a space.
        if params_str:
            params = split_arguments(params_str)
            new_params = ", ".join(p.strip() for p in params if p.strip())
        else:
            new_params = ""
        return f"{prefix}({new_params}):"

def transform_function_definitions(code, target_func=None, unfold=True):
    """
    Transforms function headers in the given code using regex.
    If target_func is provided, only that function's header is transformed;
    otherwise, all headers are transformed.
    """
    pattern = re.compile(
        r'(^\s*def\s+([A-Za-z_]\w*)\s*\(.*?\))\s*:',
        re.DOTALL | re.MULTILINE
    )
    
    def replacement(match):
        func_name = match.group(2)
        orig_header = match.group(0)  # Includes the colon.
        if target_func and func_name != target_func:
            return orig_header
        return transform_func_header(orig_header, unfold)
    
    return pattern.sub(replacement, code)

def extract_function_regex(code, func_name):
    """
    Fallback extraction of a full function definition (header and body)
    using regex. Assumes that the function body lines are indented.
    """
    pattern = re.compile(
        rf'(^\s*def\s+{re.escape(func_name)}\s*\(.*?\):)(\n(?:\s+.+)*)',
        re.MULTILINE
    )
    match = pattern.search(code)
    if match:
        return match.group(0)
    return None


# ---------------------------
# Command functions

def list_functions(args):
    code = get_code_from_input(args)
    try:
        funcs = get_functions_from_code(code)
    except SyntaxError as se:
        sys.stderr.write(f"SyntaxError while parsing file: {se}\n")
        sys.exit(1)
    if funcs:
        sys.stdout.write("Available functions:\n")
        for name in sorted(funcs.keys()):
            sys.stdout.write(f"- {name}\n")
    else:
        sys.stderr.write("No functions found.\n")

def view_function(args):
    code = get_code_from_input(args)
    # Apply clean formatting if requested.
    if args.clean_format:
        code = remove_comments(code)
        code = _align_assignments(code)
    # If header transformation is requested, use regex-based transformation.
    if args.unfold or args.fold:
        if args.unfold and args.fold:
            sys.stderr.write("Cannot specify both --unfold and --fold.\n")
            sys.exit(1)
        if args.function_name:
            pattern = re.compile(
                r'^\s*def\s+%s\s*\(.*?\)\s*:' % re.escape(args.function_name),
                re.DOTALL | re.MULTILINE
            )
            if not pattern.search(code):
                sys.stderr.write(f"Function '{args.function_name}' not found.\n")
                sys.exit(1)
        transform_flag = args.unfold  # True means unfold; False means fold.
        transformed_code = transform_function_definitions(code,
                                                            target_func=args.function_name,
                                                            unfold=transform_flag)
        print(transformed_code)
    else:
        # Use AST-based extraction.
        try:
            funcs = get_functions_from_code(code)
        except SyntaxError as se:
            sys.stderr.write(f"SyntaxError while parsing file: {se}\n")
            sys.exit(1)
        if args.function_name:
            if args.function_name in funcs:
                result = funcs[args.function_name]
                # If the result appears to be only a header, try regex-based fallback.
                if '\n' not in result:
                    fallback = extract_function_regex(code, args.function_name)
                    if fallback:
                        result = fallback
                print(result)
            else:
                sys.stderr.write(f"Function '{args.function_name}' not found.\n")
        else:
            list_functions(args)

def remove_function(args):
    code = get_code_from_input(args)
    funcs = get_functions_from_code(code)
    if args.function_name not in funcs:
        sys.stderr.write(f"Function '{args.function_name}' not found.\n")
        return
    tree = ast.parse(code)
    new_nodes = []
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == args.function_name):
            new_nodes.append(ast.unparse(node))
    sys.stdout.write("\n".join(new_nodes))

def add_function(args):
    original_code = get_code_from_input(args)
    with open(args.file_to_add, 'r') as f:
        new_function_code = f.read()
    if args.after:
        if args.after not in get_functions_from_code(original_code):
            sys.stderr.write(f"Function '{args.after}' not found to add after.\n")
            return
        lines = original_code.splitlines()
        new_lines = []
        added = False
        for line in lines:
            new_lines.append(line)
            if f"def {args.after}" in line:
                indentation = len(line) - len(line.lstrip())
                for i, func_line in enumerate(new_function_code.splitlines()):
                    if i == 0:
                        new_lines.append(func_line)
                    else:
                        new_lines.append(" " * indentation + func_line)
                added = True
        sys.stdout.write("\n".join(new_lines))
    elif args.before:
        if args.before not in get_functions_from_code(original_code):
            sys.stderr.write(f"Function '{args.before}' not found to add before.\n")
            return
        lines = original_code.splitlines()
        new_lines = []
        added = False
        for line in lines:
            if f"def {args.before}" in line and not added:
                indentation = len(line) - len(line.lstrip())
                for i, func_line in enumerate(new_function_code.splitlines()):
                    if i == 0:
                        new_lines.append(func_line)
                    else:
                        new_lines.append(" " * indentation + func_line)
                added = True
            new_lines.append(line)
        sys.stdout.write("\n".join(new_lines))
    else:
        sys.stdout.write(new_function_code + "\n\n" + original_code)

def diff_functions(args):
    code = get_code_from_input(args)
    funcs = get_functions_from_code(code)
    if args.function1 not in funcs:
        sys.stderr.write(f"Function '{args.function1}' not found.\n")
        return
    if args.function2 not in funcs:
        sys.stderr.write(f"Function '{args.function2}' not found.\n")
        return
    func1_code = funcs[args.function1].splitlines()
    func2_code = funcs[args.function2].splitlines()
    differ = difflib.UnifiedDiff()
    diff = differ.compare(func1_code, func2_code,
                          fromfile=args.function1,
                          tofile=args.function2)
    sys.stdout.writelines(diff)

def list_dependencies(args):
    code = get_code_from_input(args)
    try:
        tree = ast.parse(code)
    except SyntaxError as se:
        sys.stderr.write(f"SyntaxError while parsing file: {se}\n")
        sys.exit(1)
    funcs = get_functions_from_code(code)
    if args.function_name not in funcs:
        sys.stderr.write(f"Function '{args.function_name}' not found.\n")
        return
    dependencies = set()
    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in funcs:
                    dependencies.add(func_name)
            self.generic_visit(node)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == args.function_name:
            CallVisitor().visit(node)
            break
    sys.stdout.write("Dependencies:\n")
    for dep in sorted(dependencies):
        sys.stdout.write(f"- {dep}\n")

# ---------------------------
# Main command-line parsing

def main():
    parser = argparse.ArgumentParser(
        description="A command-line tool to manage Python functions.",
        add_help=False
    )
    parser.add_argument("-f", "--file",
                        help="Specify a Python file to operate on. If not provided, reads from stdin.")
    subparsers = parser.add_subparsers(dest='command',
                                       help='Available commands')

    list_parser = subparsers.add_parser('list',
                                        help='List all functions in the input.')
    list_parser.set_defaults(func=list_functions)

    view_parser = subparsers.add_parser(
        'view',
        help='View a specific function or full code. Use -c to clean (remove comments & align assignments) and --unfold/--fold to transform header.'
    )
    view_parser.add_argument("function_name", nargs="?",
                             help="Name of the function to view.")
    view_parser.add_argument("-c", "--clean-format", action="store_true",
                             help="Remove comments and align assignments.")
    view_parser.add_argument("--unfold", action="store_true",
                             help="Unfold the function header (one parameter per line).")
    view_parser.add_argument("--fold", action="store_true",
                             help="Fold the function header into a single line.")
    view_parser.set_defaults(func=view_function)

    remove_parser = subparsers.add_parser('remove',
                                           help='Remove a function.')
    remove_parser.add_argument("function_name",
                               help="Name of the function to remove.")
    remove_parser.set_defaults(func=remove_function)

    add_parser = subparsers.add_parser('add',
                                        help='Add functions from a file. Default: first.')
    add_parser.add_argument("file_to_add",
                            help="Path to the file containing the function(s) to add.")
    add_parser.add_argument("-b", "--before",
                            help="Add the function(s) before this function.")
    add_parser.add_argument("-a", "--after",
                            help="Add the function(s) after this function.")
    add_parser.set_defaults(func=add_function)

    diff_parser = subparsers.add_parser('diff',
                                         help='Show diff between two functions.')
    diff_parser.add_argument("function1",
                             help="Name of the first function.")
    diff_parser.add_argument("function2",
                             help="Name of the second function.")
    diff_parser.set_defaults(func=diff_functions)

    deps_parser = subparsers.add_parser('deps',
                                         help='List dependencies of a function.')
    deps_parser.add_argument("function_name",
                             help="Name of the function to list dependencies for.")
    deps_parser.set_defaults(func=list_dependencies)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
