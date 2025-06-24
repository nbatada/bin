#!/usr/bin/env python3

import sys
import argparse
import ast
import inspect
import difflib
import os
import re

# ---------------------------
# Existing functions here...
# (get_functions_from_code, remove_comments, _align_assignments, etc.)

def get_functions_from_code(code):
    tree = ast.parse(code)
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = ast.get_source_segment(code, node)
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
                if stripped_line.count('"""') % 2 != 0 or stripped_line.count("'''") % 2 != 0:
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
    Aligns assignment operators (including compound assignments) for lines that 
    are simple one-line assignments. Lines that begin with 'def' or 'class' are ignored.
    """
    # Order matters: longer compound operators must come before shorter ones.
    assignment_pattern = re.compile(
        r'^(?!\s*(?:def|class)\b)(\s*)(.+?)\s*(\*\*=|//=|>>=|<<=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|=)\s*(.+)$'
    )
    
    lines = code_segment.splitlines()
    aligned_output = []
    group = []  # Accumulates tuples (indent, lhs, operator, rhs) for consecutive assignments.

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

def list_functions(args):
    code = get_code_from_input(args)
    functions = get_functions_from_code(code)
    if functions:
        print("Available functions:")
        for name in sorted(functions.keys()):
            print(f"- {name}")
    else:
        print("No functions found.", file=sys.stderr)

def view_function(args):
    code = get_code_from_input(args)
    functions = get_functions_from_code(code)

    if not args.function_name:
        list_functions(args)
        return

    if args.function_name in functions:
        func_code = functions[args.function_name]

        if args.clean_format:
            func_code = remove_comments(func_code)
            func_code = _align_assignments(func_code)
        print(func_code)
    else:
        print(f"Function '{args.function_name}' not found.", file=sys.stderr)

def remove_function(args):
    code = get_code_from_input(args)
    functions = get_functions_from_code(code)
    if args.function_name not in functions:
        print(f"Function '{args.function_name}' not found.", file=sys.stderr)
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
            print(f"Function '{args.after}' not found to add after.", file=sys.stderr)
            return

        lines = original_code.splitlines()
        new_lines = []
        added = False
        for line in lines:
            new_lines.append(line)
            if f"def {args.after}" in line:
                indentation = len(line) - len(line.lstrip())
                new_function_lines = new_function_code.splitlines()
                for i, func_line in enumerate(new_function_lines):
                    if i == 0:
                        new_lines.append(func_line)
                    else:
                        new_lines.append(" " * indentation + func_line)
                added = True
        sys.stdout.write("\n".join(new_lines))

    elif args.before:
        if args.before not in get_functions_from_code(original_code):
            print(f"Function '{args.before}' not found to add before.", file=sys.stderr)
            return

        lines = original_code.splitlines()
        new_lines = []
        added = False
        for line in lines:
            if f"def {args.before}" in line and not added:
                indentation = len(line) - len(line.lstrip())
                new_function_lines = new_function_code.splitlines()
                for i, func_line in enumerate(new_function_lines):
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
    functions = get_functions_from_code(code)

    if args.function1 not in functions:
        print(f"Function '{args.function1}' not found.", file=sys.stderr)
        return
    if args.function2 not in functions:
        print(f"Function '{args.function2}' not found.", file=sys.stderr)
        return

    func1_code = functions[args.function1].splitlines()
    func2_code = functions[args.function2].splitlines()

    differ = difflib.UnifiedDiff()
    diff = differ.compare(func1_code, func2_code, fromfile=args.function1, tofile=args.function2)
    sys.stdout.writelines(diff)

def list_dependencies(args):
    code = get_code_from_input(args)
    tree = ast.parse(code)

    if args.function_name not in get_functions_from_code(code):
        print(f"Function '{args.function_name}' not found.", file=sys.stderr)
        return

    dependencies = set()

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in get_functions_from_code(code):
                    dependencies.add(func_name)
            self.generic_visit(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == args.function_name:
            visitor = CallVisitor()
            visitor.visit(node)
            break

    print("Dependencies:")
    for dep in sorted(list(dependencies)):
        print(f"- {dep}")

# ---------------------------
# New helper functions for unfolding/folding

def split_arguments(param_str):
    """
    Splits a function’s parameter string into individual arguments,
    taking into account nested delimiters and string literals.
    """
    args = []
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
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())
    return args

def transform_func_header(header_str, unfold=True):
    """
    Transforms a function header.
    If unfold==True, rewrites a (possibly single-line) function definition header
    so that each argument appears on its own line.
    If unfold==False (i.e. --fold was requested), then folds a multi-line header into one.
    Expects header_str to include the final colon.
    """
    # Remove extra whitespace and the trailing colon.
    header_body = header_str.strip()
    if header_body.endswith(':'):
        header_body = header_body[:-1].rstrip()
    # Find the opening and closing parentheses.
    i1 = header_body.find('(')
    i2 = header_body.rfind(')')
    if i1 == -1 or i2 == -1 or i2 < i1:
        return header_str  # Not a standard function header.
    prefix = header_body[:i1].rstrip()  # e.g. "def func"
    params_str = header_body[i1+1:i2].strip()
    # Determine the base indentation.
    indent_match = re.match(r'^(\s*)', header_str)
    base_indent = indent_match.group(1) if indent_match else ""
    
    if unfold:
        # If there are no parameters, return as is.
        if not params_str:
            return f"{prefix}():"
        params = split_arguments(params_str)
        new_lines = []
        new_lines.append(f"{prefix}(")
        inner_indent = base_indent + "    "  # indent inner args by 4 spaces.
        for param in params:
            if param:
                new_lines.append(f"{inner_indent}{param},")
        new_lines.append(f"{base_indent}):")
        return "\n".join(new_lines)
    else:
        # Fold: collapse newlines and extra spaces.
        if params_str:
            params = split_arguments(params_str)
            new_params_str = ", ".join(param.strip() for param in params if param.strip())
        else:
            new_params_str = ""
        folded_header = f"{prefix}({new_params_str}):"
        return base_indent + folded_header

def transform_function_definitions(code, target_func=None, unfold=True):
    """
    Searches the given code for function definitions and transforms their header
    by either unfolding (each parameter on a separate line) or folding (collapsing into one line).
    If target_func is provided, only that function is transformed.
    """
    # This pattern matches from the "def" at the start until the colon that ends the signature.
    pattern = re.compile(r'(^\s*def\s+([A-Za-z_]\w*)\s*\(.*?\))\s*:', re.DOTALL | re.MULTILINE)
    
    def replacement(match):
        func_name = match.group(2)
        original_header = match.group(0)  # includes the colon at the end.
        if target_func and func_name != target_func:
            return original_header
        return transform_func_header(original_header, unfold)
    
    new_code = pattern.sub(replacement, code)
    return new_code

# ---------------------------
# New command "unfold" (which also supports folding via --fold)

def transform_definitions(args):
    code = get_code_from_input(args)
    # Optionally apply clean formatting first.
    if args.clean_format:
        code = remove_comments(code)
        code = _align_assignments(code)
    target = args.function_name if hasattr(args, 'function_name') and args.function_name else None
    # By default, we unfold (i.e. one argument per line). If --fold is provided, do the opposite.
    new_code = transform_function_definitions(code, target_func=target, unfold=(not args.fold))
    print(new_code)

# ---------------------------
# main() and command-line argument parsing

def main():
    parser = argparse.ArgumentParser(description="A command-line tool to manage Python functions.", add_help=False)
    parser.add_argument("-f", "--file", help="Specify a Python file to operate on. If not provided, reads from stdin.")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    list_parser = subparsers.add_parser('list', help='List all functions in the input.')
    list_parser.set_defaults(func=list_functions)

    view_parser = subparsers.add_parser('view', help='View a specific function. Lists all if no name. -c to remove comments and align assignments.')
    view_parser.add_argument("function_name", nargs='?', help="Name of the function to view.")
    view_parser.add_argument("-c", "--clean-format", action="store_true", help="Remove comments, align assignment operators, and (if desired) unfold the header.")
    view_parser.set_defaults(func=view_function)

    remove_parser = subparsers.add_parser('remove', help='Remove a function.')
    remove_parser.add_argument("function_name", help="Name of the function to remove.")
    remove_parser.set_defaults(func=remove_function)

    add_parser = subparsers.add_parser('add', help='Add functions from a file. Default: first.')
    add_parser.add_argument("file_to_add", help="Path to the file containing the function(s) to add.")
    add_parser.add_argument("-b", "--before", help="Add the function(s) before this function.")
    add_parser.add_argument("-a", "--after", help="Add the function(s) after this function.")
    add_parser.set_defaults(func=add_function)

    diff_parser = subparsers.add_parser('diff', help='Show diff between two functions.')
    diff_parser.add_argument("function1", help="Name of the first function.")
    diff_parser.add_argument("function2", help="Name of the second function.")
    diff_parser.set_defaults(func=diff_functions)

    deps_parser = subparsers.add_parser('deps', help='List dependencies of a function.')
    deps_parser.add_argument("function_name", help="Name of the function to list dependencies for.")
    deps_parser.set_defaults(func=list_dependencies)

    # New "unfold" command: by default, it unfolds (i.e. splits parameters onto separate lines).
    # Use --fold to collapse a multi-line header into a single line.
    unfold_parser = subparsers.add_parser('unfold', help='Reformat function definitions to improve header readability. '
                                                         'By default, unfolds a single-line definition (one parameter per line). '
                                                         'Use --fold to collapse an unfolded header into a single line. '
                                                         'If a function name is provided, only that function is transformed.')
    unfold_parser.add_argument("function_name", nargs="?", help="Name of the function to transform. "
                                                                "If omitted, the transformation is applied to the entire code.")
    unfold_parser.add_argument("--fold", action="store_true", help="Fold multi-line function definitions into one line.")
    unfold_parser.add_argument("-c", "--clean-format", action="store_true", help="Also clean the code (remove comments & align assignments) before transforming.")
    unfold_parser.set_defaults(func=transform_definitions)

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
