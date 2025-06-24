#!/usr/bin/env python3
import sys
import argparse
import ast
import inspect
import difflib
import os
import re

# ---------------------------
# Existing helper functions

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
    Aligns assignment operators (including compound assignments) for lines that are simple one-line assignments.
    Lines that begin with 'def' or 'class' are ignored.
    
    The regex captures:
      group(1): indentation
      group(2): left-hand side (LHS)
      group(3): assignment operator (compound operators like +=, -=, etc., or plain =)
      group(4): right-hand side (RHS)
      
    Consecutive assignment lines are then aligned so that the operators line up vertically.
    """
    # Order matters: longer compound operators must come before shorter ones.
    assignment_pattern = re.compile(
        r'^(?!\s*(?:def|class)\b)(\s*)(.+?)\s*(\*\*=|//=|>>=|<<=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|=)\s*(.+)$'
    )
    
    lines = code_segment.splitlines()
    aligned_output = []
    group = []  # Will accumulate tuples: (indent, lhs, operator, rhs)

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
# New helper functions for header transformation

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
    • If unfold==True, rewrites a (possibly single-line) header so that each parameter appears on its own line.
    • If unfold==False, folds a multi-line header into a single line.
    Expects header_str to include its trailing colon.
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
        inner_indent = base_indent + "    "  # 4-space indent for parameters.
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

# ---------------------------
# Command functions

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
    # First, use the original AST extraction.
    functions = get_functions_from_code(code)
    if not args.function_name:
        list_functions(args)
        return

    if args.function_name in functions:
        func_code = functions[args.function_name]

        # Apply clean-format if requested.
        if args.clean_format:
            func_code = remove_comments(func_code)
            func_code = _align_assignments(func_code)
            
        # If header transformation (--fold or --unfold) is requested, transform only the header.
        if args.fold or args.unfold:
            # Try to match the header (which may span multiple lines)
            # The pattern matches from the beginning up to the colon after the header.
            m = re.match(r'^(\s*def\s+[A-Za-z_]\w*\s*\(.*?\))\s*:(.*)$', func_code, re.DOTALL)
            if m:
                header_part = m.group(1) + ":"
                body_part = m.group(2)
                # If both options are specified, report error.
                if args.fold and args.unfold:
                    sys.stderr.write("Cannot specify both --fold and --unfold.\n")
                    sys.exit(1)
                # Transform the header.
                transformed_header = transform_func_header(header_part, unfold=args.unfold)
                func_code = transformed_header + body_part
            else:
                # Fallback: transform the entire function code (may result in only header changes)
                func_code = transform_func_header(func_code, unfold=args.unfold)
                
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

    else:  # Default to adding at the top
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

def main():
    parser = argparse.ArgumentParser(
        description="A command-line tool to manage Python functions.",
        add_help=False)
    parser.add_argument("-f", "--file",
                        help="Specify a Python file to operate on. If not provided, reads from stdin.")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    list_parser = subparsers.add_parser('list', help='List all functions in the input.')
    list_parser.set_defaults(func=list_functions)

    view_parser = subparsers.add_parser('view', help='View a specific function. Lists all if no name. -c to exclude comments and align assignments.')
    view_parser.add_argument("function_name", nargs='?', help="Name of the function to view.")
    view_parser.add_argument("-c", "--clean-format", action="store_true", help="Remove comments and align assignment operators.")
    view_parser.add_argument("--fold", action="store_true", help="Fold the function header into a single line.")
    view_parser.add_argument("--unfold", action="store_true", help="Unfold the function header (one argument per line).")
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
