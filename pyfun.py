#!/usr/bin/env python3                                                                                                                              

import sys
import argparse
import ast
import inspect
import difflib
import os

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

##-
import re

def _align_assignments(code_segment):
    """
    Aligns assignment operators ('=') for lines that are simple one-line assignments.
    This version uses a regex that captures any left-hand side (LHS) up to the '=' 
    (allowing for attribute access, indices, etc.) while skipping function/class 
    definitions.
    """
    # This pattern ignores lines starting with "def" or "class" and captures:
    #   group(1): leading whitespace (indentation)
    #   group(2): the left-hand side (non-greedily, up to the first " =")
    #   group(3): everything after the '='
    assignment_pattern = re.compile(r'^(?!\s*(?:def|class)\b)(\s*)(.+?)\s*=\s*(.+)$')
    
    lines = code_segment.splitlines()
    aligned_output = []
    group = []  # will accumulate tuples: (indent, lhs, rhs)

    def process_group():
        nonlocal group, aligned_output
        if not group:
            return
        # Determine the maximum length of the LHS in this consecutive group.
        max_lhs = max(len(lhs) for indent, lhs, rhs in group)
        for indent, lhs, rhs in group:
            # Add one extra space so that even the longest lhs gets a trailing space.
            padding = max_lhs - len(lhs) + 1
            new_line = f"{indent}{lhs}{' ' * padding}= {rhs}"
            aligned_output.append(new_line)
        group = []  # reset the group

    for line in lines:
        # Try to match assignment lines that are not function or class definitions.
        match = assignment_pattern.match(line)
        if match:
            group.append(match.groups())
        else:
            # If a non-assignment line is encountered,
            # flush any accumulated assignment group.
            process_group()
            aligned_output.append(line)
    process_group()  # Flush any remaining group

    return "\n".join(aligned_output)

##-

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

    else: # Default to first                                                                                                                        
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
    parser = argparse.ArgumentParser(description="A command-line tool to manage Python functions.", add_help=False)
    parser.add_argument("-f", "--file", help="Specify a Python file to operate on. If not provided, reads from stdin.")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    list_parser = subparsers.add_parser('list', help='List all functions in the input.')
    list_parser.set_defaults(func=list_functions)

    view_parser = subparsers.add_parser('view', help='View a specific function. Lists all if no name. -c to exclude comments and align assignments.\
')
    view_parser.add_argument("function_name", nargs='?', help="Name of the function to view.")
    view_parser.add_argument("-c", "--clean-format", action="store_true", help="Remove comments and align assignment operators.")
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
