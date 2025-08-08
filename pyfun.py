#!/usr/bin/env python3
# VERSION=3
VERSION=4
import json
import argparse
import ast
import os
import re
import subprocess
import difflib

# Helper functions for parsing and file operations
def get_function_source(file_path, function_name):
    """Finds and returns the source code of a function."""
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            source_lines = open(file_path).readlines()
            start_line = node.lineno - 1
            end_line = node.end_lineno
            return ''.join(source_lines[start_line:end_line])
    return None

def find_function_node(file_path, function_name):
    """Finds and returns the AST node of a function."""
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    return None

def find_function_and_calls(file_path, function_name):
    """Finds function definition and all its call sites."""
    with open(file_path, 'r') as f:
        source = f.read()
    tree = ast.parse(source)
    calls = []
    function_def_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_def_node = node
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == function_name:
                calls.append(node)
    return function_def_node, calls, source

# --- Commands ---

def jup2py_command(args):
    """Converts a Jupyter notebook to a Python script."""
    notebook_path = args.file
    hide_non_code = args.hide_non_code
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            if cell['source']:
                print(''.join(cell['source']))
        elif not hide_non_code:
            if cell['source']:
                print(''.join([f"#{line}" for line in cell['source']]))

def py2jup_command(args):
    """Converts a Python script to a Jupyter notebook."""
    script_path = args.file
    single_cell = args.single_cell
    
    with open(script_path, 'r') as f:
        source = f.read()
    
    notebook = {
        "cells": [],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    if single_cell:
        notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)})
    else:
        tree = ast.parse(source)
        lines = source.splitlines(True)
        last_line_end = 0

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.lineno > last_line_end:
                    pre_function_code = ''.join(lines[last_line_end:node.lineno-1])
                    if pre_function_code.strip():
                         notebook["cells"].append({"cell_type": "code", "source": pre_function_code.splitlines(True)})
                
                notebook["cells"].append({"cell_type": "markdown", "source": [f"# {node.name}\n"]})
                function_source = ''.join(lines[node.lineno-1:node.end_lineno])
                notebook["cells"].append({"cell_type": "code", "source": function_source.splitlines(True)})
                last_line_end = node.end_lineno
        
        if last_line_end < len(lines):
            remaining_code = ''.join(lines[last_line_end:])
            if remaining_code.strip():
                notebook["cells"].append({"cell_type": "code", "source": remaining_code.splitlines(True)})

    with open(args.output, 'w') as f:
        json.dump(notebook, f, indent=4)

def list_command(args):
    """Lists functions in a Python script."""
    with open(args.file, 'r') as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not args.search or args.search in node.name:
                print(f"- {node.name}")

def explain_command(args):
    """Prints the source code of a function, including comments."""
    source_code = get_function_source(args.file, args.function_name)
    if source_code:
        print(f"Explanation for function '{args.function_name}':\n")
        print(source_code)
    else:
        print(f"Error: Function '{args.function_name}' not found.")

def diff_command(args):
    """Provides a friendly diff between a function in the script and a file."""
    try:
        source_code_old = get_function_source(args.file, args.function_name)
        if not source_code_old:
            raise ValueError(f"Function '{args.function_name}' not found in script.")
        
        with open(args.new_file, 'r') as f:
            source_code_new = f.read()

        diff = difflib.unified_diff(
            source_code_old.splitlines(keepends=True),
            source_code_new.splitlines(keepends=True),
            fromfile=f'--- Existing function {args.function_name}',
            tofile=f'+++ New function from {args.new_file}',
            n=3
        )
        for line in diff:
            if line.startswith('---') or line.startswith('+++'):
                print(line, end='')
            elif line.startswith('-'):
                print(f"Removed: {line[1:]}", end='')
            elif line.startswith('+'):
                print(f"Added: {line[1:]}", end='')
            else:
                print(f"Unchanged: {line[1:]}", end='')

    except Exception as e:
        print(f"Error: {e}")

def add_command(args):
    """Adds a function from a file to the script."""
    existing_node = find_function_node(args.file, args.function_name)
    if existing_node:
        print(f"Warning: Function '{args.function_name}' already exists.")
        response = input("Do you want to proceed and add it anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return

    with open(args.new_file, 'r') as f:
        new_source = f.read()
    
    with open(args.file, 'a') as f:
        f.write(f"\n{new_source}")
    print(f"Function '{args.function_name}' added to {args.file}.")

def replace_command(args):
    """Replaces a function in the script with one from a file."""
    with open(args.file, 'r') as f:
        lines = f.readlines()
    existing_node = find_function_node(args.file, args.function_name)
    if not existing_node:
        print(f"Error: Function '{args.function_name}' not found.")
        return

    with open(args.new_file, 'r') as f:
        new_source = f.read()

    new_lines = lines[:existing_node.lineno - 1] + new_source.splitlines(True) + lines[existing_node.end_lineno:]
    
    with open(args.file, 'w') as f:
        f.writelines(new_lines)
    print(f"Function '{args.function_name}' replaced in {args.file}.")

def rename_command(args):
    """Renames a function and all its call sites."""
    func_node, call_nodes, source = find_function_and_calls(args.file, args.old_name)
    if not func_node:
        print(f"Error: Function '{args.old_name}' not found.")
        return

    new_source = source.replace(f"def {args.old_name}", f"def {args.new_name}")
    new_source = new_source.replace(f"{args.old_name}(", f"{args.new_name}(")

    with open(args.file, 'w') as f:
        f.write(new_source)
    print(f"Function '{args.old_name}' and all its calls renamed to '{args.new_name}'.")

def clean_command(args):
    """Reformat a function or the entire script."""
    try:
        subprocess.run(['black', '--version'], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: 'black' formatter not found. Please install it with 'pip install black'.")
        return
    
    command = ['black', args.file]
    if args.function_name:
        print("Note: 'black' reformats the entire file, not just a single function.")
    
    try:
        subprocess.run(command, check=True)
        print(f"Successfully formatted {args.file} with black.")
    except subprocess.CalledProcessError as e:
        print(f"Error during formatting: {e}")

def dependency_command(args):
    """Generates a Graphviz diagram of function dependencies."""
    try:
        subprocess.run(['dot', '-V'], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: 'dot' from Graphviz not found. Please install it on your system.")
        return
    
    dependencies = {}
    with open(args.file, 'r') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            dependencies[func_name] = []
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                    called_func_name = sub_node.func.id
                    if called_func_name != func_name and called_func_name not in dependencies[func_name]:
                        dependencies[func_name].append(called_func_name)
    
    dot_source = "digraph G {\n"
    for func, deps in dependencies.items():
        for dep in deps:
            dot_source += f'    "{func}" -> "{dep}";\n'
    dot_source += "}"

    process = subprocess.Popen(['dot', '-Tsvg'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output, _ = process.communicate(dot_source)
    
    with open(args.output, 'w') as f:
        f.write(output)
    
    print(f"Dependency graph saved to {args.output}")

def remove_command(args):
    """Removes a function from a Python script."""
    with open(args.file, 'r') as f:
        lines = f.readlines()
    existing_node = find_function_node(args.file, args.function_name)
    if not existing_node:
        print(f"Error: Function '{args.function_name}' not found.")
        return
    
    start_line = existing_node.lineno - 1
    end_line = existing_node.end_lineno
    new_lines = lines[:start_line] + lines[end_line:]
    
    with open(args.file, 'w') as f:
        f.writelines(new_lines)
    print(f"Function '{args.function_name}' removed from {args.file}.")

def view_command(args):
    """Views a single function with various formatting options."""
    source = get_function_source(args.file, args.function_name)
    if not source:
        print(f"Error: Function '{args.function_name}' not found.")
        return

    lines = source.splitlines()

    if args.no_comments:
        lines = [re.sub(r'(?<![("])#.*$', '', line).rstrip() for line in lines]
        lines = [line for line in lines if line.strip()]

    if args.clean:
        max_equal_pos = 0
        for line in lines:
            if '=' in line and '==' not in line and '>=' not in line and '<=' not in line:
                pos = line.find('=')
                max_equal_pos = max(max_equal_pos, pos)
        
        if max_equal_pos > 0:
            new_lines = []
            for line in lines:
                if '=' in line and '==' not in line and '>=' not in line and '<=' not in line:
                    pos = line.find('=')
                    new_lines.append(line[:pos].rstrip().ljust(max_equal_pos) + ' ' + line[pos:])
                else:
                    new_lines.append(line)
            lines = new_lines

    if args.unfold:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == args.function_name:
                sig_parts = []
                for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                    arg_str = arg.arg
                    if arg.annotation:
                        ann_str = ast.unparse(arg.annotation).strip()
                        arg_str += f": {ann_str}"
                    sig_parts.append(arg_str)
                
                new_lines = []
                new_lines.append(f"def {node.name}(\n")
                indent = ' ' * (node.col_offset + 4)
                for part in sig_parts:
                    new_lines.append(f"{indent}{part},\n")
                new_lines.append(f"{' ' * node.col_offset}):")
                new_lines.extend(lines[node.body[0].lineno-1:])
                lines = new_lines
                break
    
    print('\n'.join(lines))


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(description='A swiss army knife for Python and Jupyter notebooks.', epilog="""
    To get help for a specific command, use:
    pyfun.py <command> -h
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # jup2py command
    jup2py_parser = subparsers.add_parser('jup2py', help='Convert a Jupyter notebook to a Python script.')
    jup2py_parser.add_argument('-f', '--file', required=True, help='Input Jupyter notebook file.')
    jup2py_parser.add_argument('--hide-non-code', action='store_true', help='Hide non-code cells from the output.')
    jup2py_parser.set_defaults(func=jup2py_command)
    
    # py2jup command
    py2jup_parser = subparsers.add_parser('py2jup', help='Convert a Python script to a Jupyter notebook.')
    py2jup_parser.add_argument('-f', '--file', required=True, help='Input Python script file.')
    py2jup_parser.add_argument('-o', '--output', required=True, help='Output Jupyter notebook file.')
    py2jup_parser.add_argument('--single-cell', action='store_true', help='Convert the entire script into a single notebook cell.')
    py2jup_parser.set_defaults(func=py2jup_command)

    # list command
    list_parser = subparsers.add_parser('list', help='List functions in a Python script.')
    list_parser.add_argument('-f', '--file', required=True, help='Input Python script file.')
    list_parser.add_argument('--search', help='Substring to search for in function names.')
    list_parser.set_defaults(func=list_command)

    # explain command
    explain_parser = subparsers.add_parser('explain', help='Print the source code of a function with comments.')
    explain_parser.add_argument('-f', '--file', required=True, help='Input Python script file.')
    explain_parser.add_argument('function_name', help='Name of the function to explain.')
    explain_parser.set_defaults(func=explain_command)

    # diff command
    diff_parser = subparsers.add_parser('diff', help='Show a friendly diff between a function in the script and a file.')
    diff_parser.add_argument('-f', '--file', required=True, help='Target Python script file.')
    diff_parser.add_argument('function_name', help='Name of the function to diff.')
    diff_parser.add_argument('new_file', help='File containing the new function code.')
    diff_parser.set_defaults(func=diff_command)

    # add command
    add_parser = subparsers.add_parser('add', help='Add a function from a file to the script.')
    add_parser.add_argument('-f', '--file', required=True, help='Target Python script file.')
    add_parser.add_argument('function_name', help='Name of the function to add.')
    add_parser.add_argument('new_file', help='File containing the new function code.')
    add_parser.set_defaults(func=add_command)

    # replace command
    replace_parser = subparsers.add_parser('replace', help='Replace a function in the script with one from a file.')
    replace_parser.add_argument('-f', '--file', required=True, help='Target Python script file.')
    replace_parser.add_argument('function_name', help='Name of the function to replace.')
    replace_parser.add_argument('new_file', help='File containing the new function code.')
    replace_parser.set_defaults(func=replace_command)

    # rename command
    rename_parser = subparsers.add_parser('rename', help='Rename a function and all its call sites.')
    rename_parser.add_argument('-f', '--file', required=True, help='Target Python script file.')
    rename_parser.add_argument('old_name', help='Old function name.')
    rename_parser.add_argument('new_name', help='New function name.')
    rename_parser.set_defaults(func=rename_command)

    # clean command
    clean_parser = subparsers.add_parser('clean', help='Reformat a file or a specific function using black.')
    clean_parser.add_argument('-f', '--file', required=True, help='Target Python script file.')
    clean_parser.add_argument('--function-name', help='Name of the function to reformat (note: black reformats the entire file).')
    clean_parser.set_defaults(func=clean_command)

    # dependency command
    dependency_parser = subparsers.add_parser('dependency', help='Generate a Graphviz dependency diagram for a script.')
    dependency_parser.add_argument('-f', '--file', required=True, help='Target Python script file.')
    dependency_parser.add_argument('-o', '--output', required=True, help='Output SVG file for the graph.')
    dependency_parser.set_defaults(func=dependency_command)
    
    # remove command
    remove_parser = subparsers.add_parser('remove', help='Removes a function from a Python script.')
    remove_parser.add_argument('-f', '--file', required=True, help='Target Python script file.')
    remove_parser.add_argument('function_name', help='Name of the function to remove.')
    remove_parser.set_defaults(func=remove_command)

    # view command
    view_parser = subparsers.add_parser('view', help='Views a single function with various formatting options.')
    view_parser.add_argument('-f', '--file', required=True, help='Target Python script file.')
    view_parser.add_argument('function_name', help='Name of the function to view.')
    view_parser.add_argument('-nc', '--no-comments', action='store_true', help='Print function without comments.')
    view_parser.add_argument('-c', '--clean', action='store_true', help='Align subsequent rows at "=".')
    view_parser.add_argument('--unfold', action='store_true', help='Unfold function definition, with one argument per line.')
    view_parser.add_argument('--fold', action='store_true', help='Fold function definition, with all arguments on one line (default).')
    view_parser.set_defaults(func=view_command)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
