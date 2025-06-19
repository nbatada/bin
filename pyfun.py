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

#==                                                                                                                                                 
import re

def _align_assignments(code_segment):
    """                                                                                                                                             
    Aligns assignment operators ('=') for lines that (likely) are simple assignments.                                                               
    This version uses a regex that matches a variable (or tuple of variables)                                                                       
    on the left-hand side.                                                                                                                          
    """
    # This pattern matches lines that start with indentation and then a “word”                                                                      
    # (or comma separated words) as the left-hand side. Modify if you need to cover                                                                 
    # more complex assignments.                                                                                                                     
    assignment_pattern = re.compile(r'^(\s*)([\w,]+)\s*=\s*(.+)$')

    lines = code_segment.splitlines()
    aligned_output = []
    group = []  # will accumulate tuples: (indent, lhs, rhs)                                                                                        

    def process_group():
        nonlocal group, aligned_output
-UUU:---  F1  pyfun.py       Top   L1     (Python ElDoc) -------------------------------------------------------------------------------------------
