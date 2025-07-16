#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import csv
import os

class TsvCsvViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSV/CSV File Viewer")
        self.root.geometry("1200x700")

        # Configure grid for responsiveness
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # PanedWindow for left (file browser) and right (data viewer) panes
        self.paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # --- Left Pane: File Browser ---
        self.file_browser_frame = ttk.Frame(self.paned_window, width=300)
        self.paned_window.add(self.file_browser_frame, weight=1)
        self.file_browser_frame.grid_rowconfigure(1, weight=1) # Row 1 for treeview
        self.file_browser_frame.grid_columnconfigure(0, weight=1)

        # "Up" button for navigation
        self.up_button = ttk.Button(self.file_browser_frame, text="Up", command=self.go_up_directory)
        self.up_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.file_tree = ttk.Treeview(self.file_browser_frame)
        self.file_tree.grid(row=1, column=0, sticky="nsew")

        # Scrollbars for file browser
        file_vsb = ttk.Scrollbar(self.file_browser_frame, orient="vertical", command=self.file_tree.yview)
        file_vsb.grid(row=1, column=1, sticky="ns")
        self.file_tree.configure(yscrollcommand=file_vsb.set)

        file_hsb = ttk.Scrollbar(self.file_browser_frame, orient="horizontal", command=self.file_tree.xview)
        file_hsb.grid(row=2, column=0, sticky="ew")
        self.file_tree.configure(xscrollcommand=file_hsb.set)

        self.file_tree.heading("#0", text="File System", anchor="w")
        self.file_tree.bind("<<TreeviewSelect>>", self.on_file_select)
        self.file_tree.bind("<<TreeviewOpen>>", self.on_folder_open)

        # --- Right Pane: Data Viewer ---
        self.data_viewer_frame = ttk.Frame(self.paned_window, width=700)
        self.paned_window.add(self.data_viewer_frame, weight=3)
        self.data_viewer_frame.grid_rowconfigure(1, weight=1) # Row for data display frames
        self.data_viewer_frame.grid_columnconfigure(0, weight=1)

        # Controls within data viewer frame
        control_sub_frame = ttk.Frame(self.data_viewer_frame, padding="5")
        control_sub_frame.grid(row=0, column=0, sticky="ew", columnspan=3)
        control_sub_frame.grid_columnconfigure(0, weight=1)

        self.file_name_label = ttk.Label(control_sub_frame, text="No file selected", font=("", 12, "bold"))
        self.file_name_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Delimiter selection
        self.delimiter_var = tk.StringVar(value='\t') # Default to tab
        ttk.Radiobutton(control_sub_frame, text="Tab", variable=self.delimiter_var, value='\t', command=self.reparse_file).grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(control_sub_frame, text="Comma", variable=self.delimiter_var, value=',', command=self.reparse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Exit Button
        self.exit_button = ttk.Button(control_sub_frame, text="Exit", command=self.root.destroy)
        self.exit_button.grid(row=0, column=3, padx=5, pady=5, sticky="e")

        # Frame to hold the frozen column Treeview and the scrollable data Treeview
        self.table_display_frame = ttk.Frame(self.data_viewer_frame)
        self.table_display_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.table_display_frame.grid_rowconfigure(0, weight=1)
        self.table_display_frame.grid_columnconfigure(1, weight=1) # Main data tree column

        # --- Frozen Column Treeview ---
        # This Treeview will display the row number and the first data column
        self.frozen_col_tree = ttk.Treeview(self.table_display_frame, show="headings", selectmode="browse")
        self.frozen_col_tree.grid(row=0, column=0, sticky="ns")

        # --- Main Data Treeview (Scrollable) ---
        # This Treeview will display the rest of the data columns
        self.data_tree = ttk.Treeview(self.table_display_frame, show="headings", selectmode="browse")
        self.data_tree.grid(row=0, column=1, sticky="nsew")

        # Apply styling for borders and header
        self.style = ttk.Style()
        self.style.configure("Treeview", borderwidth=1, relief="solid", rowheight=25)
        # Set Treeview background (for lines/borders) to gray, fieldbackground (for cells) to black, and foreground (text) to white
        self.style.configure("Treeview", background="black", fieldbackground="black", foreground="white") # Gray lines, black cells, white text
        
        # Configure selection colors for both Treeviews
        self.style.map("Treeview", 
                       background=[('selected', 'blue')], # Selected row background is black
                       foreground=[('selected', 'white')]) # Selected row text is white

        # Header style: white text on dark blue background
        self.style.configure("Treeview.Heading", font=('TkDefaultFont', 10, 'bold'), 
                             background='#000080', foreground='white', relief="raised")
        self.style.map("Treeview.Heading", background=[('active', '#0000A0')]) # Hover effect

        # Scrollbars for the Data Treeview (main scrollable part)
        data_vsb = ttk.Scrollbar(self.data_viewer_frame, orient="vertical", command=self.data_tree.yview)
        data_vsb.grid(row=1, column=2, sticky="ns") # Placed next to data_tree
        self.data_tree.configure(yscrollcommand=data_vsb.set)

        data_hsb = ttk.Scrollbar(self.data_viewer_frame, orient="horizontal", command=self.data_tree.xview)
        data_hsb.grid(row=2, column=0, sticky="ew", columnspan=2) # Spans across both treeviews visually
        self.data_tree.configure(xscrollcommand=data_hsb.set)

        # Synchronize vertical scrolling between frozen and main data Treeviews
        self.frozen_col_tree.configure(yscrollcommand=data_vsb.set)
        data_vsb.configure(command=lambda *args: (self.data_tree.yview(*args), self.frozen_col_tree.yview(*args)))

        # Label for "more lines not shown"
        self.more_lines_label = ttk.Label(self.data_viewer_frame, text="", font=("", 9, "italic"), foreground="gray")
        self.more_lines_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")


        self.current_filepath = None
        self.current_browser_path = os.getcwd() # Start in current working directory
        self.data_header = [] # Store header for sorting
        self.data_rows = []   # Store all data rows for sorting (if needed)
        self.sort_column_id = None
        self.sort_reverse = False

        # Pagination variables
        self.page_size = 100
        self.current_page_start_index = 0
        self.all_file_lines = [] # To store all lines for dynamic loading and sorting

        # Populate file browser with current directory
        self.refresh_file_browser(self.current_browser_path)

        # Bind scroll event for dynamic loading to the main data_tree
        self.data_tree.bind("<Button-4>", self.on_mouse_scroll)
        self.data_tree.bind("<Button-5>", self.on_mouse_scroll)
        self.data_tree.bind("<MouseWheel>", self.on_mouse_scroll)


    def populate_file_tree_children(self, path, parent_iid):
        children_of_parent = self.file_tree.get_children(parent_iid)
        if children_of_parent and self.file_tree.item(children_of_parent[0], "tags") == ("placeholder",):
            self.file_tree.delete(children_of_parent[0])

        try:
            items = sorted(os.listdir(path))
            
            visible_items_exist = False
            for item in items:
                if not item.startswith('.'):
                    visible_items_exist = True
                    break
            
            if not visible_items_exist and parent_iid != "":
                return

            for item in items:
                if item.startswith('.'):
                    continue

                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    folder_iid = self.file_tree.insert(parent_iid, "end", text=item, open=False, tags=("folder",), values=(full_path,))
                    if any(not sub_item.startswith('.') for sub_item in os.listdir(full_path)):
                         self.file_tree.insert(folder_iid, "end", text="loading...", tags=("placeholder",))
                elif os.path.isfile(full_path):
                    if item.lower().endswith(('.tsv', '.csv')):
                        self.file_tree.insert(parent_iid, "end", text=item, tags=("file", "data_file"), values=(full_path,))
                    else:
                        self.file_tree.insert(parent_iid, "end", text=item, tags=("file",), values=(full_path,))
        except Exception as e:
            if parent_iid == "":
                messagebox.showerror("Error", f"Failed to read current directory {path}: {e}")
            else:
                self.file_tree.insert(parent_iid, "end", text=f"Error: Cannot access folder ({e})", tags=("error_message",))
                print(f"Warning: Could not read folder {path}: {e}")


    def refresh_file_browser(self, path):
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        self.current_browser_path = path
        self.populate_file_tree_children(path, "") 

        display_path = os.path.basename(path)
        if display_path == "":
            display_path = "/"
        self.file_name_label.config(text=f"Browsing: {display_path}")
        self.clear_data_table()


    def on_folder_open(self, event):
        item_iid = self.file_tree.focus()
        item_path = self.file_tree.item(item_iid, "values")[0]
        self.populate_file_tree_children(item_path, item_iid)


    def on_file_select(self, event):
        selected_items = self.file_tree.selection()
        if not selected_items:
            return

        item_iid = selected_items[0]
        item_tags = self.file_tree.item(item_iid, "tags")
        item_path = self.file_tree.item(item_iid, "values")[0]

        if "data_file" in item_tags:
            self.current_filepath = item_path
            self.file_name_label.config(text=f"File: {os.path.basename(item_path)}")
            self.load_and_display_file()
        elif "folder" in item_tags:
            self.refresh_file_browser(item_path)
            self.clear_data_table()
        else:
            self.clear_data_table()
            self.file_name_label.config(text="Not a TSV/CSV file")

    def go_up_directory(self):
        if self.current_browser_path:
            parent_path = os.path.dirname(self.current_browser_path)
            if parent_path != self.current_browser_path:
                self.refresh_file_browser(parent_path)
            else:
                messagebox.showinfo("Navigation", "Already at the root directory.")

    def reparse_file(self):
        if self.current_filepath:
            self.load_and_display_file()

    def clear_data_table(self):
        # Clear all items (rows) from both treeviews
        for item in self.frozen_col_tree.get_children():
            self.frozen_col_tree.delete(item)
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Ensure Tkinter processes the deletion before reconfiguring columns
        self.root.update_idletasks() # Use root.update_idletasks() for broader update

        # Clear column definitions for both treeviews
        self.frozen_col_tree["displaycolumns"] = ()
        self.frozen_col_tree["columns"] = ()
        self.data_tree["displaycolumns"] = ()
        self.data_tree["columns"] = ()
        
        # Reset the #0 column heading and width for frozen_col_tree
        self.frozen_col_tree.heading("#0", text="Row #")
        self.frozen_col_tree.column("#0", width=60, anchor="center", stretch=False)

        # Reset internal data storage and pagination
        self.data_header = []
        self.data_rows = []
        self.sort_column_id = None
        self.sort_reverse = False
        self.all_file_lines = []
        self.current_page_start_index = 0
        self.more_lines_label.config(text="") # Clear the "more lines" message


    def load_and_display_file(self):
        self.clear_data_table()

        try:
            chosen_delimiter = self.delimiter_var.get()
            actual_delimiter = chosen_delimiter

            self.all_file_lines = []
            with open(self.current_filepath, 'r', newline='', encoding='utf-8') as f:
                csv_reader = csv.reader(f, delimiter=chosen_delimiter)
                self.all_file_lines = list(csv_reader)

            if not self.all_file_lines:
                messagebox.showinfo("Empty File", "The selected file is empty.")
                return

            if len(self.all_file_lines) > 0:
                sample_lines_for_sniffing = [line for line in self.all_file_lines[:5]]
                sample_text = '\n'.join([chosen_delimiter.join(line) for line in sample_lines_for_sniffing])
                
                try:
                    dialect = csv.Sniffer().sniff(sample_text, delimiters=',\t')
                    sniffed_delimiter = dialect.delimiter
                    
                    if sniffed_delimiter in [',', '\t'] and sniffed_delimiter != chosen_delimiter:
                        actual_delimiter = sniffed_delimiter
                        self.delimiter_var.set(actual_delimiter)
                        with open(self.current_filepath, 'r', newline='', encoding='utf-8') as f:
                            self.all_file_lines = list(csv.reader(f, delimiter=actual_delimiter))
                except csv.Error:
                    pass
            
            self.data_header = self.all_file_lines[0]
            self.data_rows = self.all_file_lines[1:]

            # --- Configure Frozen Column Treeview ---
            # It will show the #0 column (row number) and the first data column
            frozen_col_identifiers = []
            if len(self.data_header) > 0:
                frozen_col_identifiers.append("frozen_col_0") # Identifier for the first data column

            self.frozen_col_tree["columns"] = tuple(frozen_col_identifiers)
            self.frozen_col_tree["displaycolumns"] = tuple(frozen_col_identifiers)

            # Row number column (#0)
            self.frozen_col_tree.heading("#0", text="Row #", command=lambda: self.sort_column("#0", -1))
            self.frozen_col_tree.column("#0", width=60, anchor="center", stretch=False)

            # First data column (frozen)
            if len(self.data_header) > 0:
                first_col_header_text = f"1. {self.data_header[0]}"
                self.frozen_col_tree.heading("frozen_col_0", text=first_col_header_text, 
                                             command=lambda c="frozen_col_0", idx=0: self.sort_column(c, idx))
                
                max_width_frozen = len(first_col_header_text)
                for r_idx in range(min(len(self.data_rows), 100)):
                    if 0 < len(self.data_rows[r_idx]):
                        max_width_frozen = max(max_width_frozen, len(str(self.data_rows[r_idx][0])))
                calculated_width_frozen = min(max(max_width_frozen * 12, 150), 800)
                self.frozen_col_tree.column("frozen_col_0", width=calculated_width_frozen, anchor="w", stretch=False)
            else:
                # If no data columns, ensure frozen data column is not configured
                self.frozen_col_tree["columns"] = ()
                self.frozen_col_tree["displaycolumns"] = ()


            # --- Configure Main Data Treeview (Scrollable) ---
            # It will show columns from the second data column onwards
            scrollable_col_identifiers = [f"col_{i}" for i in range(1, len(self.data_header))]
            self.data_tree["columns"] = tuple(scrollable_col_identifiers)
            self.data_tree["displaycolumns"] = tuple(scrollable_col_identifiers)

            for i, col_id in enumerate(scrollable_col_identifiers):
                original_header_index = i + 1 # +1 because we skipped the first column (index 0)
                header_text = f"{original_header_index+1}. {self.data_header[original_header_index]}"
                self.data_tree.heading(col_id, text=header_text, 
                                       command=lambda c=col_id, idx=original_header_index: self.sort_column(c, idx))
                
                max_width = len(header_text)
                for r_idx in range(min(len(self.data_rows), 100)):
                    if original_header_index < len(self.data_rows[r_idx]):
                        max_width = max(max_width, len(str(self.data_rows[r_idx][original_header_index])))
                calculated_width = min(max(max_width * 12, 150), 800)
                self.data_tree.column(col_id, width=calculated_width, anchor="w", stretch=False)

            self.current_page_start_index = 0
            self._insert_data_into_treeviews(self.data_rows[self.current_page_start_index : self.current_page_start_index + self.page_size])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read or parse file: {e}")
            self.file_name_label.config(text=f"Error loading: {os.path.basename(self.current_filepath)}")
            self.clear_data_table()

    def _insert_data_into_treeviews(self, rows_to_display):
        # Clear existing data from both treeviews
        for item in self.frozen_col_tree.get_children():
            self.frozen_col_tree.delete(item)
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        for i, row in enumerate(rows_to_display):
            global_row_index = self.current_page_start_index + i
            
            # Insert into frozen_col_tree: row number and first data column
            frozen_values = []
            if len(row) > 0: # Check if row has at least one element for the first data column
                frozen_values.append(row[0]) # First data column
            self.frozen_col_tree.insert("", "end", iid=str(global_row_index), 
                                        text=str(global_row_index + 1), values=frozen_values)
            
            # Insert into data_tree: remaining data columns
            scrollable_values = row[1:len(self.data_header)] # From second column onwards
            self.data_tree.insert("", "end", iid=str(global_row_index), 
                                  values=scrollable_values)
        
        # Update "more lines" message
        remaining_lines = len(self.data_rows) - (self.current_page_start_index + len(rows_to_display))
        if remaining_lines > 0:
            self.more_lines_label.config(text=f"... {remaining_lines} more lines not shown ...")
        else:
            self.more_lines_label.config(text="")


    def sort_column(self, col_id, col_index):
        if col_id == "#0":
            self.data_rows = self.all_file_lines[1:] # Revert to original order
            self.sort_reverse = False # Reset sort direction for original order
        else:
            try:
                def sort_key_func(item_row):
                    value = item_row[col_index] if col_index < len(item_row) else ''
                    try:
                        return (0, float(value))
                    except ValueError:
                        return (1, str(value).lower())

                self.data_rows.sort(key=sort_key_func, reverse=self.sort_reverse)

            except Exception as e:
                messagebox.showwarning("Sorting Error", f"Could not sort by this column. Error: {e}")
                return

        if self.sort_column_id == col_id:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_reverse = False
        self.sort_column_id = col_id

        # Update heading text to show sort direction for both treeviews
        # Reset all headings first to remove old arrows
        if len(self.data_header) > 0:
            # Reset frozen column header
            self.frozen_col_tree.heading("frozen_col_0", text=f"1. {self.data_header[0]}")
        
        # Reset main data tree headers
        for i, col_identifier in enumerate(self.data_tree["columns"]):
            original_header_index = i + 1
            self.data_tree.heading(col_identifier, text=f"{original_header_index+1}. {self.data_header[original_header_index]}")
        
        # Reset row number header
        self.frozen_col_tree.heading("#0", text="Row #")

        # Apply arrow to the sorted column
        if col_id == "#0":
            if self.sort_reverse:
                self.frozen_col_tree.heading("#0", text="Row # \u2193")
            else:
                self.frozen_col_tree.heading("#0", text="Row # \u2191")
        elif col_id == "frozen_col_0":
            original_header_name = self.data_header[0]
            if self.sort_reverse:
                self.frozen_col_tree.heading(col_id, text=f"1. {original_header_name} \u2193")
            else:
                self.frozen_col_tree.heading(col_id, text=f"1. {original_header_name} \u2191")
        else:
            # col_index for the main data_tree's columns
            original_header_name = self.data_header[col_index]
            if self.sort_reverse:
                self.data_tree.heading(col_id, text=f"{col_index+1}. {original_header_name} \u2193")
            else:
                self.data_tree.heading(col_id, text=f"{col_index+1}. {original_header_name} \u2191")

        self.current_page_start_index = 0
        self._insert_data_into_treeviews(self.data_rows[self.current_page_start_index : self.current_page_start_index + self.page_size])


    def on_mouse_scroll(self, event):
        if (event.num == 5 or event.delta < 0) and self.data_tree.yview()[1] >= 0.95:
            if self.current_page_start_index + self.page_size < len(self.data_rows):
                self.current_page_start_index += self.page_size
                self._insert_data_into_treeviews(self.data_rows[self.current_page_start_index : self.current_page_start_index + self.page_size])
        elif (event.num == 4 or event.delta > 0) and self.data_tree.yview()[0] <= 0.05:
            if self.current_page_start_index > 0:
                self.current_page_start_index = max(0, self.current_page_start_index - self.page_size)
                self._insert_data_into_treeviews(self.data_rows[self.current_page_start_index : self.current_page_start_index + self.page_size])


if __name__ == "__main__":
    root = tk.Tk()
    app = TsvCsvViewerApp(root)
    root.mainloop()
