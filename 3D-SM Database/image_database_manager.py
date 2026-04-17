#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D-SM Image Database Management System
A GUI application for managing and querying 3D-SM images with Excel label database
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import pandas as pd
import math

# Add project root directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class ImageDatabaseManager:
    """
    A GUI application for managing 3D-SM images with Excel label database.
    
    Features:
    - Load image database from Excel file
    - Display images with their label information
    - Filter images by any label column
    - Query and search functionality
    - Image preview with detailed information
    """
    
    def __init__(self, root):
        """
        Initialize the ImageDatabaseManager application.
        
        Args:
            root: The main Tkinter root window
        """
        print("Initializing Image Database Management System...")
        self.root = root
        self.root.title("3D-SM Image Database Management System")
        self.root.geometry("1400x900")
        
        # Data storage
        self.df = None
        self.image_folder = None
        self.image_records = []  # List of all (image_name, row_index) tuples
        self.filtered_data = []  # List of (image_name, row_index) tuples
        self.all_columns = []  # Column names for filtering
        self.current_page = 0
        self.total_pages = 0
        self.images_per_page = 24
        self.current_image_path = None
        
        # Default paths - using relative paths
        self.default_excel_path = os.path.join(os.path.dirname(__file__), "3D-SM label.xlsx")
        self.default_image_folder = os.path.join(os.path.dirname(__file__), "3D-SM")
        
        # Create UI components
        print("Creating UI components...")
        self.create_widgets()
        print("UI components created successfully")
        
        # Set default paths and auto-load if files exist
        self.set_default_paths()
        
        # Update status
        self.update_status("Ready - Please load database")
        
    def create_widgets(self):
        """Create GUI interface components"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Database Configuration")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File selection frame
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Excel file selection
        ttk.Label(file_frame, text="Excel Label File:").pack(side=tk.LEFT, padx=(0, 5))
        self.excel_path_var = tk.StringVar()
        excel_entry = ttk.Entry(file_frame, textvariable=self.excel_path_var, width=50)
        excel_entry.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="Browse", command=self.browse_excel).pack(side=tk.LEFT, padx=(0, 5))
        
        # Image folder selection
        ttk.Label(file_frame, text="Image Folder:").pack(side=tk.LEFT, padx=(10, 5))
        self.image_folder_var = tk.StringVar()
        folder_entry = ttk.Entry(file_frame, textvariable=self.image_folder_var, width=50)
        folder_entry.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT, padx=(0, 5))
        
        # Load button
        ttk.Button(file_frame, text="Load Database", command=self.load_database).pack(side=tk.LEFT, padx=(10, 0))
        
        # Filter frame
        filter_frame = ttk.Frame(control_frame)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Filter column selection
        ttk.Label(filter_frame, text="Select Filter Column:").pack(side=tk.LEFT, padx=(0, 5))
        self.filter_column_var = tk.StringVar()
        self.filter_column_combo = ttk.Combobox(filter_frame, textvariable=self.filter_column_var, 
                                               values=[], width=30, state="readonly")
        self.filter_column_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.filter_column_combo.bind('<<ComboboxSelected>>', self.on_column_selected)
        
        # Filter value selection
        ttk.Label(filter_frame, text="Filter Value:").pack(side=tk.LEFT, padx=(10, 5))
        self.filter_value_var = tk.StringVar()
        self.filter_value_combo = ttk.Combobox(filter_frame, textvariable=self.filter_value_var, 
                                              values=[], width=20, state="readonly")
        self.filter_value_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Filter buttons
        ttk.Button(filter_frame, text="Apply Filter", command=self.apply_filter).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(filter_frame, text="Clear Filter", command=self.clear_filter).pack(side=tk.LEFT, padx=(0, 5))
        
        # Search frame
        search_frame = ttk.Frame(control_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=40)
        search_entry.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(search_frame, text="Search", command=self.search_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(search_frame, text="Clear Search", command=self.clear_search).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(search_frame, text="Show All", command=self.show_all_images).pack(side=tk.LEFT, padx=(0, 5))
        
        # Create middle display area (left-right split)
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame: Query results list
        left_frame = ttk.LabelFrame(display_frame, text="Query Results")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Create Treeview to display query results
        columns = ("Image ID", "Image Name", "Filter Column", "Filter Value")
        self.result_tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=25)
        
        # Set column headings
        for col in columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=120)
        
        # Add scrollbar
        tree_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Layout Treeview and scrollbar
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind click event
        self.result_tree.bind("<<TreeviewSelect>>", self.on_image_selected)
        
        # Right frame: Image preview area
        right_frame = ttk.LabelFrame(display_frame, text="Image Preview")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Image display label
        self.image_label = ttk.Label(right_frame, text="Please select an image to view preview", anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image information text box
        info_frame = ttk.LabelFrame(right_frame, text="Image Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=15, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def set_default_paths(self):
        """Set default paths and auto-load if files exist."""
        # Set default Excel path
        if os.path.exists(self.default_excel_path):
            self.excel_path_var.set(self.default_excel_path)
            print(f"Default Excel path set: {self.default_excel_path}")
        else:
            print(f"Default Excel file not found: {self.default_excel_path}")
        
        # Set default image folder
        if os.path.exists(self.default_image_folder):
            self.image_folder_var.set(self.default_image_folder)
            print(f"Default image folder set: {self.default_image_folder}")
        else:
            print(f"Default image folder not found: {self.default_image_folder}")
        
        # Auto-load database if both paths are set and files exist
        self.auto_load_database()
        
    def auto_load_database(self):
        """Automatically load database if both Excel file and image folder exist."""
        excel_path = self.excel_path_var.get()
        image_folder = self.image_folder_var.get()
        
        if excel_path and image_folder:
            if os.path.exists(excel_path) and os.path.exists(image_folder):
                print("Auto-loading database...")
                self.load_database()
            else:
                print("Files not ready for auto-load")
        else:
            print("Paths not set yet")
            
    def browse_excel(self):
        """Open file dialog to select Excel file."""
        file_path = filedialog.askopenfilename(
            title="Select Excel Label File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if file_path:
            self.excel_path_var.set(file_path)
            # Try auto-load after Excel file selection
            self.auto_load_database()
            
    def browse_folder(self):
        """Open file dialog to select image folder."""
        folder_path = filedialog.askdirectory(
            title="Select Image Folder"
        )
        if folder_path:
            self.image_folder_var.set(folder_path)
            # Try auto-load after folder selection
            self.auto_load_database()
            
    def load_database(self):
        """
        Load Excel database and initialize the system.
        
        This function:
        1. Reads Excel file
        2. Maps each record to an image file
        3. Populates filter column dropdown with column names
        4. Initializes the database for filtering
        """
        excel_path = self.excel_path_var.get()
        image_folder = self.image_folder_var.get()
        
        if not excel_path or not os.path.exists(excel_path):
            messagebox.showerror("Error", "Please select a valid Excel file!")
            return
            
        if not image_folder or not os.path.exists(image_folder):
            messagebox.showerror("Error", "Please select a valid image folder!")
            return
        
        try:
            print("Loading Excel database...")
            # Load Excel data
            self.df = pd.read_excel(excel_path, sheet_name=0)
            self.image_folder = image_folder
            
            # Get column names (excluding first column which is image name)
            self.all_columns = list(self.df.columns[1:])
            
            # Populate filter column dropdown
            self.filter_column_combo['values'] = self.all_columns
            if self.all_columns:
                self.filter_column_combo.current(0)
            
            # Get all available image filenames from folder
            available_files = set(f for f in os.listdir(image_folder) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))
            
            # Create mapping from Excel record to image file
            self.image_records = []
            missing_images = []
            
            for idx, row in self.df.iterrows():
                # Get image name from first column
                img_name = str(row.iloc[0])
                
                # Remove .png extension if present
                if img_name.endswith('.png'):
                    img_name = img_name[:-4]
                
                # Check if image file exists
                img_file = f"{img_name}.png"
                if img_file in available_files:
                    self.image_records.append((img_file, idx))
                else:
                    # Try without extension
                    if img_name in available_files:
                        self.image_records.append((img_name, idx))
                    else:
                        missing_images.append(img_name)
            
            # Initialize filtered data with all records
            self.filtered_data = self.image_records.copy()
            
            # Update status
            if missing_images:
                status_msg = f"Database loaded: {len(self.image_records)} images found, {len(missing_images)} images not found"
            else:
                status_msg = f"Database loaded successfully: {len(self.image_records)} images"
            
            self.update_status(status_msg)
            print(status_msg)
            
            # Populate filter value dropdown for the first column
            if self.all_columns:
                self.populate_filter_values(self.all_columns[0])
            
            # Clear previous results
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            
            # Initialize pagination
            self.current_page = 0
            self.total_pages = math.ceil(len(self.filtered_data) / self.images_per_page) if self.filtered_data else 0
            
            # Show first page
            self.show_images()
            
            print("Database loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load database: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def populate_filter_values(self, column_name):
        """
        Populate filter value dropdown with unique values from a specific column.
        
        Args:
            column_name: The name of the column to populate values from
        """
        if not column_name or column_name not in self.df.columns:
            return
            
        # Get unique values from the column
        unique_values = self.df[column_name].dropna().unique()
        
        # Sort values if possible
        try:
            unique_values = sorted(unique_values)
        except:
            pass
        
        # Convert to string list
        unique_values_str = [str(val) for val in unique_values]
        
        # Update dropdown
        self.filter_value_combo['values'] = ["All"] + unique_values_str
        self.filter_value_combo.set("All")
        self.filter_value_combo.config(state="readonly")
        
    def on_column_selected(self, event=None):
        """
        Callback when a filter column is selected.
        
        This function populates the filter value dropdown with unique values
        from the selected column.
        """
        selected_column = self.filter_column_var.get()
        if selected_column:
            self.populate_filter_values(selected_column)
            
    def apply_filter(self, event=None):
        """
        Apply filter based on selected column and value.
        
        This function filters the database records based on the selected
        column and value, then updates the image display.
        """
        if self.df is None or not self.image_records:
            return
            
        selected_column = self.filter_column_var.get()
        selected_value = self.filter_value_var.get()
        
        if not selected_column:
            messagebox.showwarning("Warning", "Please select a filter column!")
            return
        
        # If "All" is selected, show all images
        if selected_value == "All":
            self.filtered_data = self.image_records.copy()
            status_msg = f"Filter cleared: Showing all {len(self.filtered_data)} images"
            self.update_status(status_msg)
        else:
            # Filter by selected column and value
            filtered_records = []
            
            for img_file, row_idx in self.image_records:
                row_data = self.df.iloc[row_idx]
                column_value = row_data[selected_column]
                
                # Handle NaN values
                if pd.isna(column_value):
                    if selected_value == "nan":
                        filtered_records.append((img_file, row_idx))
                elif str(column_value) == selected_value:
                    filtered_records.append((img_file, row_idx))
            
            self.filtered_data = filtered_records
            status_msg = f"Filter applied: {len(self.filtered_data)} images match the criteria"
            self.update_status(status_msg)
        
        # Clear previous results
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # Reset to first page
        self.current_page = 0
        self.total_pages = math.ceil(len(self.filtered_data) / self.images_per_page) if self.filtered_data else 0
        
        # Show filtered images
        self.show_images()
        
    def search_images(self):
        """
        Search images based on search text.
        
        This function searches across all columns in the database
        and displays images that match the search criteria.
        """
        search_text = self.search_var.get().strip().lower()
        
        if not search_text:
            messagebox.showwarning("Warning", "Please enter search text!")
            return
            
        if not self.df:
            return
            
        # Search in all columns
        search_results = []
        
        for img_file, row_idx in self.image_records:
            row_data = self.df.iloc[row_idx]
            
            # Check if any column contains the search text
            for col in self.df.columns:
                value = str(row_data[col])
                if search_text in value.lower():
                    search_results.append((img_file, row_idx))
                    break
        
        self.filtered_data = search_results
        status_msg = f"Search results: {len(self.filtered_data)} images found"
        self.update_status(status_msg)
        
        # Clear previous results
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # Reset to first page
        self.current_page = 0
        self.total_pages = math.ceil(len(self.filtered_data) / self.images_per_page) if self.filtered_data else 0
        
        # Show search results
        self.show_images()
        
    def clear_search(self):
        """Clear search and show all filtered images."""
        self.search_var.set("")
        
        # If filter is active, show filtered images; otherwise show all
        if self.filter_value_var.get() != "All":
            self.apply_filter()
        else:
            self.filtered_data = self.image_records.copy()
            status_msg = f"Search cleared: Showing all {len(self.filtered_data)} images"
            self.update_status(status_msg)
            
        # Clear previous results
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # Reset to first page
        self.current_page = 0
        self.total_pages = math.ceil(len(self.filtered_data) / self.images_per_page) if self.filtered_data else 0
        
        self.show_images()
        
    def clear_filter(self):
        """Clear all filters and show all images."""
        self.filter_value_var.set("All")
        self.filtered_data = self.image_records.copy()
        status_msg = f"Filter cleared: Showing all {len(self.filtered_data)} images"
        self.update_status(status_msg)
        
        # Clear previous results
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # Reset to first page
        self.current_page = 0
        self.total_pages = math.ceil(len(self.filtered_data) / self.images_per_page) if self.filtered_data else 0
        
        self.show_images()
        
    def show_all_images(self):
        """Display all images."""
        self.filtered_data = self.image_records.copy()
        status_msg = f"Showing all {len(self.filtered_data)} images"
        self.update_status(status_msg)
        
        # Clear previous results
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # Reset to first page
        self.current_page = 0
        self.total_pages = math.ceil(len(self.filtered_data) / self.images_per_page) if self.filtered_data else 0
        
        self.show_images()
        
    def show_images(self):
        """
        Display images on the current page.
        
        This function:
        1. Calculates which images to show based on current page
        2. Loads each image and its label data
        3. Displays images in a grid layout with labels
        """
        # Clear previous images
        for widget in self.result_tree.winfo_children():
            widget.destroy()
            
        if not self.filtered_data:
            self.result_tree.insert("", tk.END, values=("No images", "", "", ""))
            self.update_status("No images to display")
            return
            
        # Calculate pagination
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(self.filtered_data))
        
        current_records = self.filtered_data[start_idx:end_idx]
        
        # Insert into Treeview
        for img_file, row_idx in current_records:
            row_data = self.df.iloc[row_idx]
            
            # Get filter column and value for display
            filter_col = self.filter_column_var.get() if self.filter_column_var.get() else "N/A"
            filter_val = self.filter_value_var.get() if self.filter_value_var.get() else "N/A"
            
            self.result_tree.insert("", tk.END, values=(
                row_data.iloc[0],  # Image ID from first column
                img_file,
                filter_col,
                filter_val
            ))
        
        # Update status
        self.update_status(f"Page {self.current_page + 1}/{self.total_pages} - Showing {len(current_records)} images")
        
    def on_image_selected(self, event):
        """
        When an image is selected in the results list, display image preview and detailed information.
        """
        selection = self.result_tree.selection()
        if selection:
            # Get selected item
            item = self.result_tree.item(selection[0])
            filename = item['values'][1]  # Image name
            
            try:
                # Get row index
                img_file, row_idx = self.filtered_data[self.result_tree.index(selection[0])]
                
                # Get Excel data for this image
                row_data = self.df.iloc[row_idx]
                
                # Update image information text box
                self.info_text.config(state=tk.NORMAL)
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, f"Image File: {filename}\n\n")
                self.info_text.insert(tk.END, "Label Information:\n")
                
                # Display all label information
                for col in self.df.columns:
                    value = row_data[col]
                    if not pd.isna(value):
                        # Format value
                        if isinstance(value, float):
                            formatted_val = f"{value:.2f}"
                        else:
                            formatted_val = str(value)
                        self.info_text.insert(tk.END, f"  {col}: {formatted_val}\n")
                
                self.info_text.config(state=tk.DISABLED)
                
                # Display image preview
                image_path = os.path.join(self.image_folder, filename)
                if os.path.exists(image_path):
                    self.display_image(image_path)
                    self.current_image_path = image_path
                else:
                    self.image_label.config(text=f"Image file not found:\n{image_path}")
                    self.current_image_path = None
                    
            except Exception as e:
                error_msg = f"Failed to load image information: {str(e)}"
                print(error_msg)
                messagebox.showerror("Error", error_msg)
                
    def display_image(self, image_path):
        """Display image preview"""
        try:
            # Open and resize image
            image = Image.open(image_path)
            image.thumbnail((500, 400))  # Resize image to fit display area
            photo = ImageTk.PhotoImage(image)
            
            # Update image label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            error_msg = f"Failed to load image: {str(e)}"
            print(error_msg)
            self.image_label.config(text=error_msg, image="")
            
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()


def main():
    """Main function - Start GUI application"""
    print("Starting Image Database Management System...")
    try:
        # Create Tkinter root window
        root = tk.Tk()
        print("Tkinter root window created")
        
        # Create GUI application instance
        app = ImageDatabaseManager(root)
        print("GUI application instance created")
        
        # Start event loop
        print("Starting event loop...")
        root.mainloop()
        print("Event loop ended")
    except Exception as e:
        print(f"GUI application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
