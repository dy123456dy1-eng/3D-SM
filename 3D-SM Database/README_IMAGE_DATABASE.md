# 3D-SM Image Database Management System

## Overview
A GUI application for managing and querying 3D-SM images with Excel label database.

## Features
- Load image database from Excel file
- Display images with their label information
- Filter images by any label column
- Query and search functionality
- Image preview with detailed information
- Pagination support

## Quick Start

### 1. Launch the Application
Simply run the application:
```bash
python image_database_manager.py
```

The application will automatically:
- Detect the default Excel file: `3D-SM label.xlsx`
- Detect the default image folder: `3D-SM`
- Auto-load the database if files exist

### 2. Manual Database Loading (if needed)
If auto-loading fails, you can manually load:
1. Click "Browse" to select the Excel label file (3D-SM label.xlsx)
2. Click "Browse" to select the image folder (3D-SM)
3. Click "Load Database" button

### 3. Filter Images
1. Select a filter column from the dropdown (e.g., "MLSS(g/L)")
2. Select a filter value from the dropdown (e.g., "10.5")
3. Click "Apply Filter" button
4. Or click "Clear Filter" to show all images

### 4. Search Images
1. Enter search text in the search box
2. Click "Search" button
3. The system will search across all columns
4. Or click "Clear Search" to clear the search
5. Or click "Show All" to display all images

### 5. View Image Details
1. Click on an image in the results list (left panel)
2. The image preview will appear in the right panel
3. Detailed label information will be displayed below the image

### 6. Navigation
- Use "Previous" and "Next" buttons to navigate between pages
- Each page displays 24 images (6 columns × 4 rows)

## File Structure
```
3D-SM Database/
├── image_database_manager.py    # Main application
├── 3D-SM label.xlsx             # Excel label file
└── 3D-SM/                       # Image folder
    └── *.png                    # Image files
```

## Excel File Format
- First column: Image filenames (without extension)
- Subsequent columns: Label attributes
- First row: Column headers (label names)

## Label Columns
The system supports filtering by the following label columns:
- MLSS(g/L)
- SV5(mL)
- SV30(mL)
- SVI5(g/L)
- SVI30(g/L)
- Temperature(℃)
- pH
- Dissolved Oxygen(mg/L)
- Electrical conductivity(μS/cm)
- Granule size (mL/g)
- Filamentous bacteria enrichment

## Requirements
- Python 3.9+
- tkinter (included with Python)
- pandas
- Pillow (PIL)
- openpyxl (for Excel support)

## Usage Notes
- All code comments are in English
- The application uses a left-right split layout
- Left panel shows query results as a tree view
- Right panel shows image preview and detailed information
- Default paths use relative paths for portability
- Database auto-loads on startup if files are present

## Troubleshooting
- If auto-loading fails, manually browse and select the files
- Ensure the Excel file and image folder are in the correct location
- Check that image filenames in Excel match the actual file names
