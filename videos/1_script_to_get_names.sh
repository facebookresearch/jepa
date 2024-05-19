#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory_path> <append_text>"
    exit 1
fi

# Assign arguments to variables
directory_path=$1
append_text=$2

# Output CSV file
output_file="output.csv"

# Check if the provided directory exists
if [ ! -d "$directory_path" ]; then
    echo "Directory $directory_path does not exist."
    exit 1
fi

# Write file names to the CSV
> "$output_file"  # Clear the output file if it exists
for file in "$directory_path"/*; 
do
    if [ -f "$file" ]; then
        echo "$(basename "$file") $append_text" >> "$output_file"
    fi
done

echo "File names have been written to $output_file"

