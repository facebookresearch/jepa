#!/bin/bash

# Input and output files
input_file="videos_csv_file_index.csv"
output_file="videos_csv_file_index2.csv"

# Create or clear the output file
> "$output_file"

# Process each line in the input file
while IFS= read -r line; do
  # Extract the filename
  filename=$(echo "$line" | awk '{print $1}')
  # Get the absolute path of the file
  abs_path=$(readlink -f "$filename")
  # Replace the filename with its absolute path
  new_line=$(echo "$line" | sed "s|$filename|$abs_path|")
  # Write the modified line to the output file
  echo "$new_line" >> "$output_file"
done < "$input_file"

echo "Processing complete. Output written to $output_file."