import os
import csv
import tempfile
import shutil

# Folder containing your CSV files
input_folder = "datasets/rawdata_csv_samples/"  # Change this if needed

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)

        # Create a temporary file to write the updated content
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False) as tmpfile:
            with open(input_path, "r", encoding="utf-8", newline="") as infile:
                reader = csv.reader(infile, delimiter=';')
                writer = csv.writer(tmpfile, delimiter=',')
                for row in reader:
                    writer.writerow(row)

        # Replace the original file with the updated one
        shutil.move(tmpfile.name, input_path)
        print(f"✅ Updated: {filename} (semicolon → comma)")
