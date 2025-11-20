import duckdb
import glob
import os

# 1. Define the input pattern and output directory
json_pattern = '/home/lateef/thesis/src/results/*sem.json'
output_dir = '/home/lateef/thesis/src/result_extraction/individual_vc_slam_semantics_results/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# 2. Connect to DuckDB
con = duckdb.connect()

# 3. Find all matching files
file_list = glob.glob(json_pattern)

print(f"Found {len(file_list)} files to process.")

# 4. Loop through each file and execute the query
for input_file in file_list:
    # Get the base filename (e.g., 'file1.json')
    base_name = os.path.basename(input_file)
    # Create a unique output filename (e.g., 'file1_results.csv')
    output_filename = base_name.replace('.json', '_results.csv')
    output_path = os.path.join(output_dir, output_filename)

    # The SQL query is modified to read only the current input_file
    # and write to the unique output_path.
    sql_query = f"""
    COPY (
      SELECT
        map_item.item->>'source' AS source_column,
        map_item.item->>'match' AS match_column
      FROM
        read_json_objects('{input_file}') AS t,
        UNNEST(
          (t.json->'mappings')::JSON[]
        ) AS map_item(item)
    ) TO '{output_path}' (FORMAT CSV, DELIMITER ',');
    """

    print(f"Processing {base_name} -> {output_filename}")
    try:
        con.execute(sql_query)
        print(f"Successfully created {output_filename}")
    except Exception as e:
        print(f"Error processing {base_name}: {e}")

# 5. Close the connection
con.close()