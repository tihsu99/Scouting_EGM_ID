import argparse
import pyarrow.parquet as pq


def print_parquet_structure(file_path):
    # Open the Parquet file
    parquet_file = pq.ParquetFile(file_path)

    # Print metadata
    print("Parquet File Metadata:")
    print(parquet_file.metadata)
    print("\nSchema:")
    print(parquet_file.schema)

    # Print column names and types
    print("\nColumns:")
    for i in range(len(parquet_file.schema.names)):
        column = parquet_file.schema.column(i)
        print(f"- {column.name}: {column.physical_type}")

    # Print number of row groups
    print("\nRow Groups:")
    print(f"Total Row Groups: {parquet_file.num_row_groups}")

    # Print row count (approximate)
    total_rows = sum(parquet_file.metadata.row_group(i).num_rows for i in range(parquet_file.num_row_groups))
    print(f"Total Rows: {total_rows}")

    # Print the first row (from the first row group)
    print("\nFirst Row:")
    if total_rows > 0:
        table = parquet_file.read_row_group(0, columns=parquet_file.schema.names)
        df = table.to_pandas()
        # Loop over columns of the first row
        first_row = df.iloc[0]
        for col_name, value in first_row.items():
             print(f"{col_name}: {value}")
    else:
        print("No rows available in the file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the structure of a Parquet file.")
    parser.add_argument("file_path", type=str, help="Path to the Parquet file.")
    args = parser.parse_args()

    print_parquet_structure(args.file_path)
