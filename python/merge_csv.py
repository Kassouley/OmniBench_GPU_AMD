import os
import pandas as pd
import argparse

def merge_csv_to_excel(folder_path, output_file):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                sheet_name = os.path.splitext(filename)[0]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"All csv have been merged in {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge all CSV files in a folder into an Excel file with each CSV as a separate sheet.')
    parser.add_argument('folder_path', type=str, help='The path to the folder containing CSV files.')
    parser.add_argument('output_file', type=str, help='The name of the output Excel file.')
    args = parser.parse_args()

    merge_csv_to_excel(args.folder_path, args.output_file)
