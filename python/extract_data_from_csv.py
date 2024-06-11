import csv
import sys
import statistics

def read_first_row_values(file_path, columns):
    """Reads the specified columns from the first row of a CSV file."""
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        first_row = next(reader)
        return {column: first_row[column] for column in columns}

def read_csv(file_path):
    """Reads a CSV file and returns a list of dictionaries representing each row."""
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
    return rows

def extract_column(rows, column_name):
    """Extracts a specified column from a list of dictionaries."""
    return [int(row[column_name]) for row in rows]

def calculate_stability(min_val, med_val):
    """Calculates the stability metric."""
    return "{:.2f}".format((med_val - min_val) * 100.0 / min_val)

def write_output_to_csv(output_path,probe_values, rocprof_values, median_row):
    fieldnames = [
            'Kernel', 'Optimization', 'ProblemSize', 'BlockSize', 'GridSize', 
            'DurationMed', 'DurationMin', 'stability', 
            'RocprofDurationMed', 'RocprofDurationMin', 'Rocprofstability', 
            'MeanOccupancyPerCU', 'MeanOccupancyPerActiveCU', 'GPUBusy', 
            'Wavefronts', 'L2CacheHit', 'SALUInsts', 'VALUInsts', 'SFetchInsts'
        ]
    with open(output_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        output_row = {
            'Kernel': probe_values['kernel'],
            'Optimization': probe_values['optim'],
            'ProblemSize': probe_values['problem_size'],
            'BlockSize': probe_values['block_size'],
            'GridSize': probe_values['grid_size'],
            'DurationMed': probe_values['time_med'],
            'DurationMin': probe_values['time_min'],
            'stability': probe_values['stability'],
            'RocprofDurationMed': rocprof_values['time_med'],
            'RocprofDurationMin': rocprof_values['time_min'],
            # 'RocprofDurationAvg': rocprof_values['time_avg'],
            # 'RocprofDurationLast': rocprof_values['time_last'],
            'Rocprofstability': rocprof_values['stability'],
            'MeanOccupancyPerCU': median_row['MeanOccupancyPerCU'],
            'MeanOccupancyPerActiveCU': median_row['MeanOccupancyPerActiveCU'],
            'GPUBusy': median_row['GPUBusy'],
            'Wavefronts': median_row['Wavefronts'],
            'L2CacheHit': median_row['L2CacheHit'],
            'SALUInsts': median_row['SALUInsts'],
            'VALUInsts': median_row['VALUInsts'],
            'SFetchInsts': median_row['SFetchInsts']
        }
        
        writer.writerow(output_row)
        

def main():
    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} <output_csv> <bench_input_csv> <rocprof_input_csv> <nwu>")
        sys.exit(1)

    output_path = sys.argv[1]
    csv1_path = sys.argv[2]
    csv2_path = sys.argv[3]
    nwu = int(sys.argv[4])

    try:
        csv1_columns = ['kernel', 'optim', 'problem_size', 'block_size', 'grid_size', 'time_med', 'time_min', 'stability']
        probe_values = read_first_row_values(csv1_path, csv1_columns)

        csv = read_csv(csv2_path)[nwu:]
        durations = extract_column(csv, 'DurationNs')
        # rocprof_last_duration = durations[-1]
        durations.sort()
        rocprof_min_duration = durations[0]
        rocprof_med_duration = durations[int(len(durations)/2)]
        # rocprof_avg_duration = int(statistics.mean(durations))
        rocprof_stability = calculate_stability(rocprof_min_duration, rocprof_med_duration)
        rocprof_values = {
            'time_min': rocprof_min_duration,
            'time_med': rocprof_med_duration,
            # 'time_avg': rocprof_avg_duration,
            # 'time_last': rocprof_last_duration,
            'stability': rocprof_stability
        }
        median_row = next(row for row in csv if int(row['DurationNs']) == rocprof_med_duration)
        if median_row is None:
            print("Error: Median row not found in the second CSV.")
            sys.exit(1)

        write_output_to_csv(output_path, probe_values, rocprof_values, median_row)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except KeyError as e:
        print(f"Error: Missing expected column - {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
