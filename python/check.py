from math import log10, floor
import sys
import traceback

def main() :
    try:
        file1 = open(f'{sys.argv[1]}')
        file2 = open(f'{sys.argv[2]}')
        errors = []
        for line_f1, line_f2 in zip(file1, file2):
            values_f1 = line_f1.split()
            values_f2 = line_f2.split()
            errors = [abs(float(a) - float(b)) for a, b in zip(values_f1, values_f2)]
        max_error=max(errors)
        if max_error == 0 :
            print("Kernel output is \u001b[42mcorrect\033[0m.")
        else :
            exponent = floor(log10(abs(max_error)))
            if exponent < -2 :
                print(f"Kernel output is \u001b[42mcorrect\033[0m with a max error of 10^{exponent}")
            else :
                print(f"Kernel output is \u001b[41mincorrect\033[0m (error max of 10^{exponent})")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(-1)

if __name__ == '__main__':
    main()