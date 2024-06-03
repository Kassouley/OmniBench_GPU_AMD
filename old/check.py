from math import log10, floor
import sys
import traceback

def main() :
    try:
        file1 = open(f'{sys.argv[1]}')
        file2 = open(f'{sys.argv[2]}')
        max_error = 0
        max_v1 = 0
        max_v2 = 0
        j=0
        i=0
        for value_f1, value_f2 in zip(file1, file2):
            error = abs(float(value_f1) - float(value_f2))
            if error > max_error :
                j=i
                max_v1 = value_f1
                max_v2 = value_f2
                max_error = error
                if floor(log10(abs(max_error))) > -1 :
                    break

            i=i+1
        print(max_v1,max_v2,j)
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