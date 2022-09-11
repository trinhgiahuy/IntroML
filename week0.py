import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')
my_numbers = [None]*len(arr)
for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)
    
# Print
print(f'Before sorting {my_numbers}')

my_number_sorted = list()
for numIter in range(len(my_numbers)-1,0,-1):	
  for index in range(numIter):
    if my_numbers[index] > my_numbers[index+1]:
      # Swap 2 adjacent nums
      tempNum = my_numbers[index];
      my_numbers[index]= my_numbers[index+1];
      my_numbers[index+1] = tempNum;

# Print
print(f'After sorting {my_numbers}')
