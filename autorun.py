import sys
import os
"""
Used for running the training multiple times. Replace `x` with repetition value.

`py autorun.py x` 
"""
x = 1  # Default to 1 iteration

if (len(sys.argv) > 1):
    x = int(sys.argv[1])

for i in range(0, x):
    print(f'\n###########\nRun\n###########\n')
    os.system("py -3.8 train.py")
