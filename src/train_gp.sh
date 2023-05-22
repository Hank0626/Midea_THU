#!/bin/bash

params=('1H' '1V' '2H' '2V' '3H' '3V' '4H' '4V' '5H' '5V' '6H' '6V' '7H' '7V' '8H' '8V' '9H' '9V' '10H' '10V' '11H' '11V' '12H' '12V' '13H' '13V')

for param in "${params[@]}"; do
    python gaussian_process.py --test_cls "$param" --save_dir "$param"
done