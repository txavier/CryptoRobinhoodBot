#!/bin/bash

# run bot every 5 minutes
while true; do
   # do stuff
   echo $(date)
   python3 main.py
   sleep $[60 * 5]
done
