#!/bin/bash

found=0

for installer in ./*.bin; do 
    if [ -e "$installer" ]; then 
        echo "Installer found!"
        chmod u+x $installer 
        $installer -f ./response.properties 
        found=1
    fi; 
done

if [ $found -eq 0 ]; then
    echo "No installer found. CPLEX not installed."
fi