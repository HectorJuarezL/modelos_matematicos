#!/bin/bash

Title_1='Trimmed_'
for f in *.nc
do
	####if [ "$f" = "gdas1.fnl0p25.2018060818.f00.grib2.nc" ] 
	####then
	 	Title="${Title_1}$f"
		echo $f
                ncea -d lat_0,234,330 -d lon_0,993,1060 "$f" "$Title"
		echo $Title
	####else
	####	echo "token"
	####fi
done

echo "DONE"
