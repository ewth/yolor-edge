#!/usr/bin/env bash


bash ./kill.sh
DATE=$(date +"%F %T")
OUTFILE=$(date +"%s")
OUTFILE="logs/tegra_${OUTFILE}.log"
echo "# Start: ${DATE}" > ${OUTFILE}

sudo echo "sudo OK"

sudo tegrastats --interval 2000 --logfile ${OUTFILE} &

echo "Tegrastats running in bg, logging to ${OUTFILE}"

# while true; do
	# tegrastats
# done
