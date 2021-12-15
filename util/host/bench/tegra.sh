#!/usr/bin/env bash



DATE=$(date +"%F %T")
OUTFILE=$(date +"%s")
OUTFILE="logs/tegra_${OUTFILE}.log"
echo "# Start: ${DATE}" > ${OUTFILE}
sudo tegrastats --verbose --interval 2000 --logfile ${OUTFILE} &

echo "Tegrastats running in bg, logging to ${OUTFILE}"

# while true; do
	# tegrastats
# done
