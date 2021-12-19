#!/usr/bin/env bash


sudo ps
PID=$(ps | grep "tegrastats"| grep -Po '^([0-9]+) ')
if [[ -z "${PID}" ]]; then
    echo "tegrastats not found".
    exit
fi
echo "Killing PID ${PID}..."
sudo kill -9 ${PID}

echo "Done."
