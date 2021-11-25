#!/usr/bin/env bash

# Check we're in docker
# I keep running the scripts on the host with... unwanted results
# Bit hacky

IN_DOCKER=false
CHECK_DOCKER=$(sudo cat /proc/self/cgroup)
echo "Checking docker..."
if [[ "${CHECK_DOCKER}" == *"cpuset:/docker/"* ]]; then
    IN_DOCKER=true
    return
fi

echo "Don't seem to be in Docker. Run this script from inside the container only!"
exit 