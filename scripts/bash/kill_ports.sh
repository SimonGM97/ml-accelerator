#!/bin/bash
# chmod +x ./scripts/bash/kill_ports.sh
# ./scripts/bash/kill_ports.sh

# Get a list of all processes listening on ports
ports_in_use=$(lsof -iTCP -sTCP:LISTEN -n -P | awk 'NR>1 {print $2}' | sort -u)

# Check if there are any processes to kill
if [ -z "$ports_in_use" ]; then
    echo "No ports are in use."
else
    echo "Killing processes using the following ports:"
    echo "$ports_in_use"

    # Kill each process using a port
    for pid in $ports_in_use; do
        echo "Killing process with PID: $pid"
        kill -9 "$pid"
    done

    echo "All processes using ports have been killed."
fi