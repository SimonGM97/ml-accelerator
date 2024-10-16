#!/bin/bash
# chmod +x ./scripts/bash/02_tests_running.sh
# ./scripts/bash/02_tests_running.sh

# Run Data Processing unit & integrity tests
python3 -m unittest test/test_data_processing/test_etl.py