#!/bin/bash
trap 'sudo shutdown -h now' EXIT
python -m PhagoPred.run_all