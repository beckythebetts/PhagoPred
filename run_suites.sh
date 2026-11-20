#!/bin/bash
python -m PhagoPred.survival_v2.run_graph_pipeline --suite "Graph All Hazard" --workers 4   # also generates all 13 datasets
python -m PhagoPred.survival_v2.run_graph_pipeline --skip-generate --suite "Graph All Binary" --workers 4
python -m PhagoPred.survival_v2.run_graph_pipeline --skip-generate --suite "Graph Linear AR" --workers 4
python -m PhagoPred.survival_v2.run_graph_pipeline --skip-generate --suite "Graph Linear Noise" --workers 4
