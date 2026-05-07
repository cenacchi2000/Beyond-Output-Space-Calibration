#!/usr/bin/env bash
set -e
python -m src.run_all   --gpus 0 1   --datasets ECG200 FordA Wafer ElectricDevices UWaveGestureLibrary BasicMotions SelfRegulationSCP1 AtrialFibrillation   --models mlp lstm gru tcn fcn resnet1d inceptionlite transformer   --calibrators none temperature platt isotonic vector dirichlet sebcal   --seeds 7 13 21   --results_dir results   --tables_dir tables   --figures_dir figures   --logs_dir logs
