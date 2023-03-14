#!/bin/bash
tmux \
  new-session -n "Main Menu" "python main.py ; read" \; \
  new-window -n "EStop" "python navigation/estop_gui.py ; read" \; \
  new-window -n "Network Server" "python network_compute_server.py -m object_models_hand_cam/exported-models/object-hand-model/saved_model object_models_hand_cam/annotations/label_map.pbtxt 138.16.161.22; read" \; \
