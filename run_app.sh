#!/bin/bash
tmux \
  new-session -n "Main Menu" "python main.py ; read" \; \
  new-window -n "EStop" "python navigation/estop_gui.py ; read" \; \
  #new-window -n "Network Server" "python network_compute_server.py -m handle/exported-models/handle-model/saved_model handle/annotations/label_map.pbtxt 138.16.161.22 ; read" \; \
