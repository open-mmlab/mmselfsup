#!/bin/bash
kill $(ps aux | grep "train.py" | grep -v grep | awk '{print $2}')
