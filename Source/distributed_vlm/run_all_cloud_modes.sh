#!/bin/bash

kill -9 $(ps -ef |grep cloud.py |grep -v grep | awk '{print $2}')

for mode in http https-aes-gcm https-chacha20; do
	echo "===== Running mode: $mode ====="
	bash start_cloud.sh "$mode"
done
