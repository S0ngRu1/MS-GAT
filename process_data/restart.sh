#!/bin/bash
while true; do
    python process_data/image_detection.py
    echo "程序退出，等待 30 秒后重启..."
    sleep 30
done
