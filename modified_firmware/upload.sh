#!/bin/bash

arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32:PartitionScheme=huge_app,PSRAM=enabled .
