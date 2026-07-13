#!/bin/bash

arduino-cli compile -v --fqbn esp32:esp32:esp32:PartitionScheme=huge_app,PSRAM=enabled .
