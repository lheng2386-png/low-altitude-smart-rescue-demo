# Radiometric Thermal Notes

This project separates simulated hotspot analysis from real radiometric thermal measurement.

## Simulated Thermal

Simulated Thermal uses ordinary RGB or grayscale image intensity to create a hotspot-style visualization. It is useful for workflow demonstration and rough visual hotspot highlighting.

It is not real temperature measurement.

## Radiometric Thermal

Radiometric Thermal requires a file that contains real thermal sensor data, such as a FLIR radiometric JPG. A normal JPG image is not enough. The system must extract a real temperature matrix before reporting Celsius values.

If no radiometric thermal data is detected, AeroRescue-AI will fail clearly and will not use grayscale values to fabricate temperatures.

## FLIR Radiometric JPG

Some FLIR cameras store raw thermal data and radiometric metadata inside a JPG file. The parsing flow follows this principle:

1. inspect metadata with ExifTool;
2. check for raw thermal data;
3. extract the raw thermal image;
4. convert raw sensor values to Celsius only when the required Planck calibration metadata is present.

## Why ExifTool Is Needed

ExifTool is used to read FLIR metadata and extract embedded raw thermal images. This project does not install ExifTool automatically. If ExifTool is missing, Radiometric Thermal mode reports the missing environment and does not fake results.

## DJI R-JPEG

DJI R-JPEG temperature parsing usually requires DJI Thermal SDK / DIRP libraries. This repository currently provides only a placeholder interface for DJI R-JPEG and does not install the DJI SDK automatically.

## Infrared Detection

Infrared Detection means object detection on infrared images, such as detecting people or vehicles in thermal imagery. It is not the same as real temperature measurement. A dataset such as HIT-UAV can be useful for future infrared object detection, but it does not by itself provide radiometric temperature parsing.

## Current MVP Boundary

- Simulated Thermal: available for ordinary images, not real measurement.
- FLIR Radiometric Thermal: attempted when ExifTool and radiometric metadata are available.
- DJI R-JPEG: placeholder only.
- No fake temperature matrix is generated.

Only successful parsing of a real radiometric thermal file can be reported as real temperature measurement.
