# IGVC_Project
Intelligent Ground Vehicle Competition: This project will be related to autonomous navigation using visual and sensory informations.
Website:
https://kopil46.github.io/IGVC_Website_2025/#hero

IGVC Software Working Mechanism & Source Code Documentation
Introduction

This document explains how the major software components of the IGVC platform operate and how they interact with each other. The system consists of several subsystems: the Jetson camera processing pipeline, the manual sender module, the motor-control receiver, the auto/manual drive controller, the GPS parser, and the encoder subsystem. Together, these modules handle perception, communication, and actuation for the robot. The focus is on the working mechanism and code behavior of each part so that future teams can understand the software flow and continue development without confusion.

1. Jetson Camera Processing (Camera Part 1)

The Jetson performs the main perception tasks using USB cameras. The camera script processes live video frames, extracts lane information, and determines the robot’s driving intent.

Working Mechanism

The camera feed is opened using OpenCV.

A region of interest is selected to isolate the road area.

The frame is converted to grayscale and blurred to reduce noise.

A binary threshold is applied to identify bright lane features.

Contours or edges are extracted to locate left and right lane boundaries.

The script calculates how far the lane center is from the image center.

Based on this offset, the system chooses whether to go straight, turn left, or turn right.

A command string such as “Forward Half”, “Left Half”, “Right Half”, or “Stop” is generated.

The command is sent to the motor controller through serial communication.

A binary lane-mask window is displayed for debugging.

Notes

If lane lines disappear, the camera script shuts off motion and waits for valid lane readings.

The Jetson only sends text commands, not motor values.

This module provides the autonomous lane-following steering logic.

2. Manual Sender (NRF24 TM4C Sender)

The sender microcontroller transmits driving commands wirelessly through an NRF24L01 module. It receives text commands over UART and converts them to compact RF packets.

Working Mechanism

Initializes system clock, UART0, UART1, SPI, and RGB LEDs.

Configures the NRF24 radio in TX mode.

Parses UART text commands such as “Forward Half”.

Converts them into RF packets like:

FH (Forward Half)

BH (Backward Half)

LH (Left Half)

RH (Right Half)

ST (Stop)

Transmits packets and waits for acknowledgment.

LED colors reflect the command sent.

Stop commands are immediately transmitted and internal states are reset.

Notes

Uses fixed-length packets to improve reliability.

Parsing happens entirely on the sender, keeping the receiver logic simple.

May also send debug values to the Sabertooth UART.

3. Motor Controller Receiver (NRF24 TM4C Receiver)

The receiver microcontroller listens for RF packets and directly controls the robot’s motors using the Sabertooth driver or alternate motor drivers. It also manages the tower light indicators.

Working Mechanism

Initializes UART1 for Sabertooth communication, PWM for alternate drivers, SPI for NRF24, and LED tower GPIO.

Configures NRF24 in RX mode and waits for incoming packets.

Interprets received motion codes:

FH → Forward Half

BH → Backward Half

LH → Left Half

RH → Right Half

ST → Stop

AU → Auto mode

MA → Manual mode

Maps each code to Sabertooth serial commands or PWM direction pins for the motor driver in use.

Calculates direction bits and speed values. One motor is reversed to match wiring.

Updates tower light behavior:

Manual mode: solid indicator

Auto mode: flashing pattern

Stop: solid red

Stop overrides all motion immediately.

Notes

Packets are small for fast decoding.

Speed caps are enforced for safety.

The receiver is the last step in the chain before the motors.

The same control logic works for both Sabertooth and HC-160A SC motor drivers.

4. Auto/Manual Arrow-Key Drive Controller (PC Python Script)

A Python script on the PC enables manual tele-operation using keyboard arrow keys and lets the operator switch between Auto and Manual modes.

Working Mechanism

Opens a serial COM port with proper baud rate and delay settings.

Captures keyboard events through the keyboard library.

Arrow keys map to:

Up → Forward Half

Down → Backward Half

Left → Left Half

Right → Right Half

Releasing keys automatically sends a Stop command.

Pressing “A” switches to Auto mode.

Pressing “M” switches to Manual mode.

Commands are only sent when different from the last command to reduce spam.

Uses a byte-by-byte typewriter transmission style.

Notes

The PC script never touches the motors directly.

Very useful for debugging motion without camera input.

5. GPS Module (TM4C GPS Parser)

The GPS subsystem processes NMEA sentences from a GPS receiver and extracts geographic location data.

Working Mechanism

UART3 receives NMEA characters at 9600 baud.

Characters accumulate until a full line is received.

GGA sentences are identified and parsed.

The line is split into fields for time, latitude, longitude, fix quality, HDOP, satellites, and altitude.

Latitude and longitude are converted from ddmm.mmmm format into decimal degrees.

Parsed values are displayed or logged for navigation.

Notes

Valid fix quality is required before data is used.

Currently supports awareness and logging more than active navigation.

6. Encoder Subsystem

Encoders track wheel movement and help compute odometry values such as speed and distance.

Working Mechanism

Wheel encoders generate pulses during rotation.

Interrupts or GPIO edge detections count these pulses.

Each wheel has its own counter.

Distance is computed using wheel circumference and ticks-per-revolution.

Speed is obtained by measuring pulse frequency.

Values may support future closed-loop control or movement verification.

Notes

Encoders help confirm that commanded motion matches actual behavior.

Currently not used for real-time control loops, but fully available for expansion.

System Flow Summary

The Jetson processes camera frames, extracts lane geometry, and determines the driving direction.

It sends text commands such as “Forward Half” through serial output.

The sender TM4C converts these commands into NRF24 packets and transmits them.

The receiver TM4C decodes the packets and drives the motors through the Sabertooth or alternate drivers.

The receiver also updates tower indicators and enforces safety modes such as Stop.

Manual control through the PC arrow-key script or text commands is available at any time.

GPS and encoder data provide additional information for navigation and diagnostics.
