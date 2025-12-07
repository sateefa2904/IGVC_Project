# IGVC_Project
Intelligent Ground Vehicle Competition: This project will be related to autonomous navigation using visual and sensory informations.
Website:
https://kopil46.github.io/IGVC_Website_2025/#hero


IGVC Software Working Mechanism & Source Code Documentation
Introduction
This document explains how the major software components of the IGVC platform operate and how they interact with each other. The system consists of several subsystems: the Jetson camera processing pipeline, the manual sender module, the motor-control receiver, the auto/manual drive controller, the GPS parser, and the encoder subsystem. Together, these modules handle perception, communication, and actuation for the robot.
The focus of this document is on the working mechanism and code behavior of each part so that future teams can understand the software flow and continue development without confusion.

1. Jetson Camera Processing (Camera Part 1)
The Jetson handles the main perception tasks using USB cameras. The camera script processes live video frames, extracts lane information, and determines the robot’s driving intent.
Working Mechanism
1.	The camera feed is opened using OpenCV.
2.	A specific region of interest is selected to isolate the road area and ignore irrelevant parts of the image.
3.	The frame is converted to grayscale and blurred to remove noise.
4.	A binary threshold is applied so that only bright lane features remain visible.
5.	Contours or edges are extracted to identify the left and right lane boundaries.
6.	The script computes how far the lane center is from the frame’s horizontal center.
7.	Based on this offset, the system decides whether the robot should go straight, turn left, or turn right.
8.	A command string (for example: “Forward Half”, “Left Half”, “Right Half”, or “Stop”) is generated.
9.	The command is sent through the serial port to the motor controller system using a typewriter-style serial transmit method to ensure clean transmission.
10.	The processed binary lane view is displayed for debugging so the operator can visually confirm that lane extraction is working.
Notes on Behavior
•	When lane lines disappear, the script shuts the camera off for safety and waits until lanes reappear.
•	All decisions are text-based, so the Jetson only sends simple command words rather than low-level motor values.
•	This module serves as the autonomous steering logic for lane following.

2. Manual Sender (NRF24 TM4C Sender)
The sender microcontroller is responsible for transmitting driving commands wirelessly using an NRF24L01 module. It receives text commands from a PC or Jetson over UART and converts them into compact RF packets.
Working Mechanism
1.	System clock, UART0, UART1, SPI, and RGB LEDs are initialized.
2.	The NRF24 radio is configured in TX mode using a fixed address and channel.
3.	When a line of text is entered on UART0 (such as “Forward Half”), the microcontroller parses the text into tokens.
4.	The command is converted to a short RF sequence such as:
•	FH (Forward Half)
•	BH (Backward Half)
•	LH (Left Half)
•	RH (Right Half)
•	ST (Stop)
5.	The NRF24 transmits the data and waits for a successful acknowledgment.
6.	LEDs change color depending on the command type to provide visual debugging feedback.
7.	If the Stop command is received, the sender immediately sends the stop code and resets internal states.
Notes on Behavior
•	Transmission uses short fixed-length packets to maintain reliability.
•	All parsing and command translation happen on the sender so the receiver only needs to interpret compact codes.
•	Sabertooth debug output may also be generated over UART1 when needed.

3. Motor Controller Receiver (NRF24 TM4C Receiver)
The receiver microcontroller listens for incoming RF packets and directly drives the robot motors using the Sabertooth motor controller. It also manages the light tower indicators for safety and mode signaling.
Working Mechanism
1.	Hardware initialization sets up UART1 for Sabertooth communication, PWM for new motor drivers communication, SPI for NRF24, and GPIO for tower LEDs.
2.	The NRF24 is configured in RX mode and waits for packets from the sender.
3.	Once a packet is received, the two-letter motion code is checked:
•	FH → Forward Half
•	BH → Backward Half
•	LH → Left Half
•	RH → Right Half
•	ST → Stop
•	AU → Auto mode
•	MA → Manual mode
4.	The receiver maps the code to actual Sabertooth serial drive commands by controlling both motor channels. It also maps the code to set the direction pins in the new motor drivers. It means that it can work with any of the motor driver that is connected to it. 
5.	Direction bits and speed values are computed based on the command type. One motor direction is reversed to match the robot’s physical wiring.
6.	The light tower updates according to the current mode:
•	Manual mode: solid indication
•	Auto mode: dynamic flashing pattern
•	Stop: red
7.	The Stop command overrides everything and immediately forces zero output to the motors.
Notes on Behavior
•	Commands are kept small so that decoding is fast and consistent.
•	Speed limits are intentionally capped for safe movement.
•	The receiver acts as the final authority for robot motion since all commands terminate here.
•	Code is kept constant to work for both motor drivers, the sabertooth and the cheap motor drivers – HC-160A SC.

4. Auto/Manual Arrow-Key Drive Controller (PC Python Script)
This script is used for manual tele-operation using a standard computer keyboard. It allows switching between Auto and Manual modes and sends direction commands based on held arrow keys.
Working Mechanism
1.	Opens a serial COM port with proper baud settings and boot timing.
2.	Hooks global keyboard events using the keyboard library.
3.	Pressing arrow keys generates continuous motion commands:
•	Up → Forward Half
•	Down → Backward Half
•	Left → Left Half
•	Right → Right Half
4.	Releasing all keys automatically sends a Stop command.
5.	Pressing A switches to Auto mode and sends “Auto” to the system.
6.	Pressing M switches to Manual mode.
7.	Commands are only transmitted when they change from the previous state to reduce serial spam.
8.	A typewriter-style serial transmitter sends data byte-by-byte with delays to prevent buffer overflows on the receiving microcontroller.
Notes on Behavior
•	This script does not talk to motors directly; it only sends text commands.
•	It is very useful for testing movement and verifying receiver functionality without any camera input.

5. GPS Module (TM4C GPS Parser)
The GPS subsystem reads NMEA data from a GPS module and extracts key geographic and positional data.
Working Mechanism
1.	UART3 receives NMEA strings at 9600 baud.
2.	Characters are stored until a full line ending in newline is detected.
3.	Only GGA sentences are processed.
4.	The line is split into fields corresponding to time, latitude, longitude, fix quality, HDOP, satellites, and altitude.
5.	Latitude and longitude are converted from ddmm.mmmm format to decimal degrees.
6.	Results are printed out for monitoring or logged for navigation.
Notes
•	The parser checks for valid fix quality before reporting position.
•	The module currently provides situational awareness and logging rather than active navigation.

6. Encoder Subsystem
The encoder subsystem measures wheel movement so distance and odometry can be calculated.
Working Mechanism
1.	Wheel encoders produce pulses as the wheels rotate.
2.	The microcontroller counts these pulses using interrupt routines or edge-triggered GPIO.
3.	Each wheel has a counter storing accumulated ticks.
4.	Distance is computed using wheel circumference and ticks-per-revolution.
5.	Speed can be derived by counting pulses per unit time.
6.	These values may be transmitted or used for higher-level control if needed.
Notes
•	Encoders help verify that commanded motion matches actual movement.
•	They are not currently used for closed-loop control but are available for expansion.

System Flow Summary
1.	The Jetson processes the camera feed, detects lane lines, and determines the intended steering direction.
2.	It sends readable text commands (“Forward Half”, “Left Half”, etc.) through serial output.
3.	The sender TM4C converts these commands into NRF24 packets and transmits them wirelessly.
4.	The receiver TM4C decodes the packets and drives the motors through the Sabertooth controller.
5.	The receiver updates tower lights and enforces safety behaviors such as Stop and mode changes.
6.	Manual control is available at any time using the arrow-key driving script or PC text input.
7.	GPS and encoders provide additional positional and motion data for navigation and analysis.

