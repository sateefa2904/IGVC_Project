# IGVC_Project
Intelligent Ground Vehicle Competition: This project will be related to autonomous navigation using visual and sensory informations.

**Website:**
https://kopil46.github.io/IGVC_Website_2025/#hero





Team Name 
MAV - Intelligent Ground Vehicle Competition 
Timeline 
Summer 2025 – Fall 2025 
Students 
● Alex Kowis – Computer Engineering 
● Kopil Sharma – Computer Engineering 
● Roza Rezwana – Computer Engineering 
● Brian Biju – Computer Engineering 
● Andy Tran  – Computer Engineering 
● Khanh Van Lam - Computer Engineering 
Abstract 
IGVC stands for Intelligent Ground Vehicle Competition. This is a yearly competition that 
involves teams of university students nationwide creating their own autonomous vehicle. 
The task is to design, build, and program a vehicle that can navigate autonomously 
through a series of obstacles in an outdoor environment. Our system consists of 3 
layers: the input, the processing, and the output layer. The input layer uses sensors to 
gather data about the surrounding environment and the vehicle. The processing layer 
takes this data and makes a decision on how the vehicle should move. The output layer 
receives this decision and outputs to the motors and emergency lights.  
Using this logic, we were successfully able to detect and follow lanes, see and avoid 
objects, and track our precise location. Our team was able to gain experience in 
advanced robotics and showcase our embedded systems integration. Moving forward, 
the foundation we built will help the future team perfect the algorithms required to qualify 
and compete in IGVC. 
Background 
Human driving mistakes, labor shortages, and unsafe environments are major reasons 
why autonomous navigation is increasingly needed in real-world applications. 
Integrating autonomy in these fields will increase safety and efficiency all around. 
Modern processors can execute millions of instructions a second, leading to rapid 
decision making based on precise sensor readings. These autonomous systems will 
utilize this capability to analyze the environment and make decisions based on what 
their functionality is. From designing and building the vehicle to programming and 
calibrating the sensors, IGVC is a great way for engineering students to gain real world 
experience in autonomous behavior. 
Project Requirements 
● Building Constraint: The vehicle must fit within specific dimensions. Length has 
to be between 3 - 7 feet. Width has to be between 2 - 4 feet. Height has to be no 
taller than 6 feet (excluding the emergency lights). 
● Propulsion: The vehicle must be powered by an onboard electrical system. Fuel 
storage or running internal combustion is prohibited.  
● Speed Constraints: During the competition, the vehicle must travel within 1mph 
to 5mph. If the ending average speed is below 1 mph, a disqualification will 
occur. 
● Emergency Stop Buttons: A mechanical and wireless stop button must be 
present and functional during the competition. Both must stop the vehicle through 
hardware, not through software. The mechanical button must be located in the 
center back of the vehicle between 2 - 4 feet off the ground. It also must be red 
and at least 1 inch in diameter. The wireless switch will be held by the judge and 
must have a range of 100 feet. 
● Safety Lights: Easily viewed lights must be present on the vehicle. Specific 
sequences should be displayed depending on the state the machine is in. On 
start up, the lights must be solid. When switching to autonomous mode, the lights 
must switch to flashing. When autonomous mode ends, the lights must go back 
to solid. 
● External Payload:  Every team must carry identical payloads during their run. 
The payload is going to be 20lbs and will be approximately the size of a cinder 
block: 16” x 8” x 8”  
● Processor Communication: The Jetson Orin NX and TM4C123GH6PM 
microcontroller must have an effective and quick two-way communication 
program set. 
● Lane Detection: Our algorithm must be able to utilize both cameras to detect the 
lane boundaries and follow them at a safe and precise distance. 
● Obstacle Avoidance: The LiDAR must be able to identify if there is an object 
that is blocking the current path of the vehicle and recognize the next best path to 
take. 
● Sensor Integration: Our system must be able to retrieve and process data from 
multiple sensors simultaneously. 
Design Constraints 
● Schedule: This project is being passed across to another team who will have 
another semester to perfect the design and programming. This motivated us to 
put them in a good spot going forward. We prioritized days for planning, building, 
or testing to ensure  the hardware based part of the project is complete for them. 
We want them to be able to focus on the programming and tests IGVC judges 
may ask them to complete. 
● Safety: Because of the use of autonomy, unexpected situations may occur. 
Having two hardware based stop buttons, a display of safety lights, and speed 
limitations add extra layers of protection to avoid any critical accidents. This 
influenced our mechanical and electrical design to follow these safety measures 
and ensure compliance. 
● Interoperability: The components we are using include: a Jetson Orin NX, two 
TM4C microcontrollers, motor drivers, LiDAR, cameras, GPS, and encoders. To 
ensure seamless and safe communication across all hardware, we carefully 
chose the protocols to use and voltage level matching. 
● Functionality: The vehicle needs to autonomously detect lanes and avoid 
objects. These constraints lead to how we selected the sensors, the algorithms 
we used in receiving/processing data, and the testing phases. 
● Aesthetics: A side challenge in the IGVC is an award for best look/design. This 
contributed to how we designed and built the sleek top look. We set days toward 
wire management, measuring and 3D printing, and assembly. We finalized on a 
simplistic silver/black design with the use of LEDs to showcase the components 
we used. 
Engineering Standards 
● IEEE 802.15.4 - Wireless Communication Protocol - This standard defines low 
power wireless communication regulations. We are using nRFs to transmit data 
over radio frequencies from one microcontroller to another. These nRFs use low 
power transmission in compliance with this standard. 
● ISO 12100 – Safety of Machinery (Risk Assessment and Risk Reduction) - This standard 
defines that in the design process, risks must be anticipated and safety measures must 
be in place to avoid them. Our vehicle includes emergency stop protocols, flashing 
safety lights, speed limitations, and manual control.   
● ISO 13850 - Safety of Machinery (Emergency Stop Function) - This standard specifies 
that a machine must have an emergency stop device that is functional, visible, and easily 
accessible. Our vehicle has two emergency stop buttons. One is wireless and handheld 
with a range of 100 feet and the other is located in the center back of the machine about 
2 feet up. 
● IEC 60204-1 - International Standard for the Safety of Electrical Equipment - This 
standard defines the requirements for wiring and grounding to protect the user 
and equipment from electrical shocks. Our vehicle complies with this as we use 
structured wiring paths and proper grounding. 
● ISO 9899 - Information Technology (C Programming) - This standard states the C 
programming syntax, library usage, and semantic requirements when writing 
code. We followed this standard when programming our microcontrollers to 
ensure predictable behavior and provide a clear, understandable code for the 
next team. 
System Overview 
Our autonomous vehicle utilizes multiple sensors to understand its surroundings and 
determine how to navigate the course. LiDAR, camera vision, GPS, and wheel 
encoders all contribute data used to calculate direction and speed. The system is 
organized into three layers: the Input Layer, the Processing Layer, and the Output 
Layer. 
The Input Layer consists of the hardware responsible for supplying real time 
environmental, positional, and safety information. Cameras, LiDAR, encoders, GPS, 
and emergency switches provide the data required for autonomous navigation and 
protective safety measures. 
The Processing Layer receives all input data and determines how the vehicle should 
respond. This layer includes an input TM4C123GH6PM microcontroller, a Jetson Orin 
NX, and an output TM4C123GH6PM microcontroller. The microcontrollers gather 
sensor data and communicate wirelessly, while the Jetson Orin NX performs the main 
perception and decision-making tasks. Together, these components calculate the 
appropriate motor commands. 
The Output Layer executes the movement and safety actions based on the processed 
commands. The output TM4C sends the correct signals to the motor drivers to control 
wheel speed and direction, while also managing the safety light indicators to reflect the 
vehicle’s current operating state. 
Results 
Our vehicle successfully demonstrated lane detection and tracking, and it was also able 
to identify and avoid obstacles using LiDAR while moving autonomously. GPS data and 
encoder feedback provided precise positional information while the vehicle is in motion. 
Overall, the project achieved a strong functional foundation and positions the next team 
well for continued refinement toward IGVC qualification. 
IGVC
Future Work 
The next team will need to integrate the lane detection and obstacle avoidance 
subsystems to create a fully unified autonomous mode. Camera vision capabilities 
should be expanded to detect stop signs, potholes, and construction workers to pass 
IGVC official scenarios/tests. Wireless switch functionality also remains to be tested. 
Additionally, future teams may consider adding hydraulic mechanisms, which our team 
planned but did not have time to complete. 
Project Files 
Project Charter 
System Requirements Specification 
Architectural Design Specification 
Detailed Design Specification 
Color-Coded Wiring Schematic (for hardware projects) 
Basic Logic Block Diagram (for hardware projects) 
Poster 
Closeout Materials 
References 
1. S. McDermott, “Motor maniacs: Autonomous ground vehicle senior design 
project,”https://cse.uta.edu/senior-design/projects/motor-maniacs/, 2025, university of 
Texas at Arlington, posted May 5, 2025. Accessed: 2025-09-19. 
2. Intelligent Ground Vehicle Competition 2026 Rulebook, 2026, https://www.igvc.org. 
3. IEEE Standards Association, “IEEE Standard for Low-Rate Wireless Networks,” IEEE 
802.15.4, 2020. 
4. International Organization for Standardization, “Safety of Machinery—General Principles 
for Design—Risk Assessment and Risk Reduction,” ISO 12100, 2010. 
5. International Organization for Standardization, “Safety of Machinery—Emergency Stop 
Function—Principles for Design,” ISO 13850, 2015. 
6. International Electrotechnical Commission, “Safety of Machinery—Electrical Equipment 
of Machines—Part 1: General Requirements,” IEC 60204-1, 2016. 
7. International Organization for Standardization, “Information Technology—Programming 
Languages—C,” ISO/IEC 9899, 2018. 
