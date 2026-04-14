/*
 * motor.h
 *
 *  Created on: Dec 1, 2025
 *      Author: Kopil Sharma
 */

#ifndef MOTORS_H_
#define MOTORS_H_

#include <stdint.h>

#define MOTOR_DIR_STOP     0
#define MOTOR_DIR_FORWARD  1
#define MOTOR_DIR_REVERSE  2

void initMotors(void);
void setMotor1(uint8_t dir);
void setMotor2(uint8_t dir);
// Initialize PB6 and PB7 as PWM outputs (~15 kHz)
void initpwm(void);

// Set PWM duty cycle in percent (0–100)
// red   -> PB6 duty
// green -> PB7 duty
// blue  -> ignored (kept for compatibility)
void setpwm(uint16_t red, uint16_t green, uint16_t blue);

#endif
