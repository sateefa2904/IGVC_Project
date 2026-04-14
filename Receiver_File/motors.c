/*
 * motor.c
 *
 *  Created on: Dec 1, 2025
 *      Author: Kopil Sharma
 */
/////////////////////////////////////////////////////////////////////
// This file is used to control the cheap motor drivers via PWM
// If you are using the 2x32 Sabertooth, this file is not needed.
/////////////////////////////////////////////////////////////////////

#include <stdint.h>
#include <stdbool.h>
#include "tm4c123gh6pm.h"
#include "gpio.h"
//#include <rgb_led.h>
#include "motors.h"

// Direction codes
#define MOTOR_DIR_STOP     0
#define MOTOR_DIR_FORWARD  1
#define MOTOR_DIR_REVERSE  2

// Initialize motor direction pins:
// Motor 1 -> PE1, PE2
// Motor 2 -> PE3, PE4
void initMotors(void)
{
    // Enable Port E
    enablePort(PORTE);

    // Configure PE1, PE2, PE3, PE4 as push-pull outputs
    selectPinPushPullOutput(PORTE, 1);
    selectPinPushPullOutput(PORTE, 2);
    selectPinPushPullOutput(PORTE, 3);
    selectPinPushPullOutput(PORTE, 4);

    // Default: all low (motors stopped)
    setPinValue(PORTE, 1, false);
    setPinValue(PORTE, 2, false);
    setPinValue(PORTE, 3, false);
    setPinValue(PORTE, 4, false);
}

// Set Motor 1 direction using PE1, PE2
// dir = MOTOR_DIR_STOP / MOTOR_DIR_FORWARD / MOTOR_DIR_REVERSE
void setMotor1(uint8_t dir)
{
    switch (dir)
    {
        case MOTOR_DIR_FORWARD:
            // PE1 = 1, PE2 = 0
            setPinValue(PORTE, 1, true);
            setPinValue(PORTE, 2, false);
            break;

        case MOTOR_DIR_REVERSE:
            // PE1 = 0, PE2 = 1
            setPinValue(PORTE, 1, false);
            setPinValue(PORTE, 2, true);
            break;

        default: // MOTOR_DIR_STOP
            // PE1 = 0, PE2 = 0  (coast / brake depends on driver)
            setPinValue(PORTE, 1, false);
            setPinValue(PORTE, 2, false);
            break;
    }
}

// Set Motor 2 direction using PE3, PE4
// dir = MOTOR_DIR_STOP / MOTOR_DIR_FORWARD / MOTOR_DIR_REVERSE
void setMotor2(uint8_t dir)
{
    switch (dir)
    {
        case MOTOR_DIR_FORWARD:
            // PE3 = 1, PE4 = 0
            setPinValue(PORTE, 3, true);
            setPinValue(PORTE, 4, false);
            break;

        case MOTOR_DIR_REVERSE:
            // PE3 = 0, PE4 = 1
            setPinValue(PORTE, 3, false);
            setPinValue(PORTE, 4, true);
            break;

        default: // MOTOR_DIR_STOP
            // PE3 = 0, PE4 = 0
            setPinValue(PORTE, 3, false);
            setPinValue(PORTE, 4, false);
            break;
    }
}


// PB6/PB7 PWM Library (replaces RGB LED library on PF1-3)
// Adapted for: TM4C123GH6PM @ 40 MHz



#define PC4_MASK   (1U << 4)
#define PB7_MASK   (1U << 7)


// Desired PWM frequency
// System clock = 40 MHz
// PWM clock = 40 MHz / 2 = 20 MHz (with PWMDIV=2)
// f_pwm = PWMclock / LOAD
// LOAD = 20,000,000 / 15,000 â‰ˆ 1333.33 -> use 1333
#define PWM_RELOAD 1333

//-----------------------------------------------------------------------------
// Initialize PC4 and PB7 as PWM outputs at ~15 kHz
// PC4 = "red" channel (M0PWM6, Generator 3A)
// PB7 = "green" channel (M0PWM1, Generator 0B)
//-----------------------------------------------------------------------------

void initpwm(void)
{
    // Enable clocks for PWM0 and Ports B,C
    SYSCTL_RCGCPWM_R |= SYSCTL_RCGCPWM_R0;            // PWM0
    SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R1           // GPIOB
                       | SYSCTL_RCGCGPIO_R2;          // GPIOC
    _delay_cycles(3);

    // Set PWM clock to system clock / 2 (20 MHz)
    SYSCTL_RCC_R |= SYSCTL_RCC_USEPWMDIV;
    SYSCTL_RCC_R &= ~SYSCTL_RCC_PWMDIV_M;
    SYSCTL_RCC_R |= SYSCTL_RCC_PWMDIV_2;

    // ---------- PC4 -> M0PWM6 ----------
    GPIO_PORTC_DEN_R   |= PC4_MASK;
    GPIO_PORTC_AFSEL_R |= PC4_MASK;
    GPIO_PORTC_PCTL_R &= ~GPIO_PCTL_PC4_M;
    GPIO_PORTC_PCTL_R |=  GPIO_PCTL_PC4_M0PWM6;

    // ---------- PB7 -> M0PWM1 ----------
    GPIO_PORTB_DEN_R   |= PB7_MASK;
    GPIO_PORTB_AFSEL_R |= PB7_MASK;
    GPIO_PORTB_PCTL_R &= ~GPIO_PCTL_PB7_M;
    GPIO_PORTB_PCTL_R |=  GPIO_PCTL_PB7_M0PWM1;

    // Reset and configure PWM0
    SYSCTL_SRPWM_R = SYSCTL_SRPWM_R0;   // reset PWM0
    SYSCTL_SRPWM_R = 0;                 // leave reset

    // ----- Generator 0 (PB7 / M0PWM1 = channel "green") -----
    PWM0_0_CTL_R = 0;   // disable while configuring
    PWM0_0_GENB_R = PWM_0_GENB_ACTLOAD_ZERO |
                    PWM_0_GENB_ACTCMPBD_ONE;
    PWM0_0_LOAD_R = PWM_RELOAD;
    PWM0_0_CMPB_R = 0;  // 0% duty to start

    // ----- Generator 3 (PC4 / M0PWM6 = channel "red") -----
    PWM0_3_CTL_R = 0;
    PWM0_3_GENA_R = PWM_3_GENA_ACTLOAD_ZERO |
                    PWM_3_GENA_ACTCMPAD_ONE;
    PWM0_3_LOAD_R = PWM_RELOAD;
    PWM0_3_CMPA_R = 0;  // 0% duty to start

    // Enable generators
    PWM0_0_CTL_R = PWM_0_CTL_ENABLE;
    PWM0_3_CTL_R = PWM_3_CTL_ENABLE;

    // Enable PWM outputs 1 and 6
    PWM0_ENABLE_R |= PWM_ENABLE_PWM1EN   // PB7
                   | PWM_ENABLE_PWM6EN;  // PC4
}

//-----------------------------------------------------------------------------
// Set PWM duty cycle on PC4 and PB7 in PERCENT (0–100)
// red   -> PC4 duty %  (M0PWM6)
// green -> PB7 duty %  (M0PWM1)
// blue  -> ignored
//-----------------------------------------------------------------------------

void setpwm(uint16_t red, uint16_t green, uint16_t blue)
{
    (void)blue;

    if (red > 100)   red = 100;
    if (green > 100) green = 100;

    uint32_t cmpRed   = (red   * PWM_RELOAD) / 100;
    uint32_t cmpGreen = (green * PWM_RELOAD) / 100;

    PWM0_3_CMPA_R = cmpRed;    // PC4 (M0PWM6)
    PWM0_0_CMPB_R = cmpGreen;  // PB7 (M0PWM1)
}

