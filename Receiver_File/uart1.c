/*
 * uart1.c
 *
 *  Created on: Oct 15, 2025
 *      Author: Kopil Sharma
 */




// UART1 Library
// Based on Jason Losh’s UART0 code
// Adapted for UART1 on PB0 (U1RX) and PB1 (U1TX)

#include <stdint.h>
#include <stdbool.h>
#include "tm4c123gh6pm.h"
#include "uart1.h"
#include "gpio.h"

#define UART1_TX PORTB,1
#define UART1_RX PORTB,0

// Initialize UART1
void initUart1(void)
{
    // Enable UART1 and GPIOB clocks
    SYSCTL_RCGCUART_R |= SYSCTL_RCGCUART_R1;
    SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R1;
    _delay_cycles(3);

    enablePort(PORTB);

    // Configure UART1 pins
    selectPinPushPullOutput(UART1_TX);
    selectPinDigitalInput(UART1_RX);
    setPinAuxFunction(UART1_TX, GPIO_PCTL_PB1_U1TX);
    setPinAuxFunction(UART1_RX, GPIO_PCTL_PB0_U1RX);

    // Disable UART1 while configuring
    UART1_CTL_R = 0;
    UART1_CC_R  = UART_CC_CS_SYSCLK;   // Use system clock
}

// Set baud rate for UART1
void setUart1BaudRate(uint32_t baudRate, uint32_t fcyc)
{
    uint32_t divisorTimes128 = (fcyc * 8) / baudRate;
    divisorTimes128 += 1;

    UART1_CTL_R = 0; // Disable UART1 for configuration
    UART1_IBRD_R = divisorTimes128 >> 7;
    UART1_FBRD_R = ((divisorTimes128) >> 1) & 63;
    UART1_LCRH_R = UART_LCRH_WLEN_8 | UART_LCRH_FEN | 8; // 8N1, FIFOs enabled
    UART1_CTL_R = UART_CTL_TXE | UART_CTL_RXE | UART_CTL_UARTEN;
}

// Write one character
void putcUart1(char c)
{
    while (UART1_FR_R & UART_FR_TXFF);  // Wait if TX FIFO full
    UART1_DR_R = c;
}

// Write a string
void putsUart1(char* str)
{
    uint8_t i = 0;
    while (str[i] != '\0')
        putcUart1(str[i++]);
}

// Read one character
char getcUart1(void)
{
    while (UART1_FR_R & UART_FR_RXFE);  // Wait if RX FIFO empty
    return UART1_DR_R & 0xFF;
}

// Check if data available
bool kbhitUart1(void)
{
    return !(UART1_FR_R & UART_FR_RXFE);
}
