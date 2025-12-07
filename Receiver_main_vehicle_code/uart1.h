/*
 * uart1.h
 *
 *  Created on: Oct 15, 2025
 *      Author: Kopil Sharma
 */

// UART1 Library Header
// For TM4C123GH6PM
// Uses PB0 (U1RX) and PB1 (U1TX)

#ifndef UART1_H_
#define UART1_H_

#include <stdint.h>
#include <stdbool.h>

void initUart1(void);
void setUart1BaudRate(uint32_t baudRate, uint32_t fcyc);
void putcUart1(char c);
void putsUart1(char* str);
char getcUart1(void);
bool kbhitUart1(void);

#endif
