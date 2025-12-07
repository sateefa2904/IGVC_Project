#ifndef UART0_H_
#define UART0_H_

#include <stdint.h>
#include <stdbool.h>

//-----------------------------------------------------------------------------
// Subroutines
//-----------------------------------------------------------------------------

void initUart0(void);
void setUart0BaudRate(uint32_t baudRate, uint32_t fcyc);
void putcUart0(char c);
void putsUart0(char *str);
char getcUart0(void);
bool kbhitUart0(void);

#endif /* UART0_H_ */
