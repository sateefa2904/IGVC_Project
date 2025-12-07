#ifndef UART3_H_
#define UART3_H_

#include <stdint.h>
#include <stdbool.h>

//-----------------------------------------------------------------------------
// Subroutines
//-----------------------------------------------------------------------------

void initUart3(void);
void setUart3BaudRate(uint32_t baudRate, uint32_t fcyc);
void putcUart3(char c);
void putsUart3(char *str);
char getcUart3(void);
bool kbhitUart3(void);

#endif /* UART3_H_ */
