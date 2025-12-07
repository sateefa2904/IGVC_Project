//-----------------------------------------------------------------------------
// UART3 on TM4C123GH6PM  (PC6=U3RX, PC7=U3TX)
// System clock assumed 40 MHz
//-----------------------------------------------------------------------------
#include <stdint.h>
#include <stdbool.h>
#include "tm4c123gh6pm.h"
#include "uart3.h"

#define UART3_TX_MASK (1u<<7)   // PC7
#define UART3_RX_MASK (1u<<6)   // PC6

void initUart3(void)
{
    // Enable clocks
    SYSCTL_RCGCUART_R |= SYSCTL_RCGCUART_R3;
    SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R2;      // Port C
    _delay_cycles(3);

    // Configure PC6/PC7 for U3RX/U3TX
    GPIO_PORTC_AFSEL_R |= (UART3_TX_MASK | UART3_RX_MASK);
    GPIO_PORTC_PCTL_R  &= ~(0xFF000000u);
    GPIO_PORTC_PCTL_R  |=  (GPIO_PCTL_PC6_U3RX | GPIO_PCTL_PC7_U3TX);
    GPIO_PORTC_DEN_R   |= (UART3_TX_MASK | UART3_RX_MASK);
    GPIO_PORTC_DIR_R   |= UART3_TX_MASK;          // TX out
    GPIO_PORTC_DIR_R   &= ~UART3_RX_MASK;         // RX in

    // UART3 @ 9600 baud, 8N1 using 40 MHz system clock
    // r = 40e6 / (16*9600) = 260.4167  -> IBRD=260, FBRD=27
    UART3_CTL_R  = 0;                              // disable for config
    UART3_CC_R   = UART_CC_CS_SYSCLK;
    UART3_IBRD_R = 260;
    UART3_FBRD_R = 27;
    UART3_LCRH_R = UART_LCRH_WLEN_8 | UART_LCRH_FEN;
    UART3_CTL_R  = UART_CTL_TXE | UART_CTL_RXE | UART_CTL_UARTEN;
}

void setUart3BaudRate(uint32_t baud, uint32_t fcyc)
{
    uint32_t div128 = (fcyc * 8u) / baud;  // = (fcyc / (16*baud)) * 128
    div128 += 1u;                           // round

    UART3_CTL_R  = 0;
    UART3_IBRD_R = (div128 >> 7);
    UART3_FBRD_R = ((div128 >> 1) & 63u);
    UART3_LCRH_R = UART_LCRH_WLEN_8 | UART_LCRH_FEN;
    UART3_CTL_R  = UART_CTL_TXE | UART_CTL_RXE | UART_CTL_UARTEN;
}

void putcUart3(char c)
{
    while (UART3_FR_R & UART_FR_TXFF);
    UART3_DR_R = (uint8_t)c;
}

void putsUart3(char *str)
{
    uint8_t i = 0;
    while (str[i] != '\0')
        putcUart3(str[i++]);
}

char getcUart3(void)
{
    while (UART3_FR_R & UART_FR_RXFE);
    return (char)(UART3_DR_R & 0xFF);
}

bool kbhitUart3(void)
{
    return !(UART3_FR_R & UART_FR_RXFE);
}
