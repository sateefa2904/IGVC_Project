//-----------------------------------------------------------------------------
// Hardware Target
//-----------------------------------------------------------------------------
//
// Target Platform: EK-TM4C123GXL
// Target uC:       TM4C123GH6PM
// System Clock:    40 MHz
//
// Hardware configuration:
// UART0 (debug to PC via ICDI):
//   U0TX (PA1) -> PC
//   U0RX (PA0) <- PC
//
//-----------------------------------------------------------------------------

#include <stdint.h>
#include <stdbool.h>
#include "tm4c123gh6pm.h"
#include "uart0.h"

// Port A masks
#define UART_TX_MASK  (1u << 1)   // PA1
#define UART_RX_MASK  (1u << 0)   // PA0

//-----------------------------------------------------------------------------
// Subroutines
//-----------------------------------------------------------------------------

void initUart0(void)
{
    // Enable clocks
    SYSCTL_RCGCUART_R |= SYSCTL_RCGCUART_R0;
    SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R0;
    _delay_cycles(3);

    // Configure PA0/PA1 for UART function
    GPIO_PORTA_AFSEL_R |= (UART_TX_MASK | UART_RX_MASK);
    GPIO_PORTA_PCTL_R  &= ~(GPIO_PCTL_PA1_M | GPIO_PCTL_PA0_M);
    GPIO_PORTA_PCTL_R  |=  (GPIO_PCTL_PA1_U0TX | GPIO_PCTL_PA0_U0RX);
    GPIO_PORTA_DEN_R   |= (UART_TX_MASK | UART_RX_MASK);
    // (No pull-ups required; ICDI handles levels)

    // Configure UART0 for 115200, 8N1 at 40 MHz
    // r = fclk/(16*baud) = 40e6/(16*115200) = 21.701
    // IBRD = 21, FBRD = round(0.701*64) = 45
    UART0_CTL_R = 0;                          // disable for config
    UART0_CC_R  = UART_CC_CS_SYSCLK;          // use system clock
    UART0_IBRD_R = 21;
    UART0_FBRD_R = 45;
    UART0_LCRH_R = UART_LCRH_WLEN_8 | UART_LCRH_FEN;
    UART0_CTL_R  = UART_CTL_TXE | UART_CTL_RXE | UART_CTL_UARTEN;
}

void setUart0BaudRate(uint32_t baudRate, uint32_t fcyc)
{
    // divisorTimes128 = r*128 where r = fcyc/(16*baud)
    uint32_t divisorTimes128 = (fcyc * 8u) / baudRate;
    divisorTimes128 += 1u; // rounding
    UART0_CTL_R = 0;
    UART0_IBRD_R = (divisorTimes128 >> 7);           // floor(r)
    UART0_FBRD_R = ((divisorTimes128 >> 1) & 63u);   // round(frac(r)*64)
    UART0_LCRH_R = UART_LCRH_WLEN_8 | UART_LCRH_FEN;
    UART0_CTL_R  = UART_CTL_TXE | UART_CTL_RXE | UART_CTL_UARTEN;
}

void putcUart0(char c)
{
    while (UART0_FR_R & UART_FR_TXFF);
    UART0_DR_R = (uint8_t)c;
}

void putsUart0(char *str)
{
    uint8_t i = 0u;
    while (str[i] != '\0')
        putcUart0(str[i++]);
}

char getcUart0(void)
{
    while (UART0_FR_R & UART_FR_RXFE);
    return (char)(UART0_DR_R & 0xFF);
}

bool kbhitUart0(void)
{
    return !(UART0_FR_R & UART_FR_RXFE);
}
