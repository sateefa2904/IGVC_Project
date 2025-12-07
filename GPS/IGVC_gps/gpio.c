//-----------------------------------------------------------------------------
// Hardware Target
//-----------------------------------------------------------------------------
//
// Target Platform: EK-TM4C123GXL
// Target uC:       TM4C123GH6PM
//
//-----------------------------------------------------------------------------

#include <stdint.h>
#include <stdbool.h>
#include "tm4c123gh6pm.h"
#include "gpio.h"

// Common GPIO register offsets (same layout for ports A-F)
#define GPIO_O_DATA    0x3FCu   // masked data address
#define GPIO_O_DIR     0x400u
#define GPIO_O_AFSEL   0x420u
#define GPIO_O_DR2R    0x500u
#define GPIO_O_DR4R    0x504u
#define GPIO_O_DR8R    0x508u
#define GPIO_O_ODR     0x50Cu
#define GPIO_O_PUR     0x510u
#define GPIO_O_PDR     0x514u
#define GPIO_O_DEN     0x51Cu
#define GPIO_O_LOCK    0x520u
#define GPIO_O_CR      0x524u
#define GPIO_O_AMSEL   0x528u
#define GPIO_O_PCTL    0x52Cu

// Return APB base address for a port letter
static inline uint32_t gpioBase(uint8_t port)
{
    switch (port)
    {
        case 'A': return 0x40004000u;
        case 'B': return 0x40005000u;
        case 'C': return 0x40006000u;
        case 'D': return 0x40007000u;
        case 'E': return 0x40024000u;
        case 'F': return 0x40025000u;
        default:  return 0u;  // invalid
    }
}

// Handy accessor
static inline volatile uint32_t *REG(uint32_t base, uint32_t off)
{
    return (volatile uint32_t *)(base + off);
}

//-----------------------------------------------------------------------------
// Subroutines
//-----------------------------------------------------------------------------

void enablePort(uint8_t port)
{
    // Turn on clock for the requested port
    switch (port)
    {
        case 'A': SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R0; break;
        case 'B': SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R1; break;
        case 'C': SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R2; break;
        case 'D': SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R3; break;
        case 'E': SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R4; break;
        case 'F': SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R5; break;
        default: return;
    }

    // Dummy read + wait until peripheral is ready
    volatile uint32_t delay = SYSCTL_RCGCGPIO_R; (void)delay;
    uint32_t bit = (uint32_t)(port - 'A');
    while ((SYSCTL_PRGPIO_R & (1u << bit)) == 0) { /* wait */ }

    // For safety, disable analog + alt function by default
    uint32_t base = gpioBase(port);
    if (base)
    {
        *REG(base, GPIO_O_AMSEL) &= 0x00000000u;      // analog off (per-pin set later as needed)
        *REG(base, GPIO_O_AFSEL) &= 0x00000000u;      // GPIO by default
        *REG(base, GPIO_O_DEN)   |= 0x00000000u;      // leave DEN as-is; set per pin
    }
}

void selectPinPushPullOutput(uint8_t port, uint8_t pin)
{
    uint32_t base = gpioBase(port);
    if (!base) return;

    // Make sure digital enabled, GPIO function, push-pull 2mA
    *REG(base, GPIO_O_AFSEL) &= ~(1u << pin);
    *REG(base, GPIO_O_AMSEL) &= ~(1u << pin);
    *REG(base, GPIO_O_DEN)   |=  (1u << pin);
    *REG(base, GPIO_O_DIR)   |=  (1u << pin);
    *REG(base, GPIO_O_DR2R)  |=  (1u << pin);
    *REG(base, GPIO_O_ODR)   &= ~(1u << pin);
}

void selectPinDigitalInput(uint8_t port, uint8_t pin)
{
    uint32_t base = gpioBase(port);
    if (!base) return;

    *REG(base, GPIO_O_AFSEL) &= ~(1u << pin);
    *REG(base, GPIO_O_AMSEL) &= ~(1u << pin);
    *REG(base, GPIO_O_DEN)   |=  (1u << pin);
    *REG(base, GPIO_O_DIR)   &= ~(1u << pin);
    // No pulls by default; add PUR/PDR here if you need them
}

void setPinValue(uint8_t port, uint8_t pin, bool value)
{
    uint32_t base = gpioBase(port);
    if (!base) return;

    volatile uint32_t *DATA = REG(base, GPIO_O_DATA);
    if (value) *DATA |=  (1u << pin);
    else       *DATA &= ~(1u << pin);
}

void togglePinValue(uint8_t port, uint8_t pin)
{
    uint32_t base = gpioBase(port);
    if (!base) return;

    *REG(base, GPIO_O_DATA) ^= (1u << pin);
}

bool getPinValue(uint8_t port, uint8_t pin)
{
    uint32_t base = gpioBase(port);
    if (!base) return false;

    return ((*REG(base, GPIO_O_DATA)) & (1u << pin)) != 0u;
}
