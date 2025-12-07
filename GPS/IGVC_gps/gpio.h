#ifndef GPIO_H_
#define GPIO_H_

#include <stdint.h>
#include <stdbool.h>

// Enable clock for a GPIO port: pass 'A'..'F'
void enablePort(uint8_t port);

// Configure pin as push-pull digital output
void selectPinPushPullOutput(uint8_t port, uint8_t pin);

// Configure pin as digital input
void selectPinDigitalInput(uint8_t port, uint8_t pin);

// Write pin value
void setPinValue(uint8_t port, uint8_t pin, bool value);

// Toggle pin
void togglePinValue(uint8_t port, uint8_t pin);

// Read pin value
bool getPinValue(uint8_t port, uint8_t pin);

#endif /* GPIO_H_ */
