#ifndef SETUPUART_H
#define SETUPUART_H

#include <stdint.h>
#include <stdbool.h>
#include "uart0.h"   // <-- use these declarations; don't redeclare them here

#define MAX_CHARS  80
#define MAX_FIELDS 5

typedef struct _USER_DATA
{
    char    buffer[MAX_CHARS+1];
    uint8_t fieldCount;
    uint8_t fieldPosition[MAX_FIELDS];
    char    fieldType[MAX_FIELDS];   // 'a' = alpha, 'n' = numeric
} USER_DATA;

// High-level line input & parser
void getsUart0(USER_DATA *data);
void parseFields(USER_DATA *data);
char*   getFieldString (USER_DATA* data, uint8_t fieldNumber);
int32_t getFieldInteger(USER_DATA* data, uint8_t fieldNumber);
bool    isCommand(USER_DATA* data, const char strCommand[], uint8_t minArguments);

// Utility compare
int my_strcmp(const char* str1, const char* str2);

#endif // SETUPUART_H
