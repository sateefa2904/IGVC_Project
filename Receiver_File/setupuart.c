#include "setupuart.h"
#include <stdlib.h>   // atoi
#include <string.h>   // strcmp
#include <ctype.h>    // isalpha, isdigit, isprint

void getsUart0(USER_DATA *data)
{
    uint8_t count = 0;
    data->fieldCount = 0;                 // reset tokens; parseFields will set later

    while (1)
    {
        char c = getcUart0();

        // Handle backspace and delete
        if (c == 8 || c == 127)
        {
            if (count > 0)
                count--;
            continue;
        }

        // End-of-line (CR or LF)
        if (c == 13 || c == '\n')
        {
            data->buffer[count] = '\0';
            return;
        }

        // Store printable characters only
        if (isprint((unsigned char)c))
        {
            if (count < MAX_CHARS)
            {
                data->buffer[count++] = c;
            }
            else
            {
                // Buffer full; terminate
                data->buffer[MAX_CHARS] = '\0';
                return;
            }
        }
        // Ignore everything else
    }
}

void parseFields(USER_DATA *dat)
{
    uint8_t i = 0;
    char prevType = 'd';   // delimiter
    dat->fieldCount = 0;

    while (1)
    {
        char c = dat->buffer[i];

        if (c == '\0')
            break;

        char type;
        if (isalpha((unsigned char)c))
            type = 'a';
        else if (isdigit((unsigned char)c))
            type = 'n';
        else
            type = 'd';

        // Replace delimiters with NUL to split tokens
        if (type == 'd')
            dat->buffer[i] = '\0';

        // New field starts when we transition from delimiter -> (alpha|numeric)
        if (prevType == 'd' && (type == 'a' || type == 'n'))
        {
            if (dat->fieldCount < MAX_FIELDS)
            {
                dat->fieldPosition[dat->fieldCount] = i;
                dat->fieldType    [dat->fieldCount] = type;
                dat->fieldCount++;
            }
            else
            {
                // Reached max fields; stop tokenizing further
                break;
            }
        }

        prevType = type;
        i++;
    }
}

char* getFieldString(USER_DATA* data, uint8_t fieldNumber)
{
    if (fieldNumber < data->fieldCount && data->fieldType[fieldNumber] == 'a')
        return &data->buffer[data->fieldPosition[fieldNumber]];
    return NULL;
}

int32_t getFieldInteger(USER_DATA* data, uint8_t fieldNumber)
{
    if (fieldNumber < data->fieldCount && data->fieldType[fieldNumber] == 'n')
        return atoi(&data->buffer[data->fieldPosition[fieldNumber]]);
    return 0;
}

bool isCommand(USER_DATA* data, const char strCommand[], uint8_t minArguments)
{
    // First field must exist and be alpha
    if (data->fieldCount == 0 || data->fieldType[0] != 'a')
        return false;

    // Command string at field 0
    const char* cmd = &data->buffer[data->fieldPosition[0]];
    if (strcmp(cmd, strCommand) != 0)
        return false;

    // arguments = fieldCount - 1
    return (data->fieldCount > 0) && ((data->fieldCount - 1) >= minArguments);
}

int my_strcmp(const char* str1, const char* str2)
{
    while (*str1 != '\0' && *str2 != '\0')
    {
        if (*str1 != *str2)
            return (int)((unsigned char)*str1 - (unsigned char)*str2);
        str1++;
        str2++;
    }
    return (int)((unsigned char)*str1 - (unsigned char)*str2);
}
