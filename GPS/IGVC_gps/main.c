#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "tm4c123gh6pm.h"
#include "clock.h"
#include "uart0.h" // 115200 to PC
#include "uart3.h" // 9600 to GPS

#define LINE_MAX 200
static char nmeaLine[LINE_MAX];
static uint16_t li = 0;

/* Convert NMEA ddmm.mmmm (or dddmm.mmmm) -> decimal degrees (C89-safe) */
static float nmea_dm_to_deg(const char *s, uint8_t deg_digits)
{
    int deg = 0;
    uint8_t i;
    float minutes;
    const char *p;
    float place;

    for (i = 0; i < deg_digits; i++)
    {
        char c = s[i];
        if (c < '0' || c > '9')
            return 0.0f;
        deg = deg * 10 + (c - '0');
    }
    if (s[deg_digits] < '0' || s[deg_digits] > '9')
        return 0.0f;
    if (s[deg_digits + 1] < '0' || s[deg_digits + 1] > '9')
        return 0.0f;

    minutes = (float)((s[deg_digits] - '0') * 10 + (s[deg_digits + 1] - '0'));
    p = s + deg_digits + 2;
    if (*p == '.')
    {
        place = 0.1f;
        p++;
        while (*p >= '0' && *p <= '9')
        {
            minutes += (float)(*p - '0') * place;
            place *= 0.1f;
            p++;
        }
    }
    return (float)deg + minutes / 60.0f;
}

/* Parse $G?GGA. Returns true iff fix>=1 and fills outputs. (line is MUTATED) */
static bool parse_gga_mutable(char *line,
                              uint8_t *hh, uint8_t *mm, uint8_t *ss,
                              float *lat_deg, char *ns,
                              float *lon_deg, char *ew,
                              int *sats, float *alt, float *hdop_out)
{
    const char *f[16];
    int fi;
    char *p;
    const char *q;
    int v;

    if (!(line[0] == '$' && line[3] == 'G' && line[4] == 'G' && line[5] == 'A'))
        return false;

    /* split commas in-place */
    for (fi = 0; fi < 16; fi++)
        f[fi] = 0;
    p = line;
    while (*p && *p != ',')
        p++; /* skip "$xxGGA" */
    while (*p && fi < 16)
    {
        p++;
        f[fi++] = p;
        while (*p && *p != ',' && *p != '*')
            p++;
        if (*p == ',')
            *p = '\0';
        else if (*p == '*')
        {
            *p = '\0';
            break;
        }
    }
    if (fi < 9)
        return false;

    /* time hhmmss */
    *hh = *mm = *ss = 0;
    if (f[0] && strlen(f[0]) >= 6)
    {
        *hh = (uint8_t)((f[0][0] - '0') * 10 + (f[0][1] - '0'));
        *mm = (uint8_t)((f[0][2] - '0') * 10 + (f[0][3] - '0'));
        *ss = (uint8_t)((f[0][4] - '0') * 10 + (f[0][5] - '0'));
    }

    /* position */
    *ns = (f[2] && *f[2]) ? f[2][0] : 'N';
    *ew = (f[4] && *f[4]) ? f[4][0] : 'E';
    *lat_deg = nmea_dm_to_deg(f[1], 2);
    *lon_deg = nmea_dm_to_deg(f[3], 3);
    if (*ns == 'S')
        *lat_deg = -*lat_deg;
    if (*ew == 'W')
        *lon_deg = -*lon_deg;

    /* fix quality (0=no fix, 1=GPS, 2=DGPS, 4=RTK�) */
    v = (f[5] && *f[5]) ? (f[5][0] - '0') : 0;

    /* satellites */
    *sats = 0;
    if (f[6])
    {
        v = 0;
        q = f[6];
        while (*q >= '0' && *q <= '9')
        {
            v = v * 10 + (*q - '0');
            q++;
        }
        *sats = v;
    }

    /* HDOP (field 7) */
    *hdop_out = 0.0f;
    if (f[7])
        *hdop_out = (float)atof(f[7]);

    /* altitude meters (field 8) */
    *alt = 0.0f;
    if (f[8])
        *alt = (float)atof(f[8]);

    /* return true for any valid fix (>=1) */
    return ((f[5] && *f[5] && (f[5][0] - '0') >= 1) ? true : false);
}

static float dm_to_deg(float dm, int deg_digits)
{
    int dd;
    float mm;
    dd = (int)(dm / 100.0f);
    mm = dm - dd * 100.0f;
    return dd + (mm / 60.0f);
}

//int main(void)
//{
//    char c;
//
//    /* parsed fields */
//    unsigned int th, tm, ts;
//    float lat_dm, lon_dm, hdop, alt;
//    int fix, sats;
//    char ns, ew;
//
//    char out[160];
//
//    initSystemClockTo40Mhz(); /* 40 MHz */
//    initUart0();              /* 115200 to PC */
//    initUart3();              /* 9600 to GPS */
//
//    putsUart0("Parsed view (GGA):\r\n");
//
//    while (1)
//    {
//        if (kbhitUart3())
//        {
//            char c = getcUart3();
//
//            if (li < (LINE_MAX - 1))
//                nmeaLine[li++] = c;
//            else
//            {
//                nmeaLine[LINE_MAX-1] = '\0';
//                li = 0;
//            }
//
//            if (c == '\r' || c == '\n')
//            {
//                while (li && (nmeaLine[li-1] == '\r' || nmeaLine[li-1] == '\n'))
//                    li--;
//                nmeaLine[li] = '\0';
//                li = 0;
//
//                // show every line (for sanity)
//                putsUart0("RAW: ");
//                putsUart0(nmeaLine);
//                putsUart0("\r\n");
//
//                // parse only $GPGGA or $GNGGA
//                if (nmeaLine[0] == '$' &&
//                    ((nmeaLine[1]=='G'&&nmeaLine[2]=='P') || (nmeaLine[1]=='G'&&nmeaLine[2]=='N')) &&
//                     nmeaLine[3]=='G' && nmeaLine[4]=='G' && nmeaLine[5]=='A')
//                {
//                    unsigned th=0, tm=0, ts=0;
//                    float lat_dm=0, lon_dm=0, hdop=0, alt=0;
//                    int   fix=0, sats=0;
//                    char  ns='N', ew='E';
//
//                    if (sscanf(nmeaLine,
//                               "$%*2cGGA,%2u%2u%2u,%f,%c,%f,%c,%d,%d,%f,%f",
//                               &th,&tm,&ts,&lat_dm,&ns,&lon_dm,&ew,&fix,&sats,&hdop,&alt) >= 10)
//                    {
//                        int dd; float mm, lat_deg, lon_deg;
//                        dd = (int)(lat_dm/100.0f);  mm = lat_dm - dd*100.0f;  lat_deg = dd + mm/60.0f;
//                        dd = (int)(lon_dm/100.0f);  mm = lon_dm - dd*100.0f;  lon_deg = dd + mm/60.0f;
//                        if (ns=='S') lat_deg = -lat_deg;
//                        if (ew=='W') lon_deg = -lon_deg;
//
//                        if (fix >= 1) //
//                        {
//                            char out[160];
//                            snprintf(out, sizeof(out),
//                              "UTC %02u:%02u:%02u  Latitude: %.6f %c  Longitude: %.6f %c  Altitude: %.1f m  Satellites: %d  HDOP: %.2f\r\n",
//                              th, tm, ts,
//                              (ns=='S')?-lat_deg:lat_deg, ns,
//                              (ew=='W')?-lon_deg:lon_deg, ew,
//                              alt, sats, hdop);
//                            putsUart0(out);
//                        }
//                    }
//                }
//            }
//        }
//    }
//}


    int main(void)
    {
        initSystemClockTo40Mhz();   // 40 MHz
        initUart0();                // 115200 to PC
        initUart3();                // 9600 to GPS

        while (1)
        {
            if (kbhitUart3())
            {
                char c = getcUart3();
                putcUart0(c);       // just forward every GPS byte to the PC
            }
        }
    }

