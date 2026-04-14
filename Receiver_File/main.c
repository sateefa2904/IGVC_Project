// ============================ Combined Sabertooth + Safety Light ============================
// - UART1 on PB0/PB1 for Sabertooth packetized serial (unchanged)
// - Safety light LEDs on PB4 (RED), PB5 (ORANGE), PB6 (GREEN)
// - STOP button moved OFF PB1 to PE1  (active-low, with pull-up)
// - MODE button on PE2               (active-low, with pull-up)
// - Solid RED when stopped; GREEN blink when driving
// - "Stop" command or STOP button -> motors stop + solid RED


#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "tm4c123gh6pm.h"
#include "clock.h"
#include "gpio.h"
#include "wait.h"
#include "uart0.h"
#include "uart1.h"
#include "rgb_led.h"
#include "spi0.h"
#include "nrf0.h"
#include "motors.h"


//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------

// ================= Sabertooth Packetized Serial helpers =================
#define SAB_ADDR   129 //changed to default 129 for 2x32 Sabertooth, originally 130
#define ST_M1_FWD  0
#define ST_M1_REV  1
#define ST_M2_FWD  4
#define ST_M2_REV  5

static inline void st_uart1_put(uint8_t b)
{
    while (UART1_FR_R & UART_FR_TXFF);
    UART1_DR_R = b;
}

static inline void st_send(uint8_t cmd, uint8_t data)
{
    uint8_t cksum = (uint8_t)((SAB_ADDR + cmd + data) & 0x7F);
    st_uart1_put(SAB_ADDR);
    st_uart1_put(cmd);
    st_uart1_put(data);
    st_uart1_put(cksum);
}

static inline void st_m1_forward(uint8_t spd)  { st_send(ST_M1_FWD, spd); }
static inline void st_m1_reverse(uint8_t spd)  { st_send(ST_M1_REV, spd); }
static inline void st_m2_forward(uint8_t spd)  { st_send(ST_M2_FWD, spd); }
static inline void st_m2_reverse(uint8_t spd)  { st_send(ST_M2_REV, spd); }

static inline void st_stop_all(void)
{
    st_m1_forward(0);
    st_m2_forward(0);
}

// ==================================
void     NRF_WR_REG(uint8_t reg, uint8_t data);
uint8_t  NRF_READ_REG(uint8_t reg);
void     NRF_COMMAND(uint8_t cmd);
void     NRF_WR_MULTI(uint8_t reg, uint8_t *data, uint8_t len);

void     NRF24_Init(void);
void     NRF24_InitRX(void);
void     NRF24_Tx(uint8_t *Address, uint8_t channel);
void     NRF24_Rx(uint8_t *Address, uint8_t channel);
uint8_t  NRF24_Transmit(uint8_t *data, uint8_t len);
void     NRF24_Receive(uint8_t *data, uint8_t len);
uint8_t  isDataAvailable (int num);

// ================= Onboard RGB (Port F) =================
#define RED_LED_F    PORTF,1
#define BLUE_LED_F   PORTF,2
#define GREEN_LED_F  PORTF,3

// ================= Safety Light (external tower) on Port B =================
#define RED_PIN_PORT      PORTB
#define ORANGE_PIN_PORT   PORTB
#define GREEN_PIN_PORT    PORTB
#define RED_PIN           4    // PB4
#define ORANGE_PIN        5    // PB5
#define GREEN_PIN         6    // PB6

// ================= Buttons (moved STOP off PB1!) =================
// Active LOW with internal pull-ups
#define STOP_BTN_PORT   PORTE
#define STOP_BTN_PIN    1      // PE1  (STOP button)
#define MODE_BTN_PORT   PORTE
#define MODE_BTN_PIN    2      // PE2  (optional mode)

// ================= NRF =================
static uint8_t RxAddress[5] = {0xA0,0xB0,0xA0,0xB0,0xA0};
#define RF_CHANNEL 10

// handy raw byte put on UART1 (old code path)
static inline void putnum(uint8_t number)
{
    while (UART1_FR_R & UART_FR_TXFF);
    UART1_DR_R = number;
}

// ================= Console parser buffers =================
#define MAX_CHARS 80
static char strInput[MAX_CHARS+1];
static char* token;
static uint8_t count = 0;

// ================= Drive state -> ties lights to motion =================
typedef enum {
    LIGHT_STOPPED = 0,  // solid RED
    LIGHT_DRIVING       // running pattern
} light_state_t;

// ----- Light modes & auto-pattern engine -----
typedef enum { MODE_MANUAL = 0, MODE_AUTO = 1 } light_mode_t;
volatile light_mode_t light_mode = MODE_MANUAL;

// Auto pattern state
static volatile uint16_t pattern_tick = 0;   // tick timer
static volatile uint8_t  pattern_step = 0;   // step within pattern
static volatile uint8_t  pattern_id   = 0;   // which pattern to play (cycles 0,1,...)
static volatile uint16_t pattern_cycles = 0; // how many full sequences completed


static volatile light_state_t light_state = LIGHT_STOPPED;
static volatile bool driving = false;          // set true on any motion cmd, false on stop
static volatile uint32_t light_tick = 0;       // simple software ticker for patterns


// ===== Auto-mode pattern tables =====
// Bitmask per step: bit0=R, bit1=O, bit2=G
enum { LED_R = 1u<<0, LED_O = 1u<<1, LED_G = 1u<<2 };

// Pattern 0: forward chase with blanks  (G, off, O, off, R, off)
static const uint8_t PAT0[] = { LED_G, 0, LED_O, 0, LED_R, 0 };

// Pattern 1: “double with blanks”: R,0,R,0,O,0,O,0,G,0,G,0
static const uint8_t PAT1[] = { LED_R,0, LED_R,0, LED_O,0, LED_O,0, LED_G,0, LED_G,0 };

// Pattern 2: ping–pong with blanks: G,0,O,0,R,0,O,0
static const uint8_t PAT2[] = { LED_G,0, LED_O,0, LED_R,0, LED_O,0 };

// Pattern 3: strobe & mixes: GO,0, OR,0, RG,0, ALL,0, OFF
static const uint8_t PAT3[] = { LED_G|LED_O,0, LED_O|LED_R,0, LED_R|LED_G,0, LED_R|LED_O|LED_G,0, 0 };

// Pattern registry
static const uint8_t* const PATTERNS[] = { PAT0, PAT1, PAT2, PAT3 };
static const uint8_t        PATLEN  [] = { sizeof(PAT0), sizeof(PAT1), sizeof(PAT2), sizeof(PAT3) };
static const uint8_t        NPATS      = 4;


//-----------------------------------------------------------------------------
// Emergency Light Code
//-----------------------------------------------------------------------------


// ---------------- Safety Light helpers ----------------
// modified 4/8/2026 to reverse the values.
static inline void safetyLightAllOff(void)
{
    setPinValue(RED_PIN_PORT,    RED_PIN,    1);
    setPinValue(ORANGE_PIN_PORT, ORANGE_PIN, 1);
    setPinValue(GREEN_PIN_PORT,  GREEN_PIN,  1);
}

static inline void safetyLightSolidRed(void)
{
    setPinValue(RED_PIN_PORT,    RED_PIN,    0);
    setPinValue(ORANGE_PIN_PORT, ORANGE_PIN, 1);
    setPinValue(GREEN_PIN_PORT,  GREEN_PIN,  1);
}

static inline void safetyLightSolidOrange(void)
{
    setPinValue(RED_PIN_PORT,    RED_PIN,    1);
    setPinValue(ORANGE_PIN_PORT, ORANGE_PIN, 0);
    setPinValue(GREEN_PIN_PORT,  GREEN_PIN,  1);
}

static inline void safetyLightSolidGreen(void)
{
    setPinValue(RED_PIN_PORT,    RED_PIN,    1);
    setPinValue(ORANGE_PIN_PORT, ORANGE_PIN, 1);
    setPinValue(GREEN_PIN_PORT,  GREEN_PIN,  0);
}

static inline void towerSet(bool r, bool o, bool g)
{
    setPinValue(RED_PIN_PORT,    RED_PIN,    r);
    setPinValue(ORANGE_PIN_PORT, ORANGE_PIN, o);
    setPinValue(GREEN_PIN_PORT,  GREEN_PIN,  g);
}

static void safetyLightUpdate(void) // Updates the Emergency Lights
{
    // Stopped dominates everything
    if (light_state == LIGHT_STOPPED || !driving)
    {
        //setPinValue(RED_PIN_PORT,    RED_PIN,    0);
        towerSet(false, true, true);   // solid RED
        return;
    }

    // Manual mode while driving: solid Orange
    if (light_mode == MODE_MANUAL)
    {
        towerSet(true, true, true);
        return;
    }

    // ================= AUTO mode + driving =================
    // Make it fast by lowering TICKS_PER_BEAT and/or main loop wait
    const uint16_t TICKS_PER_BEAT = 40;  // smaller = faster (try 20–60)
    const uint8_t  curPat = pattern_id % NPATS;
    const uint8_t  curLen = PATLEN[curPat];

    // beat timing
    pattern_tick++;
    if (pattern_tick >= TICKS_PER_BEAT)
    {
        pattern_tick = 0;
        pattern_step = (uint8_t)((pattern_step + 1) % curLen);
        if (pattern_step == 0)           // one full sequence done
            pattern_cycles++;
    }

    // ================= Unique Patterns while driving =================
    // ================= Make sure to disable before competition =================

    /*
    // drive LEDs for this step
    uint8_t s = PATTERNS[curPat][pattern_step]; // used to set unique patterns for tower light.


    // Rotate to next pattern after a few cycles to keep it “crazy”
    if (pattern_cycles >= 4)             // 4 full loops → next pattern
    {
        pattern_cycles = 0;
        pattern_step   = 0;
        pattern_tick   = 0;
        pattern_id++;
    }
    waitMicrosecond(2000);
    */
}

// quick power-on test (one sweep), optional
static void safetyLightPowerOnTest(void)
{
    safetyLightAllOff();
    waitMicrosecond(1000);

    // RED
    setPinValue(RED_PIN_PORT, RED_PIN, 1);
    waitMicrosecond(500000);
    setPinValue(RED_PIN_PORT, RED_PIN, 0);
    waitMicrosecond(500000);

    // ORANGE
    setPinValue(ORANGE_PIN_PORT, ORANGE_PIN, 1);
    waitMicrosecond(500000);
    setPinValue(ORANGE_PIN_PORT, ORANGE_PIN, 0);
    waitMicrosecond(500000);

    // GREEN
    setPinValue(GREEN_PIN_PORT, GREEN_PIN, 1);
    waitMicrosecond(500000);
    setPinValue(GREEN_PIN_PORT, GREEN_PIN, 0);
    waitMicrosecond(500000);
}


//-----------------------------------------------------------------------------
// Initializations
//-----------------------------------------------------------------------------


// ================= Radio setup =================
static inline void rf_clear_irqs(void)
{
    NRF_WR_REG(STATUS, (1<<6)|(1<<5)|(1<<4));
}

static void rf_prepare_rx(void)
{
    NRF24_Init();
    waitMicrosecond(5000);
    NRF24_InitRX();
    waitMicrosecond(5000);
    NRF24_Rx(RxAddress, RF_CHANNEL);
    rf_clear_irqs();
    NRF_COMMAND(FLUSH_TX);
    NRF_COMMAND(FLUSH_RX);
}

// ================= Hardware init =================
static void initHw(void)
{
    initSystemClockTo40Mhz();

    // Port F RGB (onboard)
    enablePort(PORTF);
    _delay_cycles(3);
    selectPinPushPullOutput(RED_LED_F);
    selectPinPushPullOutput(GREEN_LED_F);
    selectPinPushPullOutput(BLUE_LED_F);
    initRgb();

    // UART0 console
    initUart0();
    setUart0BaudRate(19200, 40000000);

    // UART1 to Sabertooth @ 9600 (PB0/PB1)
    initUart1();
    setUart1BaudRate(9600, 40000000);

    // SPI0 (yours)
    initSpi();

    // Safety light outputs on PB4/5/6
    enablePort(PORTB);
    selectPinPushPullOutput(RED_PIN_PORT,    RED_PIN);
    selectPinPushPullOutput(ORANGE_PIN_PORT, ORANGE_PIN);
    selectPinPushPullOutput(GREEN_PIN_PORT,  GREEN_PIN);
    safetyLightPowerOnTest();
    safetyLightSolidRed();  // start safe

    // Buttons on PE1 (STOP), PE2 (MODE)
//    enablePort(PORTE);
//    selectPinDigitalInput(STOP_BTN_PORT, STOP_BTN_PIN);
//    enablePinPullup(STOP_BTN_PORT, STOP_BTN_PIN);
//    selectPinDigitalInput(MODE_BTN_PORT, MODE_BTN_PIN);
//    enablePinPullup(MODE_BTN_PORT, MODE_BTN_PIN);
}


//-----------------------------------------------------------------------------
// 2x32 Saber-tooth Data Packet Communication
//-----------------------------------------------------------------------------


static void handle_cmd_packet(const char *msg32)
{
    const uint8_t SPEED_HALF = 18;
    const uint8_t SPEED_FULL = 18;

    char a = msg32[0];
    char b = msg32[1];

    // ----- light mode commands via radio -----
    // Accepts "AU..." for Auto, "MA..." for Manual
    if ((a=='A' || a=='a') && (b=='U' || b=='u'))
    {
        light_mode     = MODE_AUTO;
        pattern_id     = 0;
        pattern_step   = 0;
        pattern_tick   = 0;
        pattern_cycles = 0;

        // If currently driving, jump into auto pattern immediately
        if (driving) { light_state = LIGHT_DRIVING; }
        return;
    }
    if ((a=='M' || a=='m') && (b=='A' || b=='a'))
    {
        light_mode = MODE_MANUAL;
        // If driving, force solid green now
        if (driving) { towerSet(false, false, true); }
        return;
    }

    //-----------------------------------------------------------------------------
    // Wheel/Motor Movement Behavior Using 2x32 Saber-tooth
    //-----------------------------------------------------------------------------
    if (a=='F' && b=='H') //Forward Half-speed
    {
        st_m1_forward(21);
        st_m2_reverse(SPEED_HALF);

        //setMotor1(MOTOR_DIR_REVERSE);
        //setMotor2(MOTOR_DIR_REVERSE);

        setRgbColor(200,200,200); //lights up LEDs on the TM4C micro-controller
        driving = true;
        light_state = LIGHT_DRIVING;

    }
    else if (a=='B' && b=='H') //Backwards Half-speed
    {
        st_m1_reverse(21);
        st_m2_forward(SPEED_HALF);

        //setMotor1(MOTOR_DIR_FORWARD);
        //setMotor2(MOTOR_DIR_FORWARD);

        setRgbColor(0,200,0);
        driving = true;
        light_state = LIGHT_DRIVING;
    }
    else if (a=='R' && b=='H') // Right Turn Half-Speed
    {
        st_m1_forward(21);
        st_m2_forward(SPEED_HALF);

        //setMotor1(MOTOR_DIR_FORWARD);
        //setMotor2(MOTOR_DIR_REVERSE);

        setRgbColor(0,0,200);
        driving = true;
        light_state = LIGHT_DRIVING;
    }
    else if (a=='L' && b=='H') // Left Turn Half-Speed
    {
        st_m1_reverse(21+2);
        st_m2_reverse(SPEED_HALF+2);

        setRgbColor(0,0,200);
        driving = true;
        light_state = LIGHT_DRIVING;
    }
    else if (a=='S' && b=='T') //Stop
    {
        st_stop_all();
        setRgbColor(0,0,0);
        driving = false;
        light_state = LIGHT_STOPPED;
        towerSet(false,true,true);   // solid RED on stop
    }
    else
    {
        st_stop_all();
        setRgbColor(50,0,50);
        waitMicrosecond(20000);
        setRgbColor(0,0,0);
        driving = false;
        light_state = LIGHT_STOPPED;
        towerSet(false,true,true);
    }
}


//-----------------------------------------------------------------------------
// Keyboard Terminal Commands to Control Vehicle
//-----------------------------------------------------------------------------


void processSerial_Packet(void)
{
    const uint8_t SPEED_FULL       = 20;
    const uint8_t SPEED_FULL_Right = 20;
    const uint8_t SPEED_HALF       = 20;

    bool end;
    char c;

    if (!kbhitUart0())
        return;

    c = getcUart0();
    end = (c == 13) || (count == MAX_CHARS);

    if (!end)
    {
        if ((c == 8 || c == 127) && count > 0)
            count--;
        else if (c >= ' ' && c < 127)
            strInput[count++] = c;
        return;
    }

    // end-of-line
    strInput[count] = '\0';
    count = 0;

    token = strtok(strInput, " ");
    if (!token) return;

    // ----- NEW: light mode text commands -----
    if (strcmp(token, "Auto") == 0)
    {
        light_mode     = MODE_AUTO;
        pattern_id     = 0;
        pattern_step   = 0;
        pattern_tick   = 0;
        pattern_cycles = 0;
        if (driving) { light_state = LIGHT_DRIVING; } // start patterns right away
        putsUart0("Light mode: AUTO\r\n");
        return;
    }
    if (strcmp(token, "Manual") == 0)
    {
        light_mode = MODE_MANUAL;
        if (driving) { towerSet(false,false,true); }   // solid green if moving
        putsUart0("Light mode: MANUAL\r\n");
        return;
    }

    // ---------- Forward ----------
    if (strcmp(token, "Forward") == 0)
    {
        token = strtok(NULL, " ");
        if (token && strcmp(token, "Full") == 0)
        {
            st_m1_forward(SPEED_FULL_Right);
            st_m2_forward(SPEED_FULL);
            setRgbColor(0, 0, 1023);
            driving = true; light_state = LIGHT_DRIVING;
        }
        else if (token && strcmp(token, "Half") == 0)
        {
            st_m1_forward(SPEED_FULL_Right);
            st_m2_forward(SPEED_HALF);
            setRgbColor(0, 0, 200);
            driving = true; light_state = LIGHT_DRIVING;
        }
        return;
    }

    // ---------- Backward ----------
    if (strcmp(token, "Backward") == 0)
    {
        token = strtok(NULL, " ");
        if (token && strcmp(token, "Full") == 0)
        {
            st_m1_reverse(SPEED_FULL_Right);
            st_m2_reverse(SPEED_FULL);
            setRgbColor(1023, 0, 0);
            driving = true; light_state = LIGHT_DRIVING;
        }
        else if (token && strcmp(token, "Half") == 0)
        {
            st_m1_reverse(SPEED_FULL_Right);
            st_m2_reverse(SPEED_HALF);
            setRgbColor(200, 0, 0);
            driving = true; light_state = LIGHT_DRIVING;
        }
        return;
    }

    // ---------- Right (pivot) ----------
    if (strcmp(token, "Right") == 0)
    {
        token = strtok(NULL, " ");
        if (token && strcmp(token, "Full") == 0)
        {
            st_m1_forward(SPEED_FULL_Right);
            st_m2_reverse(SPEED_FULL);
            setRgbColor(0, 1023, 0);
            driving = true; light_state = LIGHT_DRIVING;
        }
        else if (token && strcmp(token, "Half") == 0)
        {

            st_m1_reverse(SPEED_FULL_Right);
            st_m2_forward(SPEED_HALF);
            setRgbColor(200, 200, 200);
            driving = true; light_state = LIGHT_DRIVING;



        }
        return;
    }

    // ---------- Left (pivot) ----------
    if (strcmp(token, "Left") == 0)
    {
        token = strtok(NULL, " ");
        if (token && strcmp(token, "Full") == 0)
        {
            st_m1_reverse(SPEED_FULL_Right);
            st_m2_forward(SPEED_FULL);
            setRgbColor(1023, 1023, 1023);
            driving = true; light_state = LIGHT_DRIVING;
        }
        else if (token && strcmp(token, "Half") == 0)
        {
            st_m1_forward(SPEED_FULL_Right);
            st_m2_reverse(SPEED_HALF);
            setRgbColor(0, 200, 0);
            driving = true; light_state = LIGHT_DRIVING;
        }
        return;
    }

    // ---------- Stop ----------
    if (strcmp(token, "Stop") == 0)
    {
        st_stop_all();
        setRgbColor(0, 0, 0);
        driving = false; light_state = LIGHT_STOPPED;
        towerSet(true,false,false);   // solid red
        return;
    }

    // Unknown -> soft stop
    st_stop_all();
    setRgbColor(50, 0, 50);
    waitMicrosecond(20000);
    setRgbColor(0, 0, 0);
    driving = false; light_state = LIGHT_STOPPED;
    towerSet(true,false,false);
}


//-----------------------------------------------------------------------------
// Main
//-----------------------------------------------------------------------------


int main(void)
{
    //initpwm();  // used for cheap motor drivers.
    //initMotors();   // used for cheap motor drivers
    //setpwm(12,11,0);
    uint32_t test_variable;

    uint8_t rx[13];
    int i;
    int have0, have1;

    initHw();
    //st_stop_all(); // (optional) start with zero speed on Saber-tooth
    rf_prepare_rx();

    for (i=0;i<13;i++) rx[i]=0;

    // start safe
    driving = false;
    light_state = LIGHT_STOPPED;
    safetyLightSolidRed();

    while (true)
    {
        // 1) Local keyboard control (UART0)
        processSerial_Packet();

        // 2) STOP button (PE1) override
//        pollStopButton();
//        pollModeButton(); // currently unused

        // 3) Radio receive
        have1 = isDataAvailable(1);
        have0 = isDataAvailable(0);

        // Example periodic keepalive or other Sabertooth cmd if desired (your old st_send(16,5))
        // st_send(16, 5);

        if (have1 || have0)
        {
            for (i=0;i<13;i++) rx[i]=0;

            NRF24_Receive(rx, 13);
            rf_clear_irqs();

            // Optional debug to UART0
            putsUart0((char*)rx);

            handle_cmd_packet((char*)rx);
        }

        // 4) Safety light update (ties to driving vs. stopped)
        safetyLightUpdate();

        // Main loop cadence (~0.5-2ms depending on your needs)
//        waitMicrosecond(10000);
    }
}
