#include <stdint.h>
#include <stdbool.h>
#include "tm4c123gh6pm.h"
#include "wait.h"
#include "gpio.h"
#include "spi0.h"
#include "nrf0.h"
#include "wait.h"


#define NRF24_CE    PORTA,6
#define CS    PORTA,3
uint32_t config;

void enableDevice(void)
{
    setPinValue(NRF24_CE, 1);
}
void disableDevice(void)
{
    setPinValue(NRF24_CE, 0);
}

void enableSpiCs(void)
{
    setPinValue(CS, 0);
//    _delay_cycles(4);                  // allow line to settle
}

void disableSpiCs(void)
{
    setPinValue(CS, 1);
}
/*
void writeEtherReg(uint8_t reg, uint8_t data)
{
    enableEtherCs();
    writeSpi0Data(0x40 | (reg & 0x1F));
    readSpi0Data();
    writeSpi0Data(data);
    readSpi0Data();
    disableEtherCs();
}

uint8_t readEtherReg(uint8_t reg)
{
    uint8_t data;
    enableEtherCs();
    writeSpi0Data(0x00 | (reg & 0x1F));
    readSpi0Data();
    writeSpi0Data(0);
    data = readSpi0Data();
    disableEtherCs();
    return data;
}
*/
void NRF_WR_REG (uint8_t Register, uint8_t Data)
{
//    Register |= (1<<5);

    disableSpiCs();

    // Pull the CS Pin LOW to select the device
    enableSpiCs();

    writeSpi0Data(0x20|Register);
    readSpi0Data();
    writeSpi0Data(Data);
    readSpi0Data();

    // Pull the CS HIGH to release the device
    disableSpiCs();
}

uint8_t NRF_READ_REG(uint8_t reg)
{
    uint8_t data = 0;

    enableSpiCs();

    writeSpi0Data(reg);         // Send register address
    readSpi0Data();        // Dummy write to clock in response
    writeSpi0Data(0);
    data = readSpi0Data();     // Read response

    disableSpiCs();

    return data;
}


// send the command to the NRF
void NRF_COMMAND (uint8_t cmd)
{
    // Pull the CS Pin LOW to select the device
    enableSpiCs();

    writeSpi0Data(cmd);
    readSpi0Data();

    // Pull the CS HIGH to release the device
    disableSpiCs();
}
void NRF_WR_MULTI(uint8_t reg, uint8_t *data, uint8_t len)
{
    reg = reg | 1<<5; // Set MSB for write

    enableSpiCs();

    writeSpi0Data(reg);
    readSpi0Data();
    int i;
    for (i = 0; i < len; i++)
    {
        writeSpi0Data(data[i]);
        readSpi0Data();
    }
    disableSpiCs();
}

void NRF24_res(uint8_t REGISTER)
{
    if (REGISTER == STATUS)
    {
        NRF_WR_REG(STATUS, 0x00);
    }

    else if (REGISTER == FIFO_STATUS)
    {
        NRF_WR_REG(FIFO_STATUS, 0x11);
    }

    else {
    NRF_WR_REG(CONFIG, 0x08);
    NRF_WR_REG(EN_AA, 0x3F);
    NRF_WR_REG(EN_RXADDR, 0x03);
    NRF_WR_REG(SETUP_AW, 0x03);
    NRF_WR_REG(SETUP_RETR, 0x03);
    NRF_WR_REG(RF_CH, 0x02);
    NRF_WR_REG(RF_SETUP, 0x0E);
    NRF_WR_REG(STATUS, 0x00);
    NRF_WR_REG(OBSERVE_TX, 0x00);
    NRF_WR_REG(CD, 0x00);
    uint8_t rx_addr_p0_def[5] = {0xE7, 0xE7, 0xE7, 0xE7, 0xE7};
    NRF_WR_MULTI(RX_ADDR_P0, rx_addr_p0_def, 5);
    uint8_t rx_addr_p1_def[5] = {0xC2, 0xC2, 0xC2, 0xC2, 0xC2};
    NRF_WR_MULTI(RX_ADDR_P1, rx_addr_p1_def, 5);
    NRF_WR_REG(RX_ADDR_P2, 0xC3);
    NRF_WR_REG(RX_ADDR_P3, 0xC4);
    NRF_WR_REG(RX_ADDR_P4, 0xC5);
    NRF_WR_REG(RX_ADDR_P5, 0xC6);
    uint8_t tx_addr_def[5] = {0xE7, 0xE7, 0xE7, 0xE7, 0xE7};
    NRF_WR_MULTI(TX_ADDR, tx_addr_def, 5);
    NRF_WR_REG(RX_PW_P0, 0);
    NRF_WR_REG(RX_PW_P1, 0);
    NRF_WR_REG(RX_PW_P2, 0);
    NRF_WR_REG(RX_PW_P3, 0);
    NRF_WR_REG(RX_PW_P4, 0);
    NRF_WR_REG(RX_PW_P5, 0);
    NRF_WR_REG(FIFO_STATUS, 0x11);
    NRF_WR_REG(DYNPD, 0);
    NRF_WR_REG(FEATURE, 0);
    }
}

void initSpi(void)
{
    initSpi0(USE_SSI0_RX);
        setSpi0Mode(0, 0);
        setSpi0BaudRate(5e6, 40e6);
        selectPinPushPullOutput(CS);
         selectPinPushPullOutput(NRF24_CE);
}


void NRF24_Init (void)
{

    // disable the chip before configuring the device
//    setPinValue(CS, 1);
    disableDevice();
//    waitMicrosecond(10000);
//    NRF24_res(0);

    // reset everything
  //  nrf24_reset (0);

    NRF_WR_REG(CONFIG, 0x02);
//    uint8_t conf = NRF_READ_REG(CONFIG);
  // will be configured later
//    config = readSpi0Data();
//    config = NRF_READ_REG(CONFIG);
    NRF_WR_REG(EN_AA, 0);  // No Auto ACK

    NRF_WR_REG (EN_RXADDR, 0);  // Not Enabling any data pipe right now
//    config = readSpi0Data();
    NRF_WR_REG (SETUP_AW, 0x03);  // 5 Bytes for the TX/RX address
//    config = readSpi0Data();
//    config = NRF_READ_REG(SETUP_AW);
    NRF_WR_REG (SETUP_RETR, 0);   // No retransmission
//    config = readSpi0Data();

    NRF_WR_REG (RF_CH, 0);  // will be setup during Tx or RX
//    config = readSpi0Data();

    NRF_WR_REG (RF_SETUP, 0x06);   // Power= 0db, data rate = 2Mbps
//    config = readSpi0Data();
//    config = NRF_READ_REG(RF_SETUP);
    // Enable the chip after configuring the device
    enableDevice();

}

void NRF24_InitRX (void)
  {


      // disable the chip before configuring the device
  //    setPinValue(CS, 1);
      disableDevice();
  //    waitMicrosecond(10000);
  //    NRF24_res(0);

      // reset everything
    //  nrf24_reset (0);

      NRF_WR_REG(CONFIG, 0x03);
  //    uint8_t conf = NRF_READ_REG(CONFIG);
    // will be configured later
  //    config = readSpi0Data();
  //    config = NRF_READ_REG(CONFIG);
      NRF_WR_REG(EN_AA, 0);  // No Auto ACK

      NRF_WR_REG (EN_RXADDR, 0);  // Not Enabling any data pipe right now
  //    config = readSpi0Data();
      NRF_WR_REG (SETUP_AW, 0x03);  // 5 Bytes for the TX/RX address
  //    config = readSpi0Data();
  //    config = NRF_READ_REG(SETUP_AW);
      NRF_WR_REG (SETUP_RETR, 0);   // No retransmission
  //    config = readSpi0Data();

      NRF_WR_REG (RF_CH, 0);  // will be setup during Tx or RX
  //    config = readSpi0Data();

      NRF_WR_REG (RF_SETUP, 0x06);   // Power= 0db, data rate = 2Mbps
  //    config = readSpi0Data();
  //    config = NRF_READ_REG(RF_SETUP);
      // Enable the chip after configuring the device
      enableDevice();

  }


void NRF24_Tx (uint8_t *Address, uint8_t channel)
{
    disableDevice();

    NRF_WR_REG (RF_CH, channel);  // select the channel
    NRF_WR_MULTI(TX_ADDR, Address, 5);  // Write the TX address

    // Power up in TX mode
    config = NRF_READ_REG(CONFIG);
    config = (config) | (1 << 1);  // Clear PRIM_RX, Set PWR_UP

    NRF_WR_REG(CONFIG, config);
//    config = NRF_READ_REG(CONFIG);
//    config = NRF_READ_REG(SETUP_AW);
//    config = NRF_READ_REG(RF_SETUP);

    enableDevice();

}

uint8_t NRF24_Transmit(uint8_t *data, uint8_t len)
{
    enableSpiCs();
    disableDevice();
//    NRF_WR_REG(RX_PW_P0, 32);  // Tell NRF how many bytes you're sending

    writeSpi0Data(W_TX_PAYLOAD|0x20);
    readSpi0Data();
//    waitMicrosecond(100);
    int i;
    for (i = 0; i < len; i++)
    {
        writeSpi0Data(data[i]);
        readSpi0Data();
    }
    disableSpiCs();
disableDevice();
waitMicrosecond(50);
    uint8_t fifostatus = NRF_READ_REG(FIFO_STATUS);
    enableDevice();             // CE HIGH
    waitMicrosecond(15);        // Hold CE HIGH for at least 10us
             // CE LOW


    disableDevice();
        waitMicrosecond(10000);     // Let TX finish
    fifostatus = NRF_READ_REG(FIFO_STATUS);
    if ((fifostatus & (1 << 4)) && !(fifostatus & (1 << 3)))
    {
        NRF_COMMAND(FLUSH_TX);
//        NRF24_res(FIFO_STATUS);
        return 1;
    }

    return 0;
}


void NRF24_Rx (uint8_t *Address, uint8_t channel)
{
    disableDevice();

    NRF_WR_REG (RF_CH, channel);  // select the channel
    uint8_t en_rxaddr = NRF_READ_REG(EN_RXADDR);
    en_rxaddr = en_rxaddr | (1<<1);

//    nrf24_WriteRegMulti(RX_ADDR_P1, Address, 5);  // Write the Pipe1 address
         // Write the Pipe2 LSB address

//        nrf24_WriteReg (RX_PW_P2, 32);   // 32 bit payload size for pipe 2

    NRF_WR_REG (EN_RXADDR, en_rxaddr);
    NRF_WR_MULTI(RX_ADDR_P1, Address, 5);  // Write the TX address
//    NRF_WR_REG(RX_ADDR_P2, 0xAB);
    NRF_WR_REG (RX_PW_P1, 32);

    // Power up in TX mode
    uint8_t config = NRF_READ_REG(CONFIG);
    config = config | (1<<1) | (1<<0);
    NRF_WR_REG(CONFIG, config);

    enableDevice();
//    config = NRF_READ_REG(CONFIG);
}

uint8_t isDataAvailable (int num)
{
    uint8_t status = NRF_READ_REG(STATUS);
uint8_t statustwo = NRF_READ_REG(FIFO_STATUS);
    if (status&(1<<6)&&(status&(num<<1)))
    {

        NRF_WR_REG(STATUS, (1<<6));

        return 1;
    }


    return 0;
}
void NRF24_Receive(uint8_t *data, uint8_t len)
{
    uint8_t cmdtosend = R_RX_PAYLOAD;
//    uint32_t datarx;
    enableSpiCs();

    writeSpi0Data(cmdtosend);
    readSpi0Data();

//    enableSpiCs();

//      writeSpi0Data(reg);         // Send register address
//      readSpi0Data();        // Dummy write to clock in response
//      writeSpi0Data(0);
//      datarx = readSpi0Data();     // Read response

//      disableSpiCs();
//
    int i;
    for (i = 0; i < 13; i++)
    {
        writeSpi0Data(0);       // Dummy byte to clock in data
        data[i] = (uint8_t)readSpi0Data();  // Store received byte
    }
//
//    disableSpiCs();
disableSpiCs();

//    _delay_cycles(10);
    waitMicrosecond(10);

    NRF_COMMAND(FLUSH_RX);  // Clear RX FIFO
}







