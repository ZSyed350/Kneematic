#include "print.h"
#include "pcap04_defs.h"
#include "mux_control.h"
#include "Arduino.h"

void printResults() {
    #define PCAP_CONVERSION_NUMBER 134217728
    static int print_counter = 0;
    
    // Only print every 50 measurements (every 500ms at 100Hz)
    print_counter++;
    if (print_counter < 50) {
        return;
    }
    print_counter = 0;
    
    // Print header
    Serial.println("\n--- PCAP Measurements (100Hz sampling) ---");
    Serial.println("Chip | S0       | S1       | S2       | S3       | S4       | S5");
    Serial.println("-----|----------|----------|----------|----------|----------|----------");
    
    //int chip = PCAP_CHIP_1;

    /*
    for (int sensor = 0; sensor < NUM_SENSORS_PER_CHIP; sensor++) {
            float value = chip_data[chip].raw[sensor]/PCAP_CONVERSION_NUMBER;
            //float value = (1000*(chip_data[chip].raw[sensor] - chip_data[chip].offset[sensor])/PCAP_CONVERSION_NUMBER);

            //uint32_t value = chip_data[chip].raw[sensor] - chip_data[chip].offset[sensor];
            
            // Print value with padding
            if (value < 10000000) Serial.print(" ");
            if (value < 1000000) Serial.print(" ");
            if (value < 100000) Serial.print(" ");
            if (value < 10000) Serial.print(" ");
            if (value < 1000) Serial.print(" ");
            if (value < 100) Serial.print(" ");
            if (value < 10) Serial.print(" ");
            
            Serial.print(value);
            Serial.print(" | ");
        }
        Serial.println();
    */

    // Print data for each chip
    for (int chip = PCAP_CHIP_1; chip <= PCAP_CHIP_2; chip++) {
        Serial.print("  ");
        Serial.print(chip + 1);
        Serial.print("  | ");
        
        for (int sensor = 0; sensor < NUM_SENSORS_PER_CHIP; sensor++) {
            //float value = chip_data[chip].raw[sensor];
            float value = (1000*(chip_data[chip].raw[sensor] - chip_data[chip].offset[sensor])/PCAP_CONVERSION_NUMBER);
            //uint32_t value = chip_data[chip].raw[sensor] - chip_data[chip].offset[sensor];
            
            // Print value with padding
            if (value < 10000000) Serial.print(" ");
            if (value < 1000000) Serial.print(" ");
            if (value < 100000) Serial.print(" ");
            if (value < 10000) Serial.print(" ");
            if (value < 1000) Serial.print(" ");
            if (value < 100) Serial.print(" ");
            if (value < 10) Serial.print(" ");
            
            Serial.print(value);
            Serial.print(" | ");
        }
        Serial.println();
    }
}

void printDiagnostics() {

    Serial.println("\n--- Pin Configuration ---");
    Serial.print("MUX_S0_PIN (2): GPIO ");
    Serial.println(2);
    Serial.print("MUX_S1_PIN (3): GPIO ");
    Serial.println(3);
    Serial.print("MUX_S2_PIN (4): GPIO ");
    Serial.println(4);
    Serial.print("MUX_S3_PIN (5): GPIO ");
    Serial.println(5);
    Serial.println("CS Control: Multiplexer (COMMON_I/O = GND, pull-ups on outputs)");

    Serial.println("\n--- SPI Pin Configuration ---");
    Serial.println("Hardware SPI (VSPI on ESP32C3):");
    Serial.println("  Expected MOSI: GPIO 51");
    Serial.println("  Expected MISO: GPIO 50");
    Serial.println("  Expected SCK:  GPIO 52");
    Serial.print("  Actual MOSI: GPIO ");
    Serial.println(MOSI);
    Serial.print("  Actual MISO: GPIO ");
    Serial.println(MISO);
    Serial.print("  Actual SCK:  GPIO ");
    Serial.println(SCK);

    Serial.println("\n--- Configuration ---");
    Serial.print("Number of PCAP chips: ");
    Serial.println(NUM_PCAP_CHIPS);
    Serial.print("Sensors per chip: ");
    Serial.println(NUM_SENSORS_PER_CHIP);
    Serial.print("Config size: ");
    Serial.print(PCAP_CONFIG_SIZE);
    Serial.println(" bytes");
    Serial.print("Firmware size: ");
    Serial.print(PCAP_FW_SIZE);
    Serial.println(" bytes");
    Serial.println("========================================");
}
