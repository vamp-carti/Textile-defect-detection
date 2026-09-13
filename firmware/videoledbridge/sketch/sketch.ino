#include <Arduino_LED_Matrix.h>
#include "frames.h"
#include "Arduino_RouterBridge.h"

Arduino_LED_Matrix matrix;
volatile int led_state = 0;

void set_normal() { led_state = 0; }
void set_defect() { led_state = 1; }
void set_idle()   { led_state = 2; }

void setup() {
    matrix.begin();
    Bridge.begin();
    Bridge.provide_safe("led_normal", set_normal);
    Bridge.provide_safe("led_defect", set_defect);
    Bridge.provide_safe("led_idle",   set_idle);
}

void loop() {
    if (led_state == 0) {
        for (int i = 0; i < arrows_frameCount; i++) {
            matrix.draw(arrows_animation[i]);
            delay(100);
            if (led_state != 0) return;
        }
    } else if (led_state == 1) {
        for (int i = 0; i < cross_frameCount; i++) {
            matrix.draw(cross_animation[i]);
            delay(100);
            if (led_state != 1) return;
        }
    } else {
        matrix.clear();
        delay(100);
    }
}
