#define EIDSP_QUANTIZE_FILTERBANK 0
#include <EngineMonitor_inferencing.h>

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "driver/i2s_pdm.h"

// ─── Pin definitions ─────────────────────────────────────────────
#define SDA_PIN      5    // D4 -> OLED SDA
#define SCL_PIN      6    // D5 -> OLED SCL
#define LED_PIN      9    // D10 -> fault indicator LED

// XIAO ESP32S3 Sense built-in PDM microphone
#define I2S_MIC_CLK  GPIO_NUM_42
#define I2S_MIC_DATA GPIO_NUM_41

// ─── OLED ────────────────────────────────────────────────────────
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  64
#define OLED_ADDRESS  0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ─── Audio inference buffer ──────────────────────────────────────
typedef struct {
    int16_t         *buffer;
    volatile uint8_t buf_ready;
    uint32_t         buf_count;
    uint32_t         n_samples;
} inference_t;

static inference_t inference;
static bool        record_running = false;
static int16_t     i2s_read_buf[512];
static i2s_chan_handle_t rx_chan = NULL;

// ─── I2S capture task ────────────────────────────────────────────
static void i2s_capture_task(void *arg) {
    size_t bytes_read;
    uint32_t loop_count = 0;
    while (record_running) {
        i2s_channel_read(rx_chan, i2s_read_buf, sizeof(i2s_read_buf), &bytes_read, 100);
        if (loop_count++ % 50 == 0) {
            Serial.printf("[MIC] bytes_read=%d buf_count=%d\n", bytes_read, inference.buf_count);
        }
        int samples = bytes_read / 2;
        for (int i = 0; i < samples; i++) {
            i2s_read_buf[i] = (int16_t)(i2s_read_buf[i]) * 8;
            inference.buffer[inference.buf_count++] = i2s_read_buf[i];
            if (inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
    vTaskDelete(NULL);
}

// ─── Microphone initialization (IDF 5.x PDM API) ─────────────────
static bool mic_init(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (!inference.buffer) {
        Serial.println("ERROR: not enough memory for audio buffer");
        return false;
    }
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    if (i2s_new_channel(&chan_cfg, NULL, &rx_chan) != ESP_OK) {
        Serial.println("ERROR: i2s_new_channel failed");
        return false;
    }

    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg  = I2S_PDM_RX_CLK_DEFAULT_CONFIG(EI_CLASSIFIER_FREQUENCY),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = I2S_MIC_CLK,
            .din = I2S_MIC_DATA,
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };

    if (i2s_channel_init_pdm_rx_mode(rx_chan, &pdm_rx_cfg) != ESP_OK) {
        Serial.println("ERROR: i2s_channel_init_pdm_rx_mode failed");
        return false;
    }

    if (i2s_channel_enable(rx_chan) != ESP_OK) {
        Serial.println("ERROR: i2s_channel_enable failed");
        return false;
    }

    record_running = true;
    xTaskCreate(i2s_capture_task, "mic_task", 1024 * 32, NULL, 10, NULL);
    return true;
}

// ─── Edge Impulse audio signal callback ──────────────────────────
static int audio_signal_get_data(size_t offset, size_t length, float *out) {
    numpy::int16_to_float(&inference.buffer[offset], out, length);
    return 0;
}

// ─── OLED display ────────────────────────────────────────────────
void showResult(const char *label, float confidence) {
    String l = String(label);
    l.toLowerCase();

    bool   isFault = false;
    String shortLabel;

    if (l.indexOf("air leak") >= 0) {
        isFault    = true;
        shortLabel = "AIR LEAK!";
    } else if (l.indexOf("oil cap") >= 0) {
        isFault    = true;
        shortLabel = "OIL CAP OFF";
    } else if (l.indexOf("normal") >= 0 || l.indexOf("idling") >= 0) {
        isFault    = false;
        shortLabel = "NORMAL";
    } else {
        isFault    = false;
        shortLabel = "STANDBY";
    }

    digitalWrite(LED_PIN, isFault ? HIGH : LOW);

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("  Engine Monitor");
    display.drawLine(0, 9, 127, 9, SSD1306_WHITE);

    display.setTextSize(2);
    display.setCursor(isFault ? 4 : 16, 13);
    display.println(shortLabel);

    display.drawLine(0, 34, 127, 34, SSD1306_WHITE);

    display.setTextSize(1);
    display.setCursor(0, 37);
    display.println(label);

    display.setCursor(0, 52);
    display.print("Conf: ");
    display.print((int)(confidence * 100));
    display.println("%");

    display.display();

    Serial.printf("[Result] %s  %.1f%%\n", label, confidence * 100);
}

// ─── setup ───────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Wire.begin(SDA_PIN, SCL_PIN);

    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
        Serial.println("ERROR: OLED init failed");
        while (true) delay(1000);
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(20, 20);
    display.println("Engine Monitor");
    display.setCursor(30, 36);
    display.println("Starting...");
    display.display();

    Serial.println("Initializing microphone...");
    if (!mic_init(EI_CLASSIFIER_RAW_SAMPLE_COUNT)) {
        display.clearDisplay();
        display.setCursor(0, 20);
        display.println("MIC INIT FAILED");
        display.display();
        while (true) delay(1000);
    }

    Serial.println("Ready, listening...");
    delay(500);
}

// ─── loop ────────────────────────────────────────────────────────
void loop() {
    static unsigned long holdUntil = 0;

    while (inference.buf_ready == 0) delay(5);
    inference.buf_ready = 0;

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data     = &audio_signal_get_data;

    ei_impulse_result_t result = {0};
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    if (err != EI_IMPULSE_OK) {
        Serial.printf("Inference error: %d\n", err);
        return;
    }

    int   best_idx = 0;
    float best_val = 0.0f;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > best_val) {
            best_val = result.classification[i].value;
            best_idx = i;
        }
    }

    if (best_val < 0.65f) {
        Serial.println("Low confidence, skipping frame");
        return;
    }

    // Hold display for 3 seconds before updating
    if (millis() < holdUntil) return;
    holdUntil = millis() + 3000;

    showResult(ei_classifier_inferencing_categories[best_idx], best_val);
}
