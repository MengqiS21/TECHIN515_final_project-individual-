#define EIDSP_QUANTIZE_FILTERBANK 0
#include <EngineMonitor_inferencing.h>

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "driver/i2s.h"

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

// ─── I2S capture task (runs continuously on its own thread) ──────
static void i2s_capture_task(void *arg) {
    size_t bytes_read;
    while (record_running) {
        i2s_read(I2S_NUM_0, i2s_read_buf, sizeof(i2s_read_buf), &bytes_read, portMAX_DELAY);
        int samples = bytes_read / 2;
        for (int i = 0; i < samples; i++) {
            inference.buffer[inference.buf_count++] = i2s_read_buf[i];
            if (inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
    vTaskDelete(NULL);
}

// ─── Microphone initialization ───────────────────────────────────
static bool mic_init(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (!inference.buffer) {
        Serial.println("ERROR: not enough memory for audio buffer");
        return false;
    }
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
        .sample_rate          = EI_CLASSIFIER_FREQUENCY,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_PCM_SHORT,
        .intr_alloc_flags     = 0,
        .dma_buf_count        = 8,
        .dma_buf_len          = 512,
        .use_apll             = false,
    };
    i2s_pin_config_t pins = {
        .bck_io_num   = I2S_PIN_NO_CHANGE,
        .ws_io_num    = I2S_MIC_CLK,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = I2S_MIC_DATA,
    };

    if (i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL) != ESP_OK) return false;
    if (i2s_set_pin(I2S_NUM_0, &pins)               != ESP_OK) return false;

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

    // LED on when fault detected
    digitalWrite(LED_PIN, isFault ? HIGH : LOW);

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    // Title bar
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("  Engine Monitor");
    display.drawLine(0, 9, 127, 9, SSD1306_WHITE);

    // Large status text
    display.setTextSize(2);
    display.setCursor(isFault ? 4 : 16, 13);
    display.println(shortLabel);

    display.drawLine(0, 34, 127, 34, SSD1306_WHITE);

    // Raw label (small text)
    display.setTextSize(1);
    display.setCursor(0, 37);
    display.println(label);

    // Confidence score
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

    // Initialize I2C with specified pins
    Wire.begin(SDA_PIN, SCL_PIN);

    // Initialize OLED
    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
        Serial.println("ERROR: OLED init failed, check wiring and I2C address");
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

    // Initialize microphone
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
    // Wait for a full audio frame
    while (inference.buf_ready == 0) delay(5);
    inference.buf_ready = 0;

    // Build EI signal
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data     = &audio_signal_get_data;

    // Run inference
    ei_impulse_result_t result = {0};
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    if (err != EI_IMPULSE_OK) {
        Serial.printf("Inference error: %d\n", err);
        return;
    }

    // Find highest confidence label
    int   best_idx = 0;
    float best_val = 0.0f;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > best_val) {
            best_val = result.classification[i].value;
            best_idx = i;
        }
    }

    // Skip if confidence too low to avoid false positives
    if (best_val < 0.65f) {
        Serial.println("Low confidence, skipping frame");
        return;
    }

    showResult(ei_classifier_inferencing_categories[best_idx], best_val);
}
