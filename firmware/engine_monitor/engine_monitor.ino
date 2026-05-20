#define EIDSP_QUANTIZE_FILTERBANK 0
#include <EngineMonitorUpdate_inferencing.h>

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "driver/i2s_pdm.h"

// ─── Pin definitions ─────────────────────────────────────────────
#define SDA_PIN      5
#define SCL_PIN      6
#define LED_PIN      9

#define I2S_MIC_CLK  GPIO_NUM_42
#define I2S_MIC_DATA GPIO_NUM_41

// ─── Tunable parameters ──────────────────────────────────────────
#define RMS_THRESHOLD   0.0f      // disabled for diagnostics
#define CONF_THRESHOLD  0.75f     // minimum confidence to accept result
#define CONFIRM_NEEDED  1         // frames needed to confirm detection

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

static inference_t   inference;
static bool          record_running = false;
static int16_t       i2s_read_buf[512];
static i2s_chan_handle_t rx_chan = NULL;

// ─── RMS of current audio frame ──────────────────────────────────
float computeRMS() {
    float sum = 0.0f;
    for (uint32_t i = 0; i < inference.n_samples; i++) {
        float s = (float)inference.buffer[i];
        sum += s * s;
    }
    return sqrtf(sum / inference.n_samples);
}

// ─── I2S capture task ────────────────────────────────────────────
static void i2s_capture_task(void *arg) {
    size_t bytes_read;
    while (record_running) {
        i2s_channel_read(rx_chan, i2s_read_buf, sizeof(i2s_read_buf), &bytes_read, 100);
        int samples = bytes_read / 2;
        for (int i = 0; i < samples; i++) {
            i2s_read_buf[i] = i2s_read_buf[i]; // no amplification
            inference.buffer[inference.buf_count++] = i2s_read_buf[i];
            if (inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
    vTaskDelete(NULL);
}

// ─── Microphone init ─────────────────────────────────────────────
static bool mic_init(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (!inference.buffer) return false;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    if (i2s_new_channel(&chan_cfg, NULL, &rx_chan) != ESP_OK) return false;

    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg  = I2S_PDM_RX_CLK_DEFAULT_CONFIG(EI_CLASSIFIER_FREQUENCY),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = I2S_MIC_CLK,
            .din = I2S_MIC_DATA,
            .invert_flags = { .clk_inv = false },
        },
    };
    if (i2s_channel_init_pdm_rx_mode(rx_chan, &pdm_rx_cfg) != ESP_OK) return false;
    if (i2s_channel_enable(rx_chan) != ESP_OK) return false;

    record_running = true;
    xTaskCreate(i2s_capture_task, "mic_task", 1024 * 32, NULL, 10, NULL);
    return true;
}

// ─── EI audio callback ───────────────────────────────────────────
static int audio_signal_get_data(size_t offset, size_t length, float *out) {
    numpy::int16_to_float(&inference.buffer[offset], out, length);
    return 0;
}

// ─── OLED helpers ────────────────────────────────────────────────
void drawTitleBar() {
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(22, 2);
    display.println("Engine Monitor");
    display.drawLine(0, 11, 127, 11, SSD1306_WHITE);
}

// ─── Animated sound bars (bottom center) ─────────────────────────
// Traveling wave pattern, 8 frames
static const int8_t SOUND_BARS[8][4] = {
    { 8,  5,  4,  5},
    {11,  8,  5,  4},
    {12, 11,  8,  5},
    {11, 12, 11,  8},
    { 8, 11, 12, 11},
    { 5,  8, 11, 12},
    { 4,  5,  8, 11},
    { 5,  4,  5,  8},
};

void drawSoundBars(uint8_t frame) {
    const int barW    = 5;
    const int barGap  = 4;
    const int barN    = 4;
    const int baseY   = 62;
    const int startX  = (128 - (barN * barW + (barN - 1) * barGap)) / 2; // centered

    // clear bar zone
    display.fillRect(startX - 1, baseY - 13, barN * (barW + barGap) + 1, 15, SSD1306_BLACK);

    for (int i = 0; i < barN; i++) {
        int h = SOUND_BARS[frame % 8][i];
        int x = startX + i * (barW + barGap);
        display.fillRect(x, baseY - h + 1, barW, h, SSD1306_WHITE);
        // small rounded cap on top
        display.drawPixel(x,         baseY - h,     SSD1306_WHITE);
        display.drawPixel(x + barW - 1, baseY - h,  SSD1306_WHITE);
    }
}

// ─── Wrench icon bitmap (16x16) ──────────────────────────────────
static const unsigned char WRENCH_ICON[] PROGMEM = {
    0x00, 0x00, 0x00, 0x00, 0x30, 0x00, 0x1c, 0x00,
    0x0e, 0x00, 0x8e, 0x00, 0xde, 0x00, 0xff, 0x00,
    0x7f, 0x80, 0x07, 0xc0, 0x03, 0xe0, 0x01, 0xe0,
    0x00, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

// horizontally mirrored version of WRENCH_ICON
static const unsigned char WRENCH_ICON_FLIP[] PROGMEM = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x38,
    0x00, 0x70, 0x00, 0x71, 0x00, 0x7b, 0x00, 0xff,
    0x01, 0xfe, 0x03, 0xe0, 0x07, 0xc0, 0x07, 0x80,
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

void drawWrench(int x, int y) {
    display.drawBitmap(x, y, WRENCH_ICON, 16, 16, SSD1306_WHITE);
}

void drawWrenchFlip(int x, int y) {
    display.drawBitmap(x, y, WRENCH_ICON_FLIP, 16, 16, SSD1306_WHITE);
}

// ─── Opening / home page ─────────────────────────────────────────
void showHome() {
    display.clearDisplay();
    drawTitleBar();

    drawWrenchFlip(2, 46);
    drawWrench(110, 46);

    // Main label (centered: 7 chars * 12px = 84px, x = (128-84)/2 = 22)
    display.setTextSize(2);
    display.setCursor(22, 17);
    display.println("AI MECH");

    // Subtitle (centered: 15 chars * 6px = 90px, x = (128-90)/2 = 19)
    display.setTextSize(1);
    display.setCursor(19, 37);
    display.println("Sound Detection");

    // Sound bars drawn on first call with frame 0
    drawSoundBars(0);

    display.display();
    digitalWrite(LED_PIN, LOW);
}

void showResult(const char *label, float confidence) {
    String l = String(label);
    l.toUpperCase();

    bool   isFault = false;
    String bigText, subText;

    if (l.indexOf("ENGINE_ISSUE") >= 0) {
        isFault = true;  bigText = "ENGINE";   subText = "Engine fault detected";
    } else if (l.indexOf("IGNITION") >= 0) {
        isFault = true;  bigText = "IGNITION"; subText = "Ignition issue";
    } else if (l.indexOf("NORMAL") >= 0 || l.indexOf("STRATUP") >= 0) {
        isFault = false; bigText = "NORMAL";   subText = "Engine OK";
    } else {
        isFault = false; bigText = "UNKNOWN";  subText = String(label);
    }

    digitalWrite(LED_PIN, isFault ? HIGH : LOW);

    display.clearDisplay();
    drawTitleBar();

    int16_t x = (128 - (int16_t)(bigText.length() * 12)) / 2;
    display.setTextSize(2);
    display.setCursor(max((int16_t)0, x), 17);
    display.println(bigText);

    display.setTextSize(1);
    display.setCursor(0, 38);
    display.println(subText);
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

    Serial.println("Ready.");
    delay(500);
    showHome();
}

// ─── loop ────────────────────────────────────────────────────────
void loop() {
    static unsigned long holdUntil    = 0;
    static bool          idleShown    = false;
    static int           confirmCount  = 0;
    static int           confirmIdx    = -1;
    static uint8_t       animFrame     = 0;

    while (inference.buf_ready == 0) delay(5);
    inference.buf_ready = 0;

    // ── Return to home page once hold expires ────────────────────
    if (millis() >= holdUntil && !idleShown) {
        showHome();
        idleShown    = true;
        animFrame    = 0;
        confirmCount = 0;
        confirmIdx   = -1;
    }

    // ── Animate sound bars while on home page ─────────────────────
    if (idleShown) {
        drawSoundBars(animFrame++);
        display.display();
    }

    // ── Volume gate: skip inference if too quiet ──────────────────
    float rms = computeRMS();
    Serial.printf("[RMS] %.0f\n", rms);

    if (rms < RMS_THRESHOLD) {
        return;
    }

    // ── Run inference ─────────────────────────────────────────────
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data     = &audio_signal_get_data;

    ei_impulse_result_t result = {0};
    if (run_classifier(&signal, &result, false) != EI_IMPULSE_OK) return;

    int   best_idx = 0;
    float best_val = 0.0f;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > best_val) {
            best_val = result.classification[i].value;
            best_idx = i;
        }
    }

    Serial.printf("[Infer] %s  %.1f%%\n",
                  ei_classifier_inferencing_categories[best_idx], best_val * 100);

    // ── Confidence gate ───────────────────────────────────────────
    if (best_val < CONF_THRESHOLD) {
        confirmCount = 0;
        confirmIdx   = -1;
        return;
    }

    // ── Consecutive-frame confirmation ────────────────────────────
    if (best_idx == confirmIdx) {
        confirmCount++;
    } else {
        confirmIdx   = best_idx;
        confirmCount = 1;
    }
    Serial.printf("[Confirm] %d/%d\n", confirmCount, CONFIRM_NEEDED);

    if (confirmCount < CONFIRM_NEEDED) return;
    if (millis() < holdUntil) return;

    confirmCount = 0;
    holdUntil    = millis() + 5000;
    idleShown    = false;
    showResult(ei_classifier_inferencing_categories[best_idx], best_val);
}
