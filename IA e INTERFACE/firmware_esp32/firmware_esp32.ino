
#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include <Adafruit_NeoPixel.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "painel_tcc_esp.h"

// Rede Wi-Fi. Configure antes de gravar a placa.
const char* WIFI_SSID = "SUA_REDE_WIFI";
const char* WIFI_PASSWORD = "SUA_SENHA_WIFI";

// Camera OV2640 - SunFounder
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM 33
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

// TFT ILI9341 - pinagem original do projeto
#define TFT_CS   15
#define TFT_DC    4
#define TFT_RST   2
#define TFT_SCK  14
#define TFT_MOSI 13

// Os pinos do TFT compartilham o barramento do microSD.
// Nao utilizar o cartao SD simultaneamente ao display.
SPIClass hspi(HSPI);
Adafruit_ILI9341 tft(&hspi, TFT_DC, TFT_CS, TFT_RST);

// Anel de iluminacao WS2812B
#define NEO_PIN       12
#define NUMPIXELS     24
#define NEO_INTERVAL  80

Adafruit_NeoPixel pixels(
  NUMPIXELS,
  NEO_PIN,
  NEO_GRB + NEO_KHZ800
);

enum NeoMode {
  NEO_OFF,
  NEO_SOLID,
  NEO_DUAL,
  NEO_CHASE
};

volatile NeoMode neoMode = NEO_DUAL;
volatile uint32_t neoColor1 = 0xFF0000;
volatile uint32_t neoColor2 = 0xFA9696;
volatile uint8_t neoBrightness = 50;
volatile bool neoDirty = true;

uint32_t neoPosition = 0;
unsigned long lastNeo = 0;

// Estados do display
enum EstadoTFT : uint8_t {
  TFT_ESPERANDO = 0,
  TFT_CONFORME = 1,
  TFT_NAO_CONFORME = 2
};

volatile uint8_t estadoTFTRecebido = TFT_ESPERANDO;
uint8_t estadoTFTExibido = 255;
bool tftPronto = false;

// Camera e estatisticas
bool cameraOK = false;

volatile uint32_t streamFrameTotal = 0;
volatile uint32_t streamClients = 0;
volatile uint32_t lastJpegBytes = 0;
volatile uint16_t lastWidth = 0;
volatile uint16_t lastHeight = 0;

float streamFPS = 0.0f;
uint32_t previousStreamFrames = 0;
unsigned long statsTime = 0;

httpd_handle_t camera_httpd = NULL;
httpd_handle_t stream_httpd = NULL;
SemaphoreHandle_t cameraMutex = NULL;

// =====================================================
// TFT
// =====================================================

void textoCentralizado(const char* texto, int16_t y) {
  const uint8_t escala = 3;
  int16_t largura = strlen(texto) * 6 * escala;
  int16_t x = (tft.width() - largura) / 2;

  tft.setTextSize(escala);
  tft.setTextColor(ILI9341_WHITE);
  tft.setTextWrap(false);
  tft.setCursor(max(0, (int)x), y);
  tft.print(texto);
}

void desenharTFT(uint8_t estado) {
  if (!tftPronto) return;

  // CP437 permite exibir os caracteres ç e ã.
  const char* esperando = "Esperando Pe\x87" "a";
  const char* conforme = "Pe\x87" "a Conforme";
  const char* peca = "Pe\x87" "a";
  const char* naoConforme = "N\xC6" "o Conforme";

  switch (estado) {
    case TFT_ESPERANDO:
      tft.fillScreen(tft.color565(125, 125, 125));
      textoCentralizado(esperando, 108);
      break;

    case TFT_CONFORME:
      tft.fillScreen(ILI9341_GREEN);
      textoCentralizado(conforme, 108);
      break;

    case TFT_NAO_CONFORME:
      tft.fillScreen(ILI9341_RED);
      textoCentralizado(peca, 80);
      textoCentralizado(naoConforme, 128);
      break;

    default:
      return;
  }

  estadoTFTExibido = estado;
}

void iniciarTFT() {
  hspi.begin(TFT_SCK, -1, TFT_MOSI, TFT_CS);

  // Clock conservador para estabilidade nos fios do projeto.
  tft.begin(20000000);
  tft.setRotation(1);
  tft.cp437(true);
  tft.setTextWrap(false);

  tftPronto = true;
  estadoTFTRecebido = TFT_ESPERANDO;
  desenharTFT(TFT_ESPERANDO);

  Serial.println("TFT ILI9341 inicializado");
}

void atualizarTFT() {
  if (!tftPronto) return;

  uint8_t novoEstado = estadoTFTRecebido;

  if (novoEstado != estadoTFTExibido) {
    desenharTFT(novoEstado);
  }
}

const char* nomeEstadoTFT(uint8_t estado) {
  switch (estado) {
    case TFT_CONFORME:
      return "conforme";

    case TFT_NAO_CONFORME:
      return "nao_conforme";

    default:
      return "esperando";
  }
}

// =====================================================
// POST /tft
// =====================================================

bool interpretarEstadoTFT(
  const String& corpo,
  uint8_t& estado
) {
  String valor = corpo;
  valor.trim();

  // Aceita JSON: {"estado":"conforme"}
  if (valor.startsWith("{")) {
    int chave = valor.indexOf("\"estado\"");

    if (chave < 0) return false;

    int doisPontos = valor.indexOf(':', chave + 8);

    if (doisPontos < 0) return false;

    int inicio = valor.indexOf('"', doisPontos + 1);

    if (inicio < 0) return false;

    int fim = valor.indexOf('"', inicio + 1);

    if (fim < 0) return false;

    valor = valor.substring(inicio + 1, fim);
  }

  valor.trim();
  valor.toLowerCase();

  if (valor == "esperando") {
    estado = TFT_ESPERANDO;
    return true;
  }

  if (valor == "conforme") {
    estado = TFT_CONFORME;
    return true;
  }

  if (valor == "nao_conforme") {
    estado = TFT_NAO_CONFORME;
    return true;
  }

  return false;
}

esp_err_t tftHandler(httpd_req_t* req) {
  const size_t tamanho = req->content_len;

  if (tamanho == 0 || tamanho > 96) {
    httpd_resp_send_err(
      req,
      HTTPD_400_BAD_REQUEST,
      "Corpo da requisicao invalido"
    );
    return ESP_FAIL;
  }

  char corpo[97];
  size_t recebido = 0;

  while (recebido < tamanho) {
    int n = httpd_req_recv(
      req,
      corpo + recebido,
      tamanho - recebido
    );

    if (n == HTTPD_SOCK_ERR_TIMEOUT) {
      continue;
    }

    if (n <= 0) {
      return ESP_FAIL;
    }

    recebido += (size_t)n;
  }

  corpo[tamanho] = '\0';

  uint8_t novoEstado;

  if (!interpretarEstadoTFT(String(corpo), novoEstado)) {
    httpd_resp_send_err(
      req,
      HTTPD_400_BAD_REQUEST,
      "Estado TFT desconhecido"
    );
    return ESP_FAIL;
  }

  // O desenho ocorre no loop principal.
  estadoTFTRecebido = novoEstado;

  Serial.printf(
    "POST /tft: %s\n",
    nomeEstadoTFT(novoEstado)
  );

  httpd_resp_set_type(req, "application/json");

  return httpd_resp_sendstr(
    req,
    "{\"ok\":true}"
  );
}

// =====================================================
// PARAMETROS E RESOLUCOES
// =====================================================

bool getParam(
  httpd_req_t* req,
  const char* name,
  char* output,
  size_t outputSize
) {
  char query[192];

  if (
    httpd_req_get_url_query_str(
      req,
      query,
      sizeof(query)
    ) != ESP_OK
  ) {
    return false;
  }

  return httpd_query_key_value(
    query,
    name,
    output,
    outputSize
  ) == ESP_OK;
}

framesize_t frameSizeFromString(const char* value) {
  if (!strcmp(value, "QVGA")) return FRAMESIZE_QVGA;
  if (!strcmp(value, "VGA"))  return FRAMESIZE_VGA;
  if (!strcmp(value, "SVGA")) return FRAMESIZE_SVGA;
  if (!strcmp(value, "XGA"))  return FRAMESIZE_XGA;
  if (!strcmp(value, "SXGA")) return FRAMESIZE_SXGA;
  if (!strcmp(value, "UXGA")) return FRAMESIZE_UXGA;

  return FRAMESIZE_INVALID;
}

const char* frameSizeToString(framesize_t size) {
  switch (size) {
    case FRAMESIZE_QVGA: return "QVGA 320x240";
    case FRAMESIZE_VGA:  return "VGA 640x480";
    case FRAMESIZE_SVGA: return "SVGA 800x600";
    case FRAMESIZE_XGA:  return "XGA 1024x768";
    case FRAMESIZE_SXGA: return "SXGA 1280x1024";
    case FRAMESIZE_UXGA: return "UXGA 1600x1200";
    default:             return "Outra";
  }
}

const char* neoModeString(NeoMode mode) {
  switch (mode) {
    case NEO_SOLID: return "solid";
    case NEO_DUAL:  return "dual";
    case NEO_CHASE: return "chase";
    default:        return "off";
  }
}

// =====================================================
// PAGINA PRINCIPAL E STATUS
// =====================================================

esp_err_t indexHandler(httpd_req_t* req) {
  httpd_resp_set_type(
    req,
    "text/html; charset=utf-8"
  );

  httpd_resp_set_hdr(
    req,
    "Cache-Control",
    "no-store"
  );

  return httpd_resp_send(
    req,
    INDEX_HTML,
    HTTPD_RESP_USE_STRLEN
  );
}

esp_err_t statusHandler(httpd_req_t* req) {
  sensor_t* sensor = esp_camera_sensor_get();

  const char* size = sensor
    ? frameSizeToString(sensor->status.framesize)
    : "N/A";

  int quality = sensor ? sensor->status.quality : 0;
  int brightness = sensor ? sensor->status.brightness : 0;
  int contrast = sensor ? sensor->status.contrast : 0;
  int saturation = sensor ? sensor->status.saturation : 0;
  int mirror = sensor ? sensor->status.hmirror : 0;
  int flip = sensor ? sensor->status.vflip : 0;

  int rssi = WiFi.status() == WL_CONNECTED
    ? WiFi.RSSI()
    : 0;

  char json[560];

  snprintf(
    json,
    sizeof(json),
    "{"
      "\"stream_fps\":%.2f,"
      "\"clients\":%u,"
      "\"rssi\":%d,"
      "\"framesize\":\"%s\","
      "\"quality\":%d,"
      "\"brightness\":%d,"
      "\"contrast\":%d,"
      "\"saturation\":%d,"
      "\"hmirror\":%d,"
      "\"vflip\":%d,"
      "\"neo_mode\":\"%s\","
      "\"color1\":\"%06lX\","
      "\"color2\":\"%06lX\","
      "\"neo_brightness\":%u,"
      "\"jpeg_bytes\":%u,"
      "\"tft_estado\":\"%s\""
    "}",
    streamFPS,
    (unsigned int)streamClients,
    rssi,
    size,
    quality,
    brightness,
    contrast,
    saturation,
    mirror,
    flip,
    neoModeString(neoMode),
    (unsigned long)(neoColor1 & 0xFFFFFFUL),
    (unsigned long)(neoColor2 & 0xFFFFFFUL),
    (unsigned int)neoBrightness,
    (unsigned int)lastJpegBytes,
    nomeEstadoTFT(estadoTFTRecebido)
  );

  httpd_resp_set_type(
    req,
    "application/json"
  );

  httpd_resp_set_hdr(
    req,
    "Cache-Control",
    "no-store"
  );

  return httpd_resp_sendstr(req, json);
}

// =====================================================
// CONTROLE DA CAMERA
// =====================================================

esp_err_t controlHandler(httpd_req_t* req) {
  char key[32];
  char val[32];

  if (
    !getParam(req, "var", key, sizeof(key)) ||
    !getParam(req, "val", val, sizeof(val))
  ) {
    httpd_resp_send_err(
      req,
      HTTPD_400_BAD_REQUEST,
      "Parametros invalidos"
    );
    return ESP_FAIL;
  }

  sensor_t* sensor = esp_camera_sensor_get();

  if (!sensor) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  if (
    cameraMutex &&
    xSemaphoreTake(
      cameraMutex,
      pdMS_TO_TICKS(2500)
    ) != pdTRUE
  ) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  int result = -1;

  if (!strcmp(key, "framesize")) {
    framesize_t size = frameSizeFromString(val);

    if (size != FRAMESIZE_INVALID) {
      result = sensor->set_framesize(sensor, size);
    }
  }
  else if (!strcmp(key, "quality")) {
    result = sensor->set_quality(
      sensor,
      constrain(atoi(val), 0, 63)
    );
  }
  else if (!strcmp(key, "brightness")) {
    result = sensor->set_brightness(
      sensor,
      constrain(atoi(val), -2, 2)
    );
  }
  else if (!strcmp(key, "contrast")) {
    result = sensor->set_contrast(
      sensor,
      constrain(atoi(val), -2, 2)
    );
  }
  else if (!strcmp(key, "saturation")) {
    result = sensor->set_saturation(
      sensor,
      constrain(atoi(val), -2, 2)
    );
  }
  else if (!strcmp(key, "hmirror")) {
    result = sensor->set_hmirror(
      sensor,
      atoi(val) ? 1 : 0
    );
  }
  else if (!strcmp(key, "vflip")) {
    result = sensor->set_vflip(
      sensor,
      atoi(val) ? 1 : 0
    );
  }

  if (cameraMutex) {
    xSemaphoreGive(cameraMutex);
  }

  if (result != 0) {
    httpd_resp_send_err(
      req,
      HTTPD_500_INTERNAL_SERVER_ERROR,
      "Falha ao alterar camera"
    );
    return ESP_FAIL;
  }

  return httpd_resp_sendstr(req, "OK");
}

// =====================================================
// CONTROLE DO NEOPIXEL
// =====================================================

esp_err_t neoHandler(httpd_req_t* req) {
  char val[32];

  if (
    getParam(req, "mode", val, sizeof(val))
  ) {
    if (!strcmp(val, "off")) {
      neoMode = NEO_OFF;
    }
    else if (!strcmp(val, "solid")) {
      neoMode = NEO_SOLID;
    }
    else if (!strcmp(val, "dual")) {
      neoMode = NEO_DUAL;
    }
    else if (!strcmp(val, "chase")) {
      neoMode = NEO_CHASE;
    }
  }

  if (
    getParam(req, "color1", val, sizeof(val))
  ) {
    neoColor1 =
      strtoul(val, nullptr, 16) & 0xFFFFFFUL;
  }

  if (
    getParam(req, "color2", val, sizeof(val))
  ) {
    neoColor2 =
      strtoul(val, nullptr, 16) & 0xFFFFFFUL;
  }

  if (
    getParam(req, "brightness", val, sizeof(val))
  ) {
    neoBrightness = (uint8_t)constrain(
      atoi(val),
      0,
      255
    );
  }

  neoDirty = true;

  return httpd_resp_sendstr(req, "OK");
}

// =====================================================
// CAPTURA JPEG
// =====================================================

esp_err_t captureHandler(httpd_req_t* req) {
  if (
    cameraMutex &&
    xSemaphoreTake(
      cameraMutex,
      pdMS_TO_TICKS(1500)
    ) != pdTRUE
  ) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  camera_fb_t* fb = esp_camera_fb_get();

  if (cameraMutex) {
    xSemaphoreGive(cameraMutex);
  }

  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  if (fb->format != PIXFORMAT_JPEG) {
    esp_camera_fb_return(fb);
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");

  httpd_resp_set_hdr(
    req,
    "Cache-Control",
    "no-store"
  );

  httpd_resp_set_hdr(
    req,
    "Content-Disposition",
    "inline; filename=capture.jpg"
  );

  esp_err_t result = httpd_resp_send(
    req,
    (const char*)fb->buf,
    fb->len
  );

  esp_camera_fb_return(fb);
  return result;
}

// =====================================================
// STREAM MJPEG
// =====================================================

#define PART_BOUNDARY "123456789000000000000987654321"

static const char* STREAM_CONTENT_TYPE =
  "multipart/x-mixed-replace;boundary="
  PART_BOUNDARY;

static const char* STREAM_BOUNDARY =
  "\r\n--" PART_BOUNDARY "\r\n";

static const char* STREAM_PART =
  "Content-Type: image/jpeg\r\n"
  "Content-Length: %u\r\n\r\n";

esp_err_t streamHandler(httpd_req_t* req) {
  esp_err_t result = httpd_resp_set_type(
    req,
    STREAM_CONTENT_TYPE
  );

  if (result != ESP_OK) {
    return result;
  }

  httpd_resp_set_hdr(
    req,
    "Access-Control-Allow-Origin",
    "*"
  );

  httpd_resp_set_hdr(
    req,
    "Cache-Control",
    "no-store"
  );

  streamClients++;
  uint8_t consecutiveFailures = 0;

  while (true) {
    camera_fb_t* fb = NULL;

    if (
      !cameraMutex ||
      xSemaphoreTake(
        cameraMutex,
        pdMS_TO_TICKS(1500)
      ) == pdTRUE
    ) {
      fb = esp_camera_fb_get();

      if (cameraMutex) {
        xSemaphoreGive(cameraMutex);
      }
    }

    if (!fb) {
      consecutiveFailures++;

      if (consecutiveFailures >= 20) {
        Serial.println(
          "Stream: camera indisponivel"
        );
        result = ESP_FAIL;
        break;
      }

      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    if (fb->format != PIXFORMAT_JPEG) {
      esp_camera_fb_return(fb);
      consecutiveFailures++;

      if (consecutiveFailures >= 20) {
        result = ESP_FAIL;
        break;
      }

      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    consecutiveFailures = 0;

    lastJpegBytes = (uint32_t)fb->len;
    lastWidth = fb->width;
    lastHeight = fb->height;

    char header[64];

    int headerLength = snprintf(
      header,
      sizeof(header),
      STREAM_PART,
      (unsigned int)fb->len
    );

    result = httpd_resp_send_chunk(
      req,
      STREAM_BOUNDARY,
      strlen(STREAM_BOUNDARY)
    );

    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
        req,
        header,
        headerLength
      );
    }

    if (result == ESP_OK) {
      result = httpd_resp_send_chunk(
        req,
        (const char*)fb->buf,
        fb->len
      );
    }

    esp_camera_fb_return(fb);

    if (result != ESP_OK) {
      break;
    }

    streamFrameTotal++;
    vTaskDelay(1);
  }

  if (streamClients > 0) {
    streamClients--;
  }

  return result;
}

// =====================================================
// SERVIDORES HTTP
// =====================================================

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();

  config.max_uri_handlers = 10;

  httpd_uri_t indexUri = {};
  indexUri.uri = "/";
  indexUri.method = HTTP_GET;
  indexUri.handler = indexHandler;

  httpd_uri_t statusUri = {};
  statusUri.uri = "/status";
  statusUri.method = HTTP_GET;
  statusUri.handler = statusHandler;

  httpd_uri_t controlUri = {};
  controlUri.uri = "/control";
  controlUri.method = HTTP_GET;
  controlUri.handler = controlHandler;

  httpd_uri_t neoUri = {};
  neoUri.uri = "/neo";
  neoUri.method = HTTP_GET;
  neoUri.handler = neoHandler;

  httpd_uri_t captureUri = {};
  captureUri.uri = "/capture";
  captureUri.method = HTTP_GET;
  captureUri.handler = captureHandler;

  httpd_uri_t tftUri = {};
  tftUri.uri = "/tft";
  tftUri.method = HTTP_POST;
  tftUri.handler = tftHandler;

  if (
    httpd_start(
      &camera_httpd,
      &config
    ) == ESP_OK
  ) {
    httpd_register_uri_handler(
      camera_httpd, &indexUri
    );

    httpd_register_uri_handler(
      camera_httpd, &statusUri
    );

    httpd_register_uri_handler(
      camera_httpd, &controlUri
    );

    httpd_register_uri_handler(
      camera_httpd, &neoUri
    );

    httpd_register_uri_handler(
      camera_httpd, &captureUri
    );

    httpd_register_uri_handler(
      camera_httpd, &tftUri
    );

    Serial.println(
      "HTTP: porta 80 | POST /tft ativo"
    );
  }
  else {
    Serial.println(
      "Erro iniciando servidor HTTP"
    );
  }

  // Servidor separado para o stream.
  config.server_port += 1;
  config.ctrl_port += 1;

  httpd_uri_t streamUri = {};
  streamUri.uri = "/stream";
  streamUri.method = HTTP_GET;
  streamUri.handler = streamHandler;

  if (
    httpd_start(
      &stream_httpd,
      &config
    ) == ESP_OK
  ) {
    httpd_register_uri_handler(
      stream_httpd,
      &streamUri
    );

    Serial.println("MJPEG: porta 81");
  }
  else {
    Serial.println(
      "Erro iniciando servidor MJPEG"
    );
  }
}

// =====================================================
// INICIALIZACAO DA CAMERA
// =====================================================

bool initCamera() {
  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.jpeg_quality = 10;

  if (psramFound()) {
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  }
  else {
    Serial.println(
      "PSRAM indisponivel: usando VGA"
    );

    config.frame_size = FRAMESIZE_VGA;
    config.fb_count = 1;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf(
      "Erro na camera: 0x%x\n",
      err
    );
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();

  if (!sensor) {
    return false;
  }

  sensor->set_framesize(sensor, FRAMESIZE_VGA);
  sensor->set_quality(sensor, 10);
  sensor->set_brightness(sensor, 0);
  sensor->set_contrast(sensor, 0);
  sensor->set_saturation(sensor, 0);

  return true;
}

// =====================================================
// WIFI
// =====================================================

bool initWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  Serial.print("Conectando ao Wi-Fi");

  unsigned long inicio = millis();

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - inicio < 20000
  ) {
    delay(300);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    return true;
  }

  Serial.println(
    "Wi-Fi indisponivel. Iniciando AP."
  );

  WiFi.disconnect(true);
  delay(200);

  WiFi.mode(WIFI_AP);

  if (
    !WiFi.softAP(
      "ESP32-Camera",
      "esp32cam"
    )
  ) {
    return false;
  }

  Serial.print("IP do AP: ");
  Serial.println(WiFi.softAPIP());

  return true;
}

// =====================================================
// ILUMINACAO
// =====================================================

void updateNeoPixel() {
  if (neoDirty) {
    pixels.setBrightness(neoBrightness);

    if (neoMode == NEO_OFF) {
      pixels.clear();
      pixels.show();
    }
    else if (
      neoMode == NEO_SOLID ||
      neoMode == NEO_DUAL
    ) {
      for (int i = 0; i < NUMPIXELS; i++) {
        uint32_t rgb =
          (
            neoMode == NEO_DUAL &&
            (i % 2 == 1)
          )
          ? neoColor2
          : neoColor1;

        pixels.setPixelColor(
          i,
          pixels.Color(
            (uint8_t)((rgb >> 16) & 0xFF),
            (uint8_t)((rgb >> 8) & 0xFF),
            (uint8_t)(rgb & 0xFF)
          )
        );
      }

      pixels.show();
    }

    neoDirty = false;
  }

  if (neoMode != NEO_CHASE) {
    return;
  }

  unsigned long agora = millis();

  if (
    agora - lastNeo < NEO_INTERVAL
  ) {
    return;
  }

  lastNeo = agora;

  pixels.clear();

  uint32_t rgb = neoColor1;

  pixels.setPixelColor(
    neoPosition % NUMPIXELS,
    pixels.Color(
      (uint8_t)((rgb >> 16) & 0xFF),
      (uint8_t)((rgb >> 8) & 0xFF),
      (uint8_t)(rgb & 0xFF)
    )
  );

  pixels.show();
  neoPosition++;
}

// =====================================================
// ESTATISTICAS
// =====================================================

void updateStats() {
  unsigned long agora = millis();

  if (agora - statsTime < 2000) {
    return;
  }

  float elapsed =
    (agora - statsTime) / 1000.0f;

  uint32_t current = streamFrameTotal;

  streamFPS =
    (current - previousStreamFrames) / elapsed;

  previousStreamFrames = current;
  statsTime = agora;

  Serial.printf(
    "Stream %.1f FPS | "
    "%ux%u | "
    "JPEG %u bytes | "
    "clientes %u | "
    "Heap %u | "
    "PSRAM %u | "
    "TFT %s\n",
    streamFPS,
    (unsigned int)lastWidth,
    (unsigned int)lastHeight,
    (unsigned int)lastJpegBytes,
    (unsigned int)streamClients,
    ESP.getFreeHeap(),
    ESP.getFreePsram(),
    nomeEstadoTFT(estadoTFTRecebido)
  );
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println(
    "INSPECTA - SISTEMA DE INSPECAO VISUAL"
  );

  // Mantem a iluminacao original.
  pixels.begin();
  pixels.setBrightness(neoBrightness);
  pixels.clear();
  pixels.show();
  updateNeoPixel();

  // O TFT inicia cinza antes mesmo da conexao Wi-Fi.
  iniciarTFT();

  Serial.printf(
    "PSRAM: %s\n",
    psramFound()
      ? "disponivel"
      : "indisponivel"
  );

  cameraMutex = xSemaphoreCreateMutex();

  if (!cameraMutex) {
    Serial.println(
      "Aviso: mutex da camera indisponivel"
    );
  }

  cameraOK = initCamera();

  if (!cameraOK) {
    Serial.println(
      "Nao foi possivel iniciar a camera"
    );
    return;
  }

  if (initWiFi()) {
    startCameraServer();

    IPAddress ip =
      WiFi.getMode() == WIFI_MODE_AP
      ? WiFi.softAPIP()
      : WiFi.localIP();

    Serial.print("Interface: http://");
    Serial.println(ip);

    Serial.print("Video: http://");
    Serial.print(ip);
    Serial.println(":81/stream");

    Serial.print("TFT: http://");
    Serial.print(ip);
    Serial.println("/tft");
  }

  statsTime = millis();
  lastNeo = millis();
}

// =====================================================
// LOOP
// =====================================================

void loop() {
  updateNeoPixel();
  atualizarTFT();

  if (!cameraOK) {
    delay(50);
    return;
  }

  updateStats();
  delay(1);
}
