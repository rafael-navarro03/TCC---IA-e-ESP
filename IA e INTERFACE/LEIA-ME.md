# INSPECTA V2 — página ESP32 e painel Python

## O que mudou
- A tela principal tem a **câmera à esquerda** e o **diagnóstico da IA à direita** em monitores grandes.
- Abaixo, as configurações do modelo, ROI, câmera OV2640, NeoPixel e conexão com o PC.
- Histórico das inspeções na parte inferior.
- Em telas menores, os painéis são empilhados automaticamente.

## Firmware ESP32 (usar **dois arquivos**)
Abra `firmware_esp32/firmware_esp32.ino` no Arduino IDE. O arquivo `painel_tcc_esp.h` deve permanecer **na mesma pasta** do `.ino`. Os dois são necessários.

O firmware foi gerado a partir do **seu `teste1.ino` funcionando**, preservando a pinagem, SSID/configuração de rede, endpoints `/status`, `/control`, `/neo`, `/capture`, `:81/stream` e todas as rotinas de inicialização e iluminação. A única mudança no código-fonte funcional é substituir o HTML antigo pelo novo painel.

No Arduino IDE selecione a mesma placa e opções que usa atualmente (ESP32 Dev Module, PSRAM habilitada). Compile e envie. Faça backup do firmware anterior.

**Observação:** o sketch preserva as credenciais locais de Wi-Fi do arquivo fornecido. Não publique esse sketch em repositórios públicos.

## PC: IA e streaming
Extraia a pasta `painel_tcc` dentro de sua pasta `IA`, **substituindo** o `painel_tcc/web/index.html` anterior. Os checkpoints permanecem em `IA/modelos` (nenhum modelo é incluído no ZIP). O servidor Python ainda é necessário, pois a EfficientNet-B0 não roda na ESP32.

Na pasta `IA/painel_tcc` execute:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements-painel.txt
..\.venv\Scripts\python.exe painel_tcc.py
```

Abra `http://localhost:8765` para testar. Para usar o **site da própria ESP32**, abra `http://IP_DA_ESP32/` e informe no painel o endereço do PC, **por exemplo** `http://192.168.0.100:8765`; use o IP real da sua máquina. PC e ESP32 devem estar na mesma rede. Libere a porta TCP 8765 no firewall **somente na rede privada**.

## Diagnóstico quando aparece “Conectando à câmera” ou “Sem vídeo”
Isso **não é causado pelo layout HTML**. Primeiro verifique se a ESP32 abre `http://IP_DA_ESP32/status` e se o stream `http://IP_DA_ESP32:81/stream` funciona. Depois verifique o acesso a `http://localhost:8765/api/status` no PC e a `http://localhost:8765/api/stream`. Evite abrir a stream diretamente em outra aba enquanto o servidor usa a câmera.

A página hospedada na ESP32 depende da API do PC. Se abrir o painel em outro dispositivo, coloque o **IP LAN do PC**, não `localhost`.

O controle do NeoPixel passa pelo PC, que faz a solicitação à ESP32. A centralização da ROI também é calculada no PC por OpenCV.

## Limitações
- Sem testes na sua placa física nesta execução; verifique a compilação no Arduino IDE e transmissão real.
- A centralização automática exige iluminação consistente e peça reconhecível sobre bancada preta. Use modo manual se falhar.
- Alterar resolução, brilho e iluminação muda a distribuição das imagens; isso pode afetar a acurácia do modelo.
- A ferramenta é protótipo acadêmico, não equipamento de segurança industrial.
