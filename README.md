# INSPECTA — Inspeção visual para controle de qualidade industrial

Projeto Final de Engenharia em Engenharia de Computação — Centro Universitário Senac.

O **INSPECTA** é um protótipo de inspeção visual automatizada de peças industriais. Uma câmera conectada a uma **ESP32** captura as imagens, enquanto uma aplicação Python executada em um computador utiliza uma rede neural **EfficientNet-B0** para classificar as faces da peça como **conforme** ou **não conforme**. O resultado é apresentado em um painel web, com controle da iluminação, ajustes da câmera e registro das inspeções.

> **Escopo:** protótipo acadêmico para aquisição, classificação e análise experimental. Os resultados não constituem certificação para uso autônomo em uma linha industrial.

## Funcionalidades

- Captura de vídeo por streaming MJPEG com **ESP32-Camera / OV2640**.
- Iluminação controlada por anel de **24 LEDs WS2812B (NeoPixel)**.
- Interface web para visualizar a câmera, configurar a aquisição e acompanhar a inspeção.
- Região de interesse (**ROI**) de 320 × 320 pixels, com posicionamento automático ou manual.
- Classificação independente da **frente** e do **verso**, utilizando modelos EfficientNet-B0 treinados com PyTorch.
- Agregação de quatro leituras reais por face durante a inspeção no painel.
- Seleção de checkpoints `treino_N.pt` sem modificar o firmware.
- Histórico de inspeções e exportação dos resultados em CSV.
- Coleta de dataset rotulado por face, treinamento e relatórios comparativos de experimentos.

## Arquitetura

```text
Peça industrial
      │
      ▼
ESP32 + câmera OV2640 + iluminação WS2812B
      │  Streaming MJPEG / rede Wi-Fi
      ▼
Computador com Python
      ├── OpenCV: captura e ROI
      ├── PyTorch: EfficientNet-B0
      ├── FastAPI: API e servidor web
      └── CSV: histórico das inspeções
      │
      ▼
Painel INSPECTA no navegador
```

A **ESP32 é responsável pela câmera, iluminação e comunicação**. A inferência da rede neural é executada **no computador**, não no microcontrolador. O sistema pode funcionar em uma rede local, sem depender de serviços de inferência em nuvem. Para a modalidade sem roteador, o firmware pode criar um ponto de acesso Wi-Fi caso não consiga se conectar à rede configurada.

## Estrutura do repositório

```text
.
├── ESP/
│   ├── firmware_esp32.ino
│   └── painel_tcc_esp.h
└── IA e INTERFACE/
    ├── coletar_dataset.py
    ├── inspecao_tempo_real.py
    ├── treinar.py
    ├── requirements.txt
    ├── roi.json
    ├── painel_tcc/
    │   ├── painel_tcc.py
    │   ├── requirements-painel.txt
    │   └── web/
    │       └── index.html
    ├── dataset/             # Gerado pela coleta
    ├── modelos/             # Checkpoints .pt gerados pelo treinamento
    └── relatorios/          # Métricas, gráficos e históricos de treino
```

O pacote original também contém versões anteriores de treinamento, uma cópia do firmware em `IA e INTERFACE/firmware_esp32/` e um `dataset_antigo/`. Eles são mantidos como arquivos auxiliares/históricos; o fluxo principal utiliza os caminhos acima. Algumas pastas são criadas durante a execução e podem não estar presentes em um clone novo.

## Requisitos

### Hardware

- ESP32 com câmera compatível com o firmware e sensor **OV2640** (configuração de pinos do projeto SunFounder).
- Anel de **24 LEDs WS2812B** e alimentação adequada.
- Computador ou notebook para executar a IA e o servidor web.
- Rede Wi-Fi compartilhada entre computador e ESP32, ou conexão ao ponto de acesso criado pela ESP32.

### Software

- **Python 3.10 ou superior**, em versão compatível com o PyTorch escolhido.
- Bibliotecas Python listadas em `requirements.txt` e `painel_tcc/requirements-painel.txt`.
- **Arduino IDE** com suporte a placas ESP32 e biblioteca **Adafruit NeoPixel**.
- Navegador atualizado.

Os principais componentes Python são **PyTorch, torchvision, OpenCV, NumPy, Pillow, Matplotlib, FastAPI e Uvicorn**.

## 1. Preparar o ambiente Python

No Windows, abra o PowerShell na raiz do repositório:

```powershell
cd "IA e INTERFACE"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r painel_tcc/requirements-painel.txt
```

> O arquivo `requirements.txt` foi exportado do ambiente de desenvolvimento e fixa versões de diversas dependências. Dependendo da versão do Python e da plataforma, pode ser necessário instalar uma combinação compatível de `torch`/`torchvision`, conforme as instruções oficiais do PyTorch. A instalação inicial exige internet, salvo quando os pacotes já estão disponíveis localmente.

## 2. Configurar e gravar a ESP32

1. Abra `ESP/firmware_esp32.ino` na Arduino IDE, mantendo **`painel_tcc_esp.h` na mesma pasta**.
2. Confira a pinagem da sua placa e as configurações de Wi-Fi presentes no firmware.
3. Instale a biblioteca **Adafruit NeoPixel** e selecione a placa ESP32 correspondente. A configuração documentada no projeto é **ESP32 Dev Module**, com **PSRAM habilitada**.
4. Compile e grave o firmware.
5. Abra o **Monitor Serial** e verifique o endereço IP informado após a inicialização.

O firmware tenta se conectar à rede Wi-Fi configurada. Se essa conexão falhar, ele tenta iniciar um ponto de acesso local. Utilize **o IP indicado pelo Monitor Serial**, pois ele pode variar conforme a rede utilizada.

Endpoints úteis, substituindo `<IP_ESP32>` pelo endereço real:

| Endereço | Finalidade |
|---|---|
| `http://<IP_ESP32>/` | Página disponibilizada pela ESP32 |
| `http://<IP_ESP32>/status` | Estado do dispositivo |
| `http://<IP_ESP32>/capture` | Imagem JPEG |
| `http://<IP_ESP32>:81/stream` | Vídeo MJPEG |

**Antes de publicar o repositório:** o firmware contém valores de configuração de Wi-Fi no código-fonte. Remova ou substitua eventuais credenciais particulares, revise também o acesso ao ponto de rede criado pela ESP32 e altere senhas de demonstração antes de qualquer uso externo.

## 3. Configurar a conexão do computador com a ESP32

O painel lê o endereço da câmera pela variável de ambiente `TCC_ESP_IP` ou, na ausência dela, pelo valor padrão definido no arquivo `painel_tcc.py`.

No **mesmo terminal** em que o painel será iniciado:

```powershell
$env:TCC_ESP_IP="192.168.0.105"
```

Substitua `192.168.0.105` pelo IP exibido no Monitor Serial. Se estiver conectado diretamente ao ponto de acesso da ESP32, use o IP desse modo de operação.

O script de coleta (`coletar_dataset.py`) e a inspeção alternativa (`inspecao_tempo_real.py`) possuem uma variável própria chamada `STREAM_URL`, que também deve apontar para:

```python
STREAM_URL = "http://<IP_ESP32>:81/stream"
```

O trecho acima é **ilustrativo**: substitua `<IP_ESP32>` pelo endereço válido antes de executar o código.

## 4. Executar o painel INSPECTA

Com o ambiente Python ativado, dentro de `IA e INTERFACE`:

```powershell
python painel_tcc/painel_tcc.py
```

Abra no navegador:

**http://localhost:8765**

Em outro dispositivo conectado à mesma rede local, é possível acessar `http://<IP_DO_COMPUTADOR>:8765`, desde que a rede e o firewall permitam a conexão. A página servida pela própria ESP32 também pode utilizar a API do computador, informando seu endereço de rede na interface.

### Fluxo de inspeção

1. Verifique a transmissão da câmera e ajuste, se necessário, a iluminação e a ROI.
2. Selecione um checkpoint disponível em `modelos/`.
3. Posicione a **frente** da peça e inicie sua análise.
4. Vire a peça, posicione o **verso** e execute a segunda análise.
5. Consulte o diagnóstico final e o histórico no painel.

O painel utiliza **quatro leituras de IA por face**. A probabilidade de não conformidade da peça é obtida a partir da maior das probabilidades agregadas da frente e do verso. Assim, uma peça é rejeitada quando pelo menos uma das faces ultrapassa o limiar de classificação configurado no modelo.

O histórico do painel é registrado em `IA e INTERFACE/painel_tcc/resultados_tcc.csv`.

### Diagnóstico de conexão

Caso o vídeo não apareça, confira nesta ordem:

1. O IP da ESP32 no Monitor Serial.
2. O acesso a `http://<IP_ESP32>/status` e `http://<IP_ESP32>:81/stream`.
3. A variável `TCC_ESP_IP` usada pelo painel.
4. O acesso a `http://localhost:8765/api/status` no computador.
5. Se o firewall permite a porta **8765** somente na rede local confiável.

Evite abrir várias conexões MJPEG simultâneas com a câmera. **Não exponha a API do painel diretamente à internet**, pois ela inclui comandos capazes de alterar a configuração do hardware.

## 5. Coletar imagens para o dataset

Dentro de `IA e INTERFACE`:

```powershell
python coletar_dataset.py
```

O aplicativo apresenta a imagem da câmera e permite definir os rótulos de frente e verso separadamente:

| Tecla | Ação |
|---|---|
| `C` | Face conforme |
| `N` | Face não conforme |
| `U` | Face não avaliada (encaminha a peça para revisão) |
| `R` | Centraliza a ROI |
| `Q` ou `Esc` | Encerra a coleta |

Depois de classificar a frente, vire a peça e informe o rótulo do verso. A ROI também pode ser ajustada com o mouse.

São geradas **quatro imagens de cada face**: original, inversão vertical, espelhamento horizontal e ajuste suave de luminosidade. As imagens recebem nomes `frente_01.png` a `frente_04.png` e `verso_01.png` a `verso_04.png`, acompanhadas pelo arquivo `coleta.json` com os rótulos de cada face.

A classe da peça é **não conforme** quando ao menos uma das faces recebe esse rótulo. Uma peça só é **conforme** se as duas faces forem conformes.

> As versões transformadas ajudam no treinamento e na análise de robustez, mas não representam novas peças físicas independentes.

## 6. Treinar a rede neural

Organize o conjunto de dados dentro de `IA e INTERFACE/dataset/`:

```text
dataset/
├── conforme/
│   └── peca_.../
│       ├── coleta.json
│       ├── frente_01.png ... frente_04.png
│       └── verso_01.png  ... verso_04.png
└── nao_conforme/
    └── peca_.../
        ├── coleta.json
        ├── frente_01.png ... frente_04.png
        └── verso_01.png  ... verso_04.png
```

Execute:

```powershell
python treinar.py
```

O script utiliza **EfficientNet-B0** com transferência de aprendizado a partir de pesos pré-treinados, aprende com as quatro imagens disponíveis por face e mantém a **divisão por peça física** para evitar que fotografias da mesma peça apareçam em conjuntos diferentes.

A configuração atual trabalha com imagens de entrada de **320 × 320 pixels** e utiliza, quando a divisão existente é preservada, aproximadamente **70% das peças para treino, 15% para validação e 15% para teste**.

A validação e o teste são apresentados em duas estratégias complementares:

- **Foto original (`01`)**: utiliza uma fotografia de cada face. A perda dessa avaliação na validação determina qual época gera o checkpoint salvo.
- **Média de quatro fotografias**: calcula a média das probabilidades de cada face e fornece uma avaliação adicional de consistência. **Não treina outro modelo e não seleciona o checkpoint.**

O teste é executado somente depois de restaurar o melhor checkpoint encontrado na validação.

Cada execução gera arquivos numerados, sem sobrescrever os anteriores:

```text
modelos/treino_N.pt
relatorios/treino_N/
    parametros.json
    historico.csv
    resumo.json
    metricas_validacao.json
    metricas_teste.json
    curvas.png
    matrizes_confusao.png
    ...
```

O diretório de relatórios contém ainda previsões por imagem, por face e por peça, informações sobre as sessões de coleta e comparações entre estratégias. Esses resultados apoiam a análise de **acurácia, precisão, recall, F1-score, falsos positivos e falsos negativos**.

Para preservar a comparabilidade dos experimentos, registre sempre a resolução, os hiperparâmetros, a assinatura do dataset e a divisão das peças. **Não utilize resultados repetidamente observados no teste para ajustar o modelo e depois relatar esse mesmo teste como independente.**

## 7. Inspeção alternativa pelo OpenCV

O projeto também inclui uma interface de inspeção fora do navegador:

```powershell
python inspecao_tempo_real.py --listar-modelos
python inspecao_tempo_real.py --modelo mais_recente
```

Para selecionar um checkpoint específico:

```powershell
python inspecao_tempo_real.py --modelo treino_12.pt
```

Esse programa utiliza a stream da ESP32 e oferece comandos de inspeção e alternância de modelo na janela OpenCV. Consulte as instruções apresentadas no terminal ao iniciar.

## 8. Versionamento e reprodução

Recomenda-se manter no repositório os códigos-fonte, os arquivos de dependências, a documentação e os relatórios necessários ao TCC. A pasta `.venv/` não deve ser versionada.

Exemplo mínimo de `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
.pytest_cache/
```

Datasets e checkpoints podem ocupar bastante espaço. Se forem distribuídos no GitHub, considere o **Git LFS** ou disponibilize-os separadamente, com identificação da versão utilizada em cada experimento. Nunca publique senhas pessoais ou arquivos contendo credenciais.

## Limitações e validação

O resultado de um treinamento não garante o mesmo desempenho em todas as condições reais. Reflexos, luminosidade, posicionamento da peça, resolução da câmera e diversidade do dataset influenciam a classificação. A avaliação final deve utilizar **peças físicas novas**, rótulos conferidos e medição do tempo total de inspeção, incluindo a aquisição e o processamento das duas faces.

O INSPECTA foi desenvolvido como projeto experimental de **visão computacional aplicada ao controle de qualidade industrial**, com foco na integração entre hardware, aquisição de imagens, aprendizado de máquina e análise quantitativa de resultados.
