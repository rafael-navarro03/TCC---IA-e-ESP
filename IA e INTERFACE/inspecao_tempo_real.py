
"""
Inspecao experimental em tempo real: ESP32 + EfficientNet-B0.

- Nao altera ESP32 ou iluminacao: le somente MJPEG.
- Escolhe automaticamente o treino_N mais recente, ou usa --modelo.
- M/B: proximo/anterior checkpoint; U: atualizar lista.
- I: 4 leituras da frente, ESPACO: 4 do verso.
- L: modo continuo; R: limpar; Q/ESC: sair.
- Registra resultados por modelo em resultados_inspecao_modelos.csv.
"""

import argparse
import csv
import json
import os
import pickle
import re
import threading
import time

from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent

PASTA_MODELOS = BASE / "modelos"

STREAM_URL = "http://192.168.0.105:81/stream"

ROI_FILE = BASE / "roi.json"

RELATORIO = BASE / "resultados_inspecao_modelos.csv"

# Escolha inicial:
# "mais_recente", "treino_4.pt", etc.
MODELO_INICIAL = "treino_9.pt"

ROI_LADO = 320

INTERVALO_IA = 0.25

JANELA_TEMPORAL = 4

FRAMES_POR_FACE = 4

PREVIA_MAX_W = 1050
PREVIA_MAX_H = 690

ALTURA_PAINEL = 204

JANELA = "Inspecao visual - selecao de modelo"


# ============================================================
# DESCOBERTA E CARREGAMENTO DOS MODELOS
# ============================================================

def ordenar_modelos(caminho):

    match = re.fullmatch(
        r"treino_(\d+)\.pt",
        caminho.name,
        re.IGNORECASE,
    )

    if match:

        return (
            1,
            int(match.group(1)),
            caminho.name.lower(),
        )

    return (
        0,
        0,
        caminho.name.lower(),
    )


def listar_modelos():

    if not PASTA_MODELOS.is_dir():
        return []

    return sorted(
        PASTA_MODELOS.glob("*.pt"),
        key=ordenar_modelos,
    )


def carregar_checkpoint(caminho):

    try:

        return torch.load(
            caminho,
            map_location="cpu",
            weights_only=True,
        )

    except pickle.UnpicklingError as erro:

        # Compatibilidade restrita com checkpoints
        # antigos deste projeto.
        #
        # Utilize somente arquivos de origem confiavel.

        if (
            "torch.torch_version.TorchVersion"
            not in str(erro)
        ):

            raise

        from torch.torch_version import TorchVersion

        with torch.serialization.safe_globals(
            [TorchVersion]
        ):

            return torch.load(
                caminho,
                map_location="cpu",
                weights_only=True,
            )


# ============================================================
# CLASSIFICADOR
# ============================================================

class Classificador:

    def __init__(self):

        if torch.cuda.is_available():

            self.device = torch.device("cuda")

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):

            self.device = torch.device("mps")

        else:

            self.device = torch.device("cpu")

            torch.set_num_threads(
                max(
                    1,
                    min(
                        os.cpu_count() or 4,
                        4,
                    ),
                )
            )

        self.modelo = None
        self.preprocess = None

        self.caminho = None

        self.limiar = 0.5
        self.melhor_epoca = None
        self.modo_treinamento = ""

    # ========================================================
    # CARREGAR UM MODELO
    # ========================================================

    def carregar(self, caminho):

        if not caminho.is_file():

            raise FileNotFoundError(
                f"Checkpoint nao encontrado: {caminho}"
            )

        checkpoint = carregar_checkpoint(
            caminho
        )

        # ----------------------------------------------------
        # VERIFICACAO DAS CLASSES
        # ----------------------------------------------------

        if checkpoint.get("class_names") != [
            "conforme",
            "nao_conforme",
        ]:

            raise ValueError(
                "Ordem das classes incompativel "
                "com a inspecao."
            )

        # ----------------------------------------------------
        # VERIFICACAO DA ARQUITETURA
        # ----------------------------------------------------

        if checkpoint.get(
            "architecture",
            "efficientnet_b0",
        ) != "efficientnet_b0":

            raise ValueError(
                "Esta inspecao aceita somente "
                "EfficientNet-B0 convencional."
            )

        interpolacao = checkpoint.get(
            "resize_interpolation",
            "bicubic",
        )

        if interpolacao != "bicubic":

            raise ValueError(
                "Interpolacao diferente daquela "
                "utilizada no treinamento."
            )

        if not isinstance(
            checkpoint.get("model_state_dict"),
            dict,
        ):

            raise ValueError(
                "Checkpoint sem model_state_dict."
            )

        # ----------------------------------------------------
        # RECONSTRUIR A EFFICIENTNET
        # ----------------------------------------------------

        rede = efficientnet_b0(
            weights=None
        )

        entradas = (
            rede.classifier[1].in_features
        )

        rede.classifier[1] = nn.Linear(
            entradas,
            2,
        )

        try:

            rede.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

        except RuntimeError as erro:

            raise ValueError(
                "Pesos incompativeis. Modelos "
                "multivisao com duas cabecas "
                "precisam de outro carregador."
            ) from erro

        rede = rede.to(
            self.device
        ).eval()

        # ----------------------------------------------------
        # PRE-PROCESSAMENTO DO CHECKPOINT
        # ----------------------------------------------------

        tamanho = int(
            checkpoint.get(
                "image_size",
                224,
            )
        )

        media = checkpoint.get(
            "mean",
            [0.485, 0.456, 0.406],
        )

        desvio = checkpoint.get(
            "std",
            [0.229, 0.224, 0.225],
        )

        preprocess = transforms.Compose([

            transforms.Resize(
                (tamanho, tamanho),
                interpolation=(
                    transforms.InterpolationMode.BICUBIC
                ),
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=media,
                std=desvio,
            ),
        ])

        limiar = float(
            checkpoint.get(
                "threshold",
                0.5,
            )
        )

        if not 0 < limiar < 1:

            raise ValueError(
                "Limiar fora do intervalo (0, 1)."
            )

        # ----------------------------------------------------
        # ATIVAR MODELO SOMENTE APOS VALIDAR TUDO
        # ----------------------------------------------------

        self.modelo = rede

        self.preprocess = preprocess

        self.caminho = caminho

        self.limiar = limiar

        self.melhor_epoca = checkpoint.get(
            "best_epoch",
            "?",
        )

        self.modo_treinamento = str(
            checkpoint.get(
                "training_mode",
                "nao_informado",
            )
        )

        print(
            f"[MODELO] {caminho.name} "
            f"| epoca={self.melhor_epoca} "
            f"| limiar={limiar:.3f} "
            f"| dispositivo={self.device}"
        )

    # ========================================================
    # INFERENCIA
    # ========================================================

    @torch.inference_mode()
    def prever(self, roi_bgr):

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            roi_bgr,
            cv2.COLOR_BGR2RGB,
        )

        imagem = Image.fromarray(
            rgb
        )

        entrada = self.preprocess(
            imagem
        )

        entrada = entrada.unsqueeze(
            0
        ).to(self.device)

        # ----------------------------------------------------
        # TEMPO DE INFERENCIA
        # ----------------------------------------------------

        if self.device.type == "cuda":

            torch.cuda.synchronize()

        inicio = time.perf_counter()

        logits = self.modelo(
            entrada
        )

        probabilidades = (
            logits.softmax(dim=1)[0]
        )

        prob_nc = float(
            probabilidades[1].item()
        )

        if self.device.type == "cuda":

            torch.cuda.synchronize()

        duracao_ms = (
            time.perf_counter() - inicio
        ) * 1000

        return (
            prob_nc,
            duracao_ms,
        )


# ============================================================
# CAMERA MJPEG EM THREAD
# ============================================================

class CameraMJPEG:

    def __init__(self, url):

        self.url = url

        self.lock = threading.Lock()

        self.parar = threading.Event()

        self.frame = None

        self.sequencia = 0

        self.ultimo_frame = 0.0

        self.fps = 0.0

        self.conectada = False

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )

    def iniciar(self):

        self.thread.start()

    def _loop(self):

        while not self.parar.is_set():

            cap = cv2.VideoCapture(
                self.url
            )

            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )

            if not cap.isOpened():

                with self.lock:
                    self.conectada = False

                cap.release()

                self.parar.wait(1.5)

                continue

            contador = 0

            inicio = time.monotonic()

            with self.lock:
                self.conectada = True

            while not self.parar.is_set():

                ok, frame = cap.read()

                if not ok or frame is None:
                    break

                agora = time.monotonic()

                contador += 1

                with self.lock:

                    self.frame = frame

                    self.sequencia += 1

                    self.ultimo_frame = agora

                if agora - inicio >= 1.5:

                    with self.lock:

                        self.fps = (
                            contador / (agora - inicio)
                        )

                    contador = 0
                    inicio = agora

            cap.release()

            with self.lock:

                self.conectada = False

                self.fps = 0.0

            self.parar.wait(1.0)

    def obter(self, ultima_sequencia):

        with self.lock:

            # Copia o frame apenas quando ele mudou.

            frame = (
                self.frame.copy()

                if (
                    self.frame is not None
                    and self.sequencia != ultima_sequencia
                )

                else None
            )

            return (
                frame,
                self.sequencia,
                self.fps,
                self.conectada,
                self.ultimo_frame,
            )

    def fechar(self):

        self.parar.set()

        if self.thread.is_alive():

            self.thread.join(
                timeout=2.0
            )


# ============================================================
# INTERFACE DE INSPECAO
# ============================================================

class Interface:

    def __init__(
        self,
        largura,
        altura,
        classificador,
    ):

        self.w = largura
        self.h = altura

        if (
            largura < ROI_LADO
            or altura < ROI_LADO
        ):

            raise ValueError(
                "Stream menor que a ROI de 320x320."
            )

        escala = min(
            1.0,
            PREVIA_MAX_W / largura,
            PREVIA_MAX_H / altura,
        )

        self.pw = max(
            1,
            round(largura * escala),
        )

        self.ph = max(
            1,
            round(altura * escala),
        )

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        self.x = (
            largura - ROI_LADO
        ) // 2

        self.y = (
            altura - ROI_LADO
        ) // 2

        self.carregar_roi()

        self.arrastando = False

        self.dx = 0
        self.dy = 0

        # ----------------------------------------------------
        # MODELO
        # ----------------------------------------------------

        self.arquivo_modelo = ""

        self.epoca_modelo = "?"

        self.limiar = 0.5

        self.indice_modelo = 0
        self.total_modelos = 0

        self.atualizar_modelo(
            classificador,
            0,
            1,
        )

    # ========================================================
    # ATUALIZAR MODELO ATIVO
    # ========================================================

    def atualizar_modelo(
        self,
        classificador,
        indice,
        total,
    ):

        self.arquivo_modelo = (
            classificador.caminho.name
        )

        self.epoca_modelo = (
            classificador.melhor_epoca
        )

        self.limiar = (
            classificador.limiar
        )

        self.indice_modelo = (
            indice + 1
        )

        self.total_modelos = total

        # Ao trocar, apaga somente a analise atual.
        self.reiniciar()

    # ========================================================
    # REINICIAR ANALISE
    # ========================================================

    def reiniciar(self):

        self.estado = "continuo"

        self.historico = deque(
            maxlen=JANELA_TEMPORAL
        )

        self.prob_inst = None

        self.probs_frente = []
        self.probs_verso = []

        self.media_frente = None
        self.media_verso = None

        self.resultado_peca = None

        self.roi_frente = None
        self.roi_verso = None

        self.inicio_peca = None

        self.ultima_inferencia = 0.0

        self.ultima_inferencia_ms = 0.0

        self.contagem_ia = 0

        self.fps_ia = 0.0

        self.tempo_fps = time.monotonic()

    # ========================================================
    # LIMITES DA ROI
    # ========================================================

    def limitar_roi(self, x, y):

        self.x = max(
            0,
            min(
                int(x),
                self.w - ROI_LADO,
            ),
        )

        self.y = max(
            0,
            min(
                int(y),
                self.h - ROI_LADO,
            ),
        )

    # ========================================================
    # CARREGAR ROI ANTERIOR
    # ========================================================

    def carregar_roi(self):

        if not ROI_FILE.is_file():
            return

        try:

            dados = json.loads(
                ROI_FILE.read_text(
                    encoding="utf-8"
                )
            )

            resolucao = (
                dados.get("frame_width"),
                dados.get("frame_height"),
            )

            if resolucao != (
                self.w,
                self.h,
            ):

                print(
                    "[ROI] Resolucao mudou; "
                    "usando centro da imagem."
                )

                return

            cx = (
                int(dados["x"])
                + int(dados["w"]) // 2
            )

            cy = (
                int(dados["y"])
                + int(dados["h"]) // 2
            )

            self.limitar_roi(
                cx - ROI_LADO // 2,
                cy - ROI_LADO // 2,
            )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as erro:

            print(
                "[ROI] Arquivo ignorado:",
                erro,
            )

    # ========================================================
    # SALVAR ROI
    # ========================================================

    def salvar_roi(self):

        dados = {
            "x": self.x,
            "y": self.y,
            "w": ROI_LADO,
            "h": ROI_LADO,
            "frame_width": self.w,
            "frame_height": self.h,
        }

        ROI_FILE.write_text(
            json.dumps(
                dados,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ========================================================
    # CONTROLE DO MOUSE
    # ========================================================

    def mouse(
        self,
        evento,
        px,
        py,
        flags,
        userdata,
    ):

        # Trava a ROI durante as capturas.

        if self.estado in (
            "frente",
            "verso",
        ):
            return

        if (
            evento == cv2.EVENT_LBUTTONDOWN
            and py >= self.ph
        ):
            return

        x = max(
            0,
            min(
                int(
                    px * self.w / self.pw
                ),
                self.w - 1,
            ),
        )

        y = max(
            0,
            min(
                int(
                    py * self.h / self.ph
                ),
                self.h - 1,
            ),
        )

        if evento == cv2.EVENT_LBUTTONDOWN:

            dentro = (
                self.x <= x < self.x + ROI_LADO
                and
                self.y <= y < self.y + ROI_LADO
            )

            self.dx = (
                x - self.x
                if dentro
                else ROI_LADO // 2
            )

            self.dy = (
                y - self.y
                if dentro
                else ROI_LADO // 2
            )

            self.arrastando = True

            self.limitar_roi(
                x - self.dx,
                y - self.dy,
            )

            self.historico.clear()

        elif (
            evento == cv2.EVENT_MOUSEMOVE
            and self.arrastando
        ):

            self.limitar_roi(
                x - self.dx,
                y - self.dy,
            )

        elif (
            evento == cv2.EVENT_LBUTTONUP
            and self.arrastando
        ):

            self.limitar_roi(
                x - self.dx,
                y - self.dy,
            )

            self.arrastando = False

            self.salvar_roi()

            self.historico.clear()

    # ========================================================
    # INICIAR INSPECAO DE UMA PECA
    # ========================================================

    def iniciar_peca(self):

        self.reiniciar()

        self.estado = "frente"

        self.inicio_peca = (
            time.monotonic()
        )

        self.roi_frente = (
            self.x,
            self.y,
        )

        print(
            "[PECA] Coletando frente "
            f"(0/{FRAMES_POR_FACE})."
        )

    # ========================================================
    # INICIAR VERSO
    # ========================================================

    def iniciar_verso(self):

        if self.estado != "virar":
            return

        self.estado = "verso"

        self.roi_verso = (
            self.x,
            self.y,
        )

        self.ultima_inferencia = 0.0

        print(
            "[PECA] Coletando verso "
            f"(0/{FRAMES_POR_FACE})."
        )

    # ========================================================
    # CLASSIFICAR PROBABILIDADE
    # ========================================================

    def classificar(self, prob):

        return (
            "NAO CONFORME"
            if prob >= self.limiar
            else "CONFORME"
        )

    # ========================================================
    # RECEBER PREVISAO
    # ========================================================

    def receber_previsao(
        self,
        prob,
        ms,
    ):

        self.prob_inst = prob

        self.ultima_inferencia_ms = ms

        self.contagem_ia += 1

        self.historico.append(
            prob
        )

        # ----------------------------------------------------
        # FRENTE
        # ----------------------------------------------------

        if self.estado == "frente":

            self.probs_frente.append(
                prob
            )

            if (
                len(self.probs_frente)
                == FRAMES_POR_FACE
            ):

                self.media_frente = (
                    sum(self.probs_frente)
                    / FRAMES_POR_FACE
                )

                self.estado = "virar"

                print(
                    "[PECA] Frente:",
                    self.classificar(
                        self.media_frente
                    ),
                    f"({self.media_frente:.1%}). "
                    "Vire a peca e pressione ESPACO."
                )

        # ----------------------------------------------------
        # VERSO
        # ----------------------------------------------------

        elif self.estado == "verso":

            self.probs_verso.append(
                prob
            )

            if (
                len(self.probs_verso)
                == FRAMES_POR_FACE
            ):

                self.media_verso = (
                    sum(self.probs_verso)
                    / FRAMES_POR_FACE
                )

                # Mesma regra do treino:
                #
                # max(media(frente), media(verso))

                self.resultado_peca = max(
                    self.media_frente,
                    self.media_verso,
                )

                self.estado = "concluida"

                self.gravar_resultado()

    # ========================================================
    # GRAVAR RESULTADO
    # ========================================================

    def gravar_resultado(self):

        novo = (
            not RELATORIO.exists()
            or RELATORIO.stat().st_size == 0
        )

        duracao = (
            time.monotonic()
            - self.inicio_peca

            if self.inicio_peca is not None

            else 0.0
        )

        with RELATORIO.open(
            "a",
            newline="",
            encoding=(
                "utf-8-sig"
                if novo
                else "utf-8"
            ),
        ) as arquivo:

            gravador = csv.writer(
                arquivo,
                delimiter=";",
            )

            if novo:

                gravador.writerow([
                    "data_hora",
                    "modelo",
                    "epoca_modelo",
                    "limiar",
                    "resultado_frente",
                    "prob_nc_frente",
                    "resultado_verso",
                    "prob_nc_verso",
                    "resultado_peca",
                    "prob_nc_peca",
                    "frames_por_face",
                    "duracao_s",
                    "roi_frente_xy",
                    "roi_verso_xy",
                ])

            gravador.writerow([

                datetime.now().isoformat(
                    timespec="seconds"
                ),

                self.arquivo_modelo,

                self.epoca_modelo,

                f"{self.limiar:.4f}",

                self.classificar(
                    self.media_frente
                ),

                f"{self.media_frente:.6f}",

                self.classificar(
                    self.media_verso
                ),

                f"{self.media_verso:.6f}",

                self.classificar(
                    self.resultado_peca
                ),

                f"{self.resultado_peca:.6f}",

                FRAMES_POR_FACE,

                f"{duracao:.3f}",

                str(self.roi_frente),

                str(self.roi_verso),
            ])

        print(
            f"[RESULTADO] "
            f"Modelo={self.arquivo_modelo}; "
            f"Frente={self.classificar(self.media_frente)} "
            f"({self.media_frente:.1%}); "
            f"Verso={self.classificar(self.media_verso)} "
            f"({self.media_verso:.1%}); "
            f"Peca={self.classificar(self.resultado_peca)} "
            f"({self.resultado_peca:.1%})"
        )

        print(
            "[CSV]",
            RELATORIO
        )

    # ========================================================
    # DESENHAR INTERFACE
    # ========================================================

    def desenhar(
        self,
        frame,
        fps_stream,
        conectado,
    ):

        previa = cv2.resize(
            frame,
            (
                self.pw,
                self.ph,
            ),
            interpolation=cv2.INTER_AREA,
        )

        escala_x = (
            self.pw / self.w
        )

        escala_y = (
            self.ph / self.h
        )

        cor_roi = (
            (50, 220, 90)

            if self.estado in (
                "frente",
                "verso",
            )

            else (0, 220, 255)
        )

        cv2.rectangle(
            previa,

            (
                round(self.x * escala_x),
                round(self.y * escala_y),
            ),

            (
                round(
                    (self.x + ROI_LADO)
                    * escala_x
                ),
                round(
                    (self.y + ROI_LADO)
                    * escala_y
                ),
            ),

            cor_roi,
            2,
        )

        # ----------------------------------------------------
        # PAINEL
        # ----------------------------------------------------

        tela = cv2.copyMakeBorder(
            previa,
            0,
            ALTURA_PAINEL,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(20, 25, 31),
        )

        def escrever(
            mensagem,
            y,
            cor=(235, 239, 242),
            tamanho=0.52,
        ):

            cv2.putText(
                tela,
                str(mensagem),

                (
                    12,
                    self.ph + y,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                tamanho,
                cor,
                1,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # MODELO ATIVO
        # ----------------------------------------------------

        escrever(
            f"MODELO: {self.arquivo_modelo} "
            f"({self.indice_modelo}/{self.total_modelos}) "
            f"| Melhor epoca: {self.epoca_modelo} "
            f"| Limiar: {self.limiar:.2f}",
            24,
            (165, 210, 245),
            0.52,
        )

        # ----------------------------------------------------
        # RESULTADO CONFORME O MODO
        # ----------------------------------------------------

        valor = None

        if self.estado == "continuo":

            if (
                len(self.historico)
                < JANELA_TEMPORAL
            ):

                rotulo = (
                    "ANALISANDO... "
                    f"{len(self.historico)}/"
                    f"{JANELA_TEMPORAL}"
                )

            else:

                valor = (
                    sum(self.historico)
                    / len(self.historico)
                )

                rotulo = (
                    self.classificar(valor)
                    + " (CONTINUO)"
                )

        elif self.estado == "frente":

            rotulo = (
                "CAPTURANDO FRENTE: "
                f"{len(self.probs_frente)}/"
                f"{FRAMES_POR_FACE}"
            )

        elif self.estado == "virar":

            valor = self.media_frente

            rotulo = (
                "FRENTE CONCLUIDA - "
                "VIRE A PECA (ESPACO)"
            )

        elif self.estado == "verso":

            rotulo = (
                "CAPTURANDO VERSO: "
                f"{len(self.probs_verso)}/"
                f"{FRAMES_POR_FACE}"
            )

        else:

            valor = (
                self.resultado_peca
            )

            rotulo = (
                self.classificar(valor)
                + " (PECA)"
            )

        # ----------------------------------------------------
        # COR DO RESULTADO
        # ----------------------------------------------------

        if valor is not None:

            if valor >= self.limiar:

                cor_resultado = (
                    80,
                    90,
                    245,
                )

            else:

                cor_resultado = (
                    100,
                    220,
                    110,
                )

        else:

            cor_resultado = (
                185,
                225,
                240,
            )

        escrever(
            rotulo,
            54,
            cor_resultado,
            0.69,
        )

        # ----------------------------------------------------
        # RESULTADOS INDIVIDUAIS
        # ----------------------------------------------------

        if self.estado == "concluida":

            escrever(
                f"Frente: "
                f"{self.classificar(self.media_frente)} "
                f"({self.media_frente:.1%} NC) | "
                f"Verso: "
                f"{self.classificar(self.media_verso)} "
                f"({self.media_verso:.1%} NC)",
                83,
            )

        elif self.estado == "virar":

            escrever(
                "Resultado frente: "
                f"{self.classificar(self.media_frente)} "
                f"| Prob. NC={self.media_frente:.1%}",
                83,
            )

        else:

            prob = (
                valor
                if valor is not None
                else self.prob_inst
            )

            texto = (
                f"{prob:.1%}"
                if prob is not None
                else "--"
            )

            escrever(
                "Prob. estimada de nao conforme: "
                + texto,
                83,
            )

        # ----------------------------------------------------
        # DESEMPENHO
        # ----------------------------------------------------

        situacao = (
            "CONECTADA"
            if conectado
            else "SEM IMAGEM NOVA"
        )

        escrever(
            f"Camera {self.w}x{self.h} | "
            f"Stream {fps_stream:.1f} FPS | "
            f"IA {self.fps_ia:.1f} inf/s | "
            f"Inferencia "
            f"{self.ultima_inferencia_ms:.1f} ms | "
            f"{situacao}",
            111,
            (177, 192, 207),
            0.44,
        )

        # ----------------------------------------------------
        # CONTROLES
        # ----------------------------------------------------

        escrever(
            "M: proximo modelo | "
            "B: anterior | "
            "U: atualizar lista | "
            "I: inspecionar peca",
            145,
            (180, 199, 215),
            0.48,
        )

        escrever(
            "ESPACO: verso | "
            "L: continuo | "
            "R: reiniciar | "
            "Mouse: ROI | "
            "Q/ESC: sair",
            175,
            (180, 199, 215),
            0.48,
        )

        cv2.imshow(
            JANELA,
            tela,
        )


# ============================================================
# LISTA DE MODELOS
# ============================================================

def imprimir_modelos(
    caminhos,
    atual=None,
):

    print()
    print("Modelos encontrados:")

    for i, caminho in enumerate(
        caminhos,
        1,
    ):

        marcador = (
            "*"
            if caminho == atual
            else " "
        )

        print(
            f"  {marcador} "
            f"{i:2d} - {caminho.name}"
        )


# ============================================================
# ESCOLHER MODELO INICIAL
# ============================================================

def escolher_inicial(
    modelos,
    escolha,
):

    if escolha == "mais_recente":

        return len(modelos) - 1

    solicitado = Path(
        escolha
    ).name

    if not solicitado.endswith(".pt"):

        solicitado += ".pt"

    for i, caminho in enumerate(
        modelos
    ):

        if (
            caminho.name.lower()
            == solicitado.lower()
        ):

            return i

    raise ValueError(
        f"Modelo '{escolha}' nao encontrado "
        f"em {PASTA_MODELOS}."
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENTOS DO TERMINAL
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Inspecao em tempo real - EfficientNet-B0"
        )
    )

    parser.add_argument(
        "--modelo",
        default=MODELO_INICIAL,
        help=(
            "treino_4.pt, treino_5.pt "
            "ou mais_recente"
        ),
    )

    parser.add_argument(
        "--listar-modelos",
        action="store_true",
        help=(
            "Mostra checkpoints e encerra"
        ),
    )

    argumentos = parser.parse_args()

    # --------------------------------------------------------
    # ENCONTRAR MODELOS
    # --------------------------------------------------------

    modelos = listar_modelos()

    if not modelos:

        raise FileNotFoundError(
            "Nenhum arquivo .pt encontrado em "
            f"{PASTA_MODELOS}"
        )

    if argumentos.listar_modelos:

        imprimir_modelos(
            modelos
        )

        return

    indice = escolher_inicial(
        modelos,
        argumentos.modelo,
    )

    # --------------------------------------------------------
    # CARREGAR PRIMEIRO MODELO
    # --------------------------------------------------------

    ia = Classificador()

    ia.carregar(
        modelos[indice]
    )

    imprimir_modelos(
        modelos,
        ia.caminho,
    )

    print()
    print("CONTROLES:")
    print("M - proximo modelo")
    print("B - modelo anterior")
    print("U - atualizar modelos")
    print("I - inspecionar uma peca")
    print("ESPACO - iniciar verso")
    print("L - modo continuo")
    print("R - reiniciar")
    print("Q - sair")

    print()
    print(
        "A ESP32 e o NeoPixel "
        "nao serao reconfigurados."
    )

    # --------------------------------------------------------
    # INICIAR CAMERA
    # --------------------------------------------------------

    camera = CameraMJPEG(
        STREAM_URL
    )

    camera.iniciar()

    cv2.namedWindow(
        JANELA,
        cv2.WINDOW_AUTOSIZE,
    )

    ui = None

    ultima_seq = -1

    ultimo_frame = None

    ultimo_aviso = 0.0

    # ========================================================
    # TROCAR MODELO SEM REINICIAR A CAMERA
    # ========================================================

    def trocar_modelo(
        novo_indice,
        novos_modelos=None,
    ):

        nonlocal indice, modelos

        lista = (
            novos_modelos
            if novos_modelos is not None
            else modelos
        )

        caminho = lista[
            novo_indice
        ]

        try:

            ia.carregar(
                caminho
            )

        except Exception as erro:

            print(
                "[ERRO] Nao foi possivel selecionar "
                f"{caminho.name}: {erro}"
            )

            return

        modelos = lista

        indice = novo_indice

        if ui is not None:

            ui.atualizar_modelo(
                ia,
                indice,
                len(modelos),
            )

            print(
                "[AVISO] Historico e inspecao "
                "parcial reiniciados."
            )

        imprimir_modelos(
            modelos,
            ia.caminho,
        )

    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    try:

        while True:

            (
                frame,
                seq,
                fps_stream,
                conectado,
                ultimo_recebimento,
            ) = camera.obter(
                ultima_seq
            )

            agora = time.monotonic()

            # ------------------------------------------------
            # NOVO FRAME
            # ------------------------------------------------

            if frame is not None:

                ultima_seq = seq

                ultimo_frame = frame

                h, w = frame.shape[:2]

                # --------------------------------------------
                # CRIAR INTERFACE
                # --------------------------------------------

                if (
                    ui is None
                    or (ui.w, ui.h) != (w, h)
                ):

                    ui = Interface(
                        w,
                        h,
                        ia,
                    )

                    ui.atualizar_modelo(
                        ia,
                        indice,
                        len(modelos),
                    )

                    cv2.setMouseCallback(
                        JANELA,
                        ui.mouse,
                    )

                    print(
                        f"[STREAM] {w}x{h}; "
                        "ROI de 320x320."
                    )

                # --------------------------------------------
                # INFERENCIA
                # --------------------------------------------

                ativo = ui.estado in (
                    "continuo",
                    "frente",
                    "verso",
                )

                intervalo_ok = (
                    agora - ui.ultima_inferencia
                    >= INTERVALO_IA
                )

                if (
                    ativo
                    and not ui.arrastando
                    and intervalo_ok
                ):

                    recorte = frame[
                        ui.y:ui.y + ROI_LADO,
                        ui.x:ui.x + ROI_LADO,
                    ]

                    if recorte.shape[:2] == (
                        ROI_LADO,
                        ROI_LADO,
                    ):

                        prob_nc, ms = ia.prever(
                            recorte
                        )

                        ui.receber_previsao(
                            prob_nc,
                            ms,
                        )

                        ui.ultima_inferencia = (
                            time.monotonic()
                        )

            # ------------------------------------------------
            # MOSTRAR JANELA
            # ------------------------------------------------

            if (
                ui is not None
                and ultimo_frame is not None
            ):

                # --------------------------------------------
                # MEDIR INFERENCIAS POR SEGUNDO
                # --------------------------------------------

                intervalo = (
                    agora - ui.tempo_fps
                )

                if intervalo >= 1.5:

                    ui.fps_ia = (
                        ui.contagem_ia
                        / max(intervalo, 0.001)
                    )

                    ui.contagem_ia = 0

                    ui.tempo_fps = agora

                # --------------------------------------------
                # ESTADO DA CAMERA
                # --------------------------------------------

                atualizado = (
                    conectado
                    and
                    agora - ultimo_recebimento < 3.0
                )

                ui.desenhar(
                    ultimo_frame,
                    fps_stream,
                    atualizado,
                )

            elif agora - ultimo_aviso >= 3:

                print(
                    "[CAMERA] Aguardando stream:",
                    STREAM_URL,
                )

                ultimo_aviso = agora

            # ------------------------------------------------
            # TECLADO
            # ------------------------------------------------

            tecla = (
                cv2.waitKey(10)
                & 0xFF
            )

            # SAIR

            if tecla in (
                ord("q"),
                ord("Q"),
                27,
            ):

                break

            # PROXIMO MODELO

            if tecla in (
                ord("m"),
                ord("M"),
            ):

                trocar_modelo(
                    (indice + 1) % len(modelos)
                )

            # MODELO ANTERIOR

            elif tecla in (
                ord("b"),
                ord("B"),
            ):

                trocar_modelo(
                    (indice - 1) % len(modelos)
                )

            # ATUALIZAR LISTA

            elif tecla in (
                ord("u"),
                ord("U"),
            ):

                novos = listar_modelos()

                if not novos:

                    print(
                        "[MODELOS] Pasta vazia; "
                        "mantendo o modelo atual."
                    )

                else:

                    # Seleciona o mais recente.
                    trocar_modelo(
                        len(novos) - 1,
                        novos,
                    )

            elif ui is not None:

                # INICIAR INSPECAO DE PECA

                if tecla in (
                    ord("i"),
                    ord("I"),
                ):

                    ui.iniciar_peca()

                # CONFIRMAR VERSO

                elif tecla == 32:

                    ui.iniciar_verso()

                # REINICIAR / MODO CONTINUO

                elif tecla in (
                    ord("l"),
                    ord("L"),
                    ord("r"),
                    ord("R"),
                ):

                    ui.reiniciar()

                    print(
                        "[UI] Modo continuo; "
                        "historico limpo."
                    )

    except KeyboardInterrupt:

        print(
            "[INFO] Interrompido pelo usuario."
        )

    finally:

        camera.fechar()

        cv2.destroyAllWindows()

        print(
            "[INFO] Inspecao encerrada."
        )


# ============================================================
# INICIO
# ============================================================

if __name__ == "__main__":

    main()
