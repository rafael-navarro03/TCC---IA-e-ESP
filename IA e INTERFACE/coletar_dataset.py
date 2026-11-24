
"""
Coleta padronizada com rotulos independentes por face.

C/N/U antes da frente:
    Define o rotulo visual da frente e inicia 4 fotos.

C/N/U apos virar:
    Define o rotulo visual do verso e inicia 4 fotos.

U = nao avaliada.
Qualquer U envia a peca para dataset/_revisar.

Quatro transformacoes por face, em frames diferentes:
01 - Original
02 - Inversao vertical
03 - Espelhamento horizontal
04 - Clareamento suave por gamma

A classe fisica e NAO CONFORME se ao menos uma
face for nao conforme.

A classe fisica e CONFORME se ambas forem conformes.

O Python nao modifica a ESP32 nem o NeoPixel.
"""

import json
import time

from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import cv2
import numpy as np


# ============================================================
# CONFIGURACAO
# ============================================================

STREAM_URL = "http://192.168.0.105:81/stream"

BASE = Path(__file__).resolve().parent

DATASET = BASE / "dataset"
ROI_FILE = BASE / "roi.json"

ROI_LADO = 320

FOTOS_POR_FACE = 4

INTERVALO_FOTOS = 0.60
ESPERA_INICIAL = 1.20

GAMMA_CLAREAMENTO = 0.90

PREVIEW_MAX_W = 1100
PREVIEW_MAX_H = 760

PAINEL_H = 148

JANELA = "Coleta por face | ESP32"


# ============================================================
# TRANSFORMACOES
# ============================================================

TRANSFORMACOES = (
    "original",
    "inversao_vertical",
    "espelhamento_horizontal",
    "clareamento_gamma",
)

ROTULOS_TECLAS = {
    ord("c"): "conforme",
    ord("n"): "nao_conforme",
    ord("u"): "nao_avaliada",
}


# ============================================================
# TABELA DE CLAREAMENTO
# ============================================================

GAMMA_LUT = np.clip(
    (
        np.arange(
            256,
            dtype=np.float32
        ) / 255.0
    ) ** GAMMA_CLAREAMENTO * 255.0,
    0,
    255
).round().astype(np.uint8)


# ============================================================
# DATA E HORA
# ============================================================

def agora_iso():

    return datetime.now().isoformat(
        timespec="milliseconds"
    )


# ============================================================
# GRAVACAO PNG COMPATIVEL COM CAMINHOS UNICODE
# ============================================================

def gravar_png(caminho, imagem):

    if (
        imagem is None
        or imagem.shape[:2] != (
            ROI_LADO,
            ROI_LADO
        )
    ):

        raise ValueError(
            "Imagem vazia ou diferente de 320x320."
        )

    sucesso, buffer = cv2.imencode(
        ".png",
        imagem,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            3
        ]
    )

    if not sucesso:

        raise OSError(
            "O OpenCV nao conseguiu codificar o PNG."
        )

    caminho.write_bytes(
        buffer.tobytes()
    )

    if caminho.stat().st_size == 0:

        raise OSError(
            "Arquivo PNG vazio."
        )


# ============================================================
# TRANSFORMACOES DE IMAGEM
# ============================================================

def transformar(imagem, transformacao):

    if transformacao == "original":

        return imagem.copy()

    if transformacao == "inversao_vertical":

        return cv2.flip(
            imagem,
            0
        )

    if transformacao == "espelhamento_horizontal":

        return cv2.flip(
            imagem,
            1
        )

    if transformacao == "clareamento_gamma":

        return cv2.LUT(
            imagem,
            GAMMA_LUT
        )

    raise ValueError(
        f"Transformacao desconhecida: {transformacao}"
    )


# ============================================================
# CONSULTA STATUS DA ESP32
#
# Apenas leitura.
# Nao utiliza /control nem /neo.
# ============================================================

def consultar_status():

    partes = urlsplit(
        STREAM_URL
    )

    if not partes.hostname:
        return {}

    url = (
        f"{partes.scheme}://"
        f"{partes.hostname}/status"
    )

    try:

        with urlopen(
            url,
            timeout=2.5
        ) as resposta:

            return json.loads(
                resposta.read().decode("utf-8")
            )

    except Exception as erro:

        print(
            "Aviso: nao foi possivel consultar /status:",
            erro
        )

        return {}


# ============================================================
# DEFINICAO DA CLASSE DA PECA FISICA
# ============================================================

def classe_da_peca(rotulos):

    frente = rotulos["frente"]
    verso = rotulos["verso"]

    if "nao_avaliada" in (
        frente,
        verso
    ):

        return "revisao_pendente"

    if "nao_conforme" in (
        frente,
        verso
    ):

        return "nao_conforme"

    return "conforme"


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class Coletor:

    def __init__(self):

        # ----------------------------------------------------
        # PASTAS
        # ----------------------------------------------------

        for pasta in (
            "conforme",
            "nao_conforme",
            "_incompletas",
            "_revisar"
        ):

            (
                DATASET / pasta
            ).mkdir(
                parents=True,
                exist_ok=True
            )

        # ----------------------------------------------------
        # STREAM
        # ----------------------------------------------------

        print(
            "Conectando a:",
            STREAM_URL
        )

        self.cap = cv2.VideoCapture(
            STREAM_URL
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Nao foi possivel abrir "
                "a stream da ESP32."
            )

        ok, frame = self.cap.read()

        if not ok or frame is None:

            self.cap.release()

            raise RuntimeError(
                "A stream nao forneceu "
                "o primeiro frame."
            )

        self.frame_inicial = frame

        self.altura, self.largura = (
            frame.shape[:2]
        )

        if (
            self.largura < ROI_LADO
            or self.altura < ROI_LADO
        ):

            self.cap.release()

            raise RuntimeError(
                "Resolucao insuficiente "
                "para ROI de 320x320."
            )

        # ----------------------------------------------------
        # ESCALA DA PREVIA
        # ----------------------------------------------------

        escala = min(
            1.0,
            PREVIEW_MAX_W / self.largura,
            PREVIEW_MAX_H / self.altura
        )

        self.preview_w = max(
            1,
            round(self.largura * escala)
        )

        self.preview_h = max(
            1,
            round(self.altura * escala)
        )

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        self.roi_x = (
            self.largura - ROI_LADO
        ) // 2

        self.roi_y = (
            self.altura - ROI_LADO
        ) // 2

        self.carregar_roi()

        self.arrastando = False

        self.offset_x = 0
        self.offset_y = 0

        # ----------------------------------------------------
        # ESTADOS
        #
        # aguardando_frente
        # capturando_frente
        # aguardando_verso
        # capturando_verso
        # erro
        # ----------------------------------------------------

        self.etapa = "aguardando_frente"

        self.rotulos = {
            "frente": None,
            "verso": None
        }

        self.contadores = {
            "frente": 0,
            "verso": 0
        }

        self.provisoria = None
        self.destino = None
        self.metadados = None

        self.proxima_foto = 0.0

        self.aviso = (
            "Avalie a FRENTE: "
            "C=conforme  "
            "N=nao conforme  "
            "U=nao avaliada."
        )

        # ----------------------------------------------------
        # TESTE DE GRAVACAO
        # ----------------------------------------------------

        teste = (
            DATASET
            / "_incompletas"
            / "_teste_escrita.png"
        )

        try:

            gravar_png(
                teste,
                frame[:ROI_LADO, :ROI_LADO]
            )

            print(
                "Teste de gravacao PNG: OK"
            )

        except Exception:

            self.cap.release()
            raise

        finally:

            teste.unlink(
                missing_ok=True
            )

    # ========================================================
    # LIMITAR ROI
    # ========================================================

    def limitar_roi(self, x, y):

        self.roi_x = max(
            0,
            min(
                int(x),
                self.largura - ROI_LADO
            )
        )

        self.roi_y = max(
            0,
            min(
                int(y),
                self.altura - ROI_LADO
            )
        )

    # ========================================================
    # ROI ATUAL
    # ========================================================

    def roi_atual(self):

        return {
            "x": self.roi_x,
            "y": self.roi_y,
            "w": ROI_LADO,
            "h": ROI_LADO
        }

    # ========================================================
    # SALVAR ROI
    # ========================================================

    def salvar_roi(self):

        dados = {
            **self.roi_atual(),
            "frame_width": self.largura,
            "frame_height": self.altura
        }

        ROI_FILE.write_text(
            json.dumps(
                dados,
                indent=2
            ),
            encoding="utf-8"
        )

    # ========================================================
    # CARREGAR ROI
    # ========================================================

    def carregar_roi(self):

        if not ROI_FILE.exists():
            return

        try:

            dados = json.loads(
                ROI_FILE.read_text(
                    encoding="utf-8"
                )
            )

            resolucao_salva = (
                dados.get("frame_width"),
                dados.get("frame_height")
            )

            resolucao_atual = (
                self.largura,
                self.altura
            )

            if resolucao_salva != resolucao_atual:

                print(
                    "Resolucao mudou: "
                    "ROI centralizada novamente."
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
                cy - ROI_LADO // 2
            )

            print(
                "ROI anterior carregada."
            )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError
        ) as erro:

            print(
                "Aviso: ROI anterior ignorada:",
                erro
            )

    # ========================================================
    # CONTROLE DA ROI COM O MOUSE
    # ========================================================

    def mouse(
        self,
        evento,
        px,
        py,
        flags,
        userdata
    ):

        # Somente antes da frente
        # ou durante a pausa antes do verso.

        if self.etapa not in (
            "aguardando_frente",
            "aguardando_verso"
        ):

            return

        if (
            evento == cv2.EVENT_LBUTTONDOWN
            and py >= self.preview_h
        ):

            return

        x = min(
            max(
                int(
                    px * self.largura
                    / self.preview_w
                ),
                0
            ),
            self.largura - 1
        )

        y = min(
            max(
                int(
                    py * self.altura
                    / self.preview_h
                ),
                0
            ),
            self.altura - 1
        )

        # ----------------------------------------------------
        # CLIQUE
        # ----------------------------------------------------

        if evento == cv2.EVENT_LBUTTONDOWN:

            dentro = (
                self.roi_x <= x
                < self.roi_x + ROI_LADO
                and
                self.roi_y <= y
                < self.roi_y + ROI_LADO
            )

            if dentro:

                self.offset_x = (
                    x - self.roi_x
                )

                self.offset_y = (
                    y - self.roi_y
                )

            else:

                self.offset_x = (
                    ROI_LADO // 2
                )

                self.offset_y = (
                    ROI_LADO // 2
                )

            self.arrastando = True

            self.limitar_roi(
                x - self.offset_x,
                y - self.offset_y
            )

        # ----------------------------------------------------
        # ARRASTAR
        # ----------------------------------------------------

        elif (
            evento == cv2.EVENT_MOUSEMOVE
            and self.arrastando
        ):

            self.limitar_roi(
                x - self.offset_x,
                y - self.offset_y
            )

        # ----------------------------------------------------
        # SOLTAR
        # ----------------------------------------------------

        elif (
            evento == cv2.EVENT_LBUTTONUP
            and self.arrastando
        ):

            self.limitar_roi(
                x - self.offset_x,
                y - self.offset_y
            )

            self.arrastando = False

            self.salvar_roi()

            print(
                f"ROI: x={self.roi_x}, "
                f"y={self.roi_y}, 320x320"
            )

    # ========================================================
    # SALVAR METADADOS
    # ========================================================

    def salvar_metadados(self):

        arquivo = (
            self.provisoria
            / "coleta.json"
        )

        arquivo.write_text(
            json.dumps(
                self.metadados,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    # ========================================================
    # ROTULAR FRENTE E INICIAR COLETA
    # ========================================================

    def rotular_frente(self, rotulo):

        if self.etapa != "aguardando_frente":
            return

        identificador = (
            "peca_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        self.provisoria = (
            DATASET
            / "_incompletas"
            / f"coleta_{identificador}"
        )

        self.provisoria.mkdir(
            parents=True,
            exist_ok=False
        )

        self.destino = None

        self.rotulos = {
            "frente": rotulo,
            "verso": None
        }

        self.contadores = {
            "frente": 0,
            "verso": 0
        }

        # ----------------------------------------------------
        # METADADOS INICIAIS
        # ----------------------------------------------------

        self.metadados = {

            "identificador": identificador,

            "situacao": "em_coleta",

            "inicio": agora_iso(),

            "stream": STREAM_URL,

            "resolucao_frame": [
                self.largura,
                self.altura
            ],

            "roi_frente": self.roi_atual(),

            "roi_verso": None,

            "rotulo_frente": rotulo,

            "rotulo_verso": None,

            "classe_peca": None,

            "regra_peca": (
                "Nao conforme se qualquer face "
                "e nao conforme; conforme se "
                "ambas sao conformes."
            ),

            "plano": {
                "frente": FOTOS_POR_FACE,
                "verso": FOTOS_POR_FACE
            },

            "transformacoes_por_lado": list(
                TRANSFORMACOES
            ),

            "gamma_clareamento": (
                GAMMA_CLAREAMENTO
            ),

            "configuracao_esp32": (
                consultar_status()
            ),

            "imagens": []
        }

        self.salvar_metadados()

        self.salvar_roi()

        # ----------------------------------------------------
        # INICIAR FRENTE
        # ----------------------------------------------------

        self.etapa = "capturando_frente"

        self.proxima_foto = (
            time.monotonic()
            + ESPERA_INICIAL
        )

        self.aviso = (
            f"FRENTE ({rotulo}): "
            "capturando 4 imagens."
        )

        print()
        print(
            f"Nova peca: {identificador}"
        )

        print(
            f"Rotulo da frente: {rotulo}"
        )

    # ========================================================
    # ROTULAR VERSO E INICIAR COLETA
    # ========================================================

    def rotular_verso(self, rotulo):

        if self.etapa != "aguardando_verso":
            return

        self.rotulos["verso"] = rotulo

        self.metadados[
            "rotulo_verso"
        ] = rotulo

        self.metadados[
            "roi_verso"
        ] = self.roi_atual()

        self.salvar_metadados()

        self.salvar_roi()

        # ----------------------------------------------------
        # INICIAR VERSO
        # ----------------------------------------------------

        self.etapa = "capturando_verso"

        self.proxima_foto = (
            time.monotonic()
            + ESPERA_INICIAL
        )

        self.aviso = (
            f"VERSO ({rotulo}): "
            "capturando 4 imagens."
        )

        print(
            f"Rotulo do verso: {rotulo}"
        )

        print(
            "Iniciando 4 fotografias."
        )

    # ========================================================
    # CAPTURAR E SALVAR FOTOGRAFIA
    # ========================================================

    def salvar_foto(self, frame):

        if self.etapa not in (
            "capturando_frente",
            "capturando_verso"
        ):

            return

        if (
            time.monotonic()
            < self.proxima_foto
        ):

            return

        # ----------------------------------------------------
        # IDENTIFICAR FACE ATUAL
        # ----------------------------------------------------

        lado = (
            "frente"
            if self.etapa == "capturando_frente"
            else "verso"
        )

        indice = self.contadores[lado]

        if indice >= FOTOS_POR_FACE:
            return

        transformacao = (
            TRANSFORMACOES[indice]
        )

        # ----------------------------------------------------
        # RECORTE ORIGINAL 320X320
        # ----------------------------------------------------

        recorte = frame[
            self.roi_y:self.roi_y + ROI_LADO,
            self.roi_x:self.roi_x + ROI_LADO
        ]

        if recorte.shape[:2] != (
            ROI_LADO,
            ROI_LADO
        ):

            self.etapa = "erro"

            raise RuntimeError(
                "ROI fora dos limites do frame."
            )

        # ----------------------------------------------------
        # NOME DO ARQUIVO
        # ----------------------------------------------------

        nome = (
            f"{lado}_{indice + 1:02d}.png"
        )

        # ----------------------------------------------------
        # TRANSFORMAR E GRAVAR
        # ----------------------------------------------------

        try:

            imagem = transformar(
                recorte,
                transformacao
            )

            gravar_png(
                self.provisoria / nome,
                imagem
            )

            # ------------------------------------------------
            # REGISTRAR ROTULO DA FACE
            # ------------------------------------------------

            self.metadados[
                "imagens"
            ].append({

                "arquivo": nome,

                "lado": lado,

                "rotulo_face": (
                    self.rotulos[lado]
                ),

                "numero": indice + 1,

                "transformacao": (
                    transformacao
                ),

                "horario": agora_iso(),

                "roi": self.roi_atual()
            })

            self.contadores[lado] += 1

            self.salvar_metadados()

        except (
            OSError,
            ValueError,
            cv2.error
        ) as erro:

            self.etapa = "erro"

            self.aviso = (
                "ERRO AO SALVAR. "
                "Consulte o terminal."
            )

            print(
                "Falha ao salvar foto "
                "ou metadados:",
                erro
            )

            return

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        print(
            f"{lado.upper()} "
            f"{indice + 1}/4: "
            f"{nome} | "
            f"{transformacao} | "
            f"rotulo={self.rotulos[lado]}"
        )

        self.proxima_foto = (
            time.monotonic()
            + INTERVALO_FOTOS
        )

        # ----------------------------------------------------
        # CONCLUIU FRENTE
        # ----------------------------------------------------

        if (
            lado == "frente"
            and
            self.contadores[lado]
            == FOTOS_POR_FACE
        ):

            self.etapa = "aguardando_verso"

            self.aviso = (
                "VIRE A PECA. Ajuste ROI. "
                "Rotule o VERSO: C / N / U."
            )

            print()
            print(
                "Frente concluida."
            )

            print(
                "Vire a peca e rotule "
                "o verso: C, N ou U."
            )
            print()

        # ----------------------------------------------------
        # CONCLUIU VERSO
        # ----------------------------------------------------

        elif (
            lado == "verso"
            and
            self.contadores[lado]
            == FOTOS_POR_FACE
        ):

            self.concluir_peca()

    # ========================================================
    # FINALIZAR PECA
    # ========================================================

    def concluir_peca(self):

        # ----------------------------------------------------
        # VERIFICAR ARQUIVOS
        # ----------------------------------------------------

        esperados = [

            self.provisoria
            / f"{lado}_{i:02d}.png"

            for lado in (
                "frente",
                "verso"
            )

            for i in range(
                1,
                FOTOS_POR_FACE + 1
            )
        ]

        if any(
            not arquivo.is_file()
            for arquivo in esperados
        ):

            self.etapa = "erro"

            print(
                "Erro: a coleta esta incompleta. "
                "Pasta preservada."
            )

            return

        # ----------------------------------------------------
        # DECIDIR CLASSE DA PECA
        # ----------------------------------------------------

        resultado = classe_da_peca(
            self.rotulos
        )

        identificador = (
            self.metadados["identificador"]
        )

        if resultado == "revisao_pendente":

            destino = (
                DATASET
                / "_revisar"
                / identificador
            )

        else:

            destino = (
                DATASET
                / resultado
                / identificador
            )

        self.metadados[
            "classe_peca"
        ] = (
            None
            if resultado == "revisao_pendente"
            else resultado
        )

        self.metadados[
            "situacao"
        ] = (
            "revisao_pendente"
            if resultado == "revisao_pendente"
            else "concluida"
        )

        self.metadados["fim"] = (
            agora_iso()
        )

        self.metadados[
            "total_imagens"
        ] = len(esperados)

        # ----------------------------------------------------
        # SALVAR E MOVER PASTA
        # ----------------------------------------------------

        try:

            self.salvar_metadados()

            self.provisoria.rename(
                destino
            )

        except OSError as erro:

            self.etapa = "erro"

            self.aviso = (
                "ERRO AO FINALIZAR. "
                "Consulte o terminal."
            )

            print(
                "Erro ao finalizar coleta:",
                erro
            )

            return

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        print()
        print(
            "PECA FINALIZADA:",
            destino
        )

        print(
            "Frente:",
            self.rotulos["frente"]
        )

        print(
            "Verso:",
            self.rotulos["verso"]
        )

        print(
            "Classe da peca:",
            resultado
        )

        print(
            "8 PNGs de 320x320 salvos."
        )
        print()

        # ----------------------------------------------------
        # PREPARAR PROXIMA PECA
        # ----------------------------------------------------

        self.etapa = "aguardando_frente"

        self.aviso = (
            f"Peca: {resultado}. "
            "Nova peca: avalie a FRENTE (C/N/U)."
        )

        self.provisoria = None
        self.destino = None
        self.metadados = None

        self.rotulos = {
            "frente": None,
            "verso": None
        }

        self.contadores = {
            "frente": 0,
            "verso": 0
        }

    # ========================================================
    # INTERFACE
    # ========================================================

    def apresentar(self, frame):

        previa = cv2.resize(
            frame,
            (
                self.preview_w,
                self.preview_h
            ),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        x1 = round(
            self.roi_x
            * self.preview_w
            / self.largura
        )

        y1 = round(
            self.roi_y
            * self.preview_h
            / self.altura
        )

        x2 = round(
            (self.roi_x + ROI_LADO)
            * self.preview_w
            / self.largura
        )

        y2 = round(
            (self.roi_y + ROI_LADO)
            * self.preview_h
            / self.altura
        )

        cor = (
            (0, 220, 255)
            if self.etapa.startswith("aguardando")
            else (70, 220, 90)
        )

        cv2.rectangle(
            previa,
            (x1, y1),
            (x2, y2),
            cor,
            2
        )

        # ----------------------------------------------------
        # PAINEL
        # ----------------------------------------------------

        tela = cv2.copyMakeBorder(
            previa,
            0,
            PAINEL_H,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(20, 25, 31)
        )

        def escrever(
            texto,
            y,
            cor_texto=(235, 240, 245),
            escala=0.49
        ):

            cv2.putText(
                tela,
                texto,
                (
                    12,
                    self.preview_h + y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                escala,
                cor_texto,
                1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # INFORMACOES
        # ----------------------------------------------------

        escrever(
            self.aviso,
            24,
            (130, 230, 180),
            0.53
        )

        escrever(
            (
                "Rotulos: "
                f"frente={self.rotulos['frente'] or '--'}"
                " | "
                f"verso={self.rotulos['verso'] or '--'}"
            ),
            50
        )

        escrever(
            (
                "Fotos: "
                f"frente {self.contadores['frente']}/4"
                " | "
                f"verso {self.contadores['verso']}/4"
                " | ROI 320x320"
                f" | Stream {self.largura}x{self.altura}"
            ),
            76
        )

        escrever(
            (
                "1 Normal | "
                "2 Vertical | "
                "3 Espelhada | "
                "4 Clareamento leve"
            ),
            104,
            (170, 185, 200),
            0.47
        )

        escrever(
            (
                "C: conforme | "
                "N: nao conforme | "
                "U: nao avaliada | "
                "Mouse: ROI | "
                "R: centralizar | "
                "Q: sair"
            ),
            132,
            (170, 185, 200),
            0.43
        )

        cv2.imshow(
            JANELA,
            tela
        )

    # ========================================================
    # EXECUTAR
    # ========================================================

    def executar(self):

        print()
        print("=" * 58)
        print(
            "COLETA COM ROTULOS POR FACE - ESP32"
        )
        print("=" * 58)

        print(
            "Stream:",
            STREAM_URL
        )

        print(
            "Resolucao:",
            f"{self.largura}x{self.altura}"
        )

        print(
            "ROI: 320x320"
        )

        print(
            "4 imagens por face, "
            "8 imagens por peca."
        )

        print(
            "A ESP32 e o NeoPixel "
            "NAO serao reconfigurados."
        )

        print()
        print(
            "C/N/U: rotular a frente."
        )

        print(
            "Apos virar, C/N/U: rotular o verso."
        )

        print(
            "Mouse: mover ROI."
        )

        print(
            "R: centralizar ROI."
        )

        print(
            "Q ou ESC: encerrar."
        )
        print()

        # ----------------------------------------------------
        # JANELA
        # ----------------------------------------------------

        cv2.namedWindow(
            JANELA,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.setMouseCallback(
            JANELA,
            self.mouse
        )

        frame = self.frame_inicial

        falhas = 0

        try:

            while True:

                # --------------------------------------------
                # FRAME
                # --------------------------------------------

                if frame is None:

                    falhas += 1

                    if falhas >= 15:

                        print(
                            "Muitos frames perdidos. "
                            "Encerrando."
                        )

                        break

                    time.sleep(0.10)

                else:

                    falhas = 0

                    # ----------------------------------------
                    # VERIFICAR RESOLUCAO
                    # ----------------------------------------

                    if frame.shape[:2] != (
                        self.altura,
                        self.largura
                    ):

                        print(
                            "Resolucao da stream mudou. "
                            "Encerrando por seguranca."
                        )

                        break

                    # ----------------------------------------
                    # SALVAR QUANDO PROGRAMADO
                    # ----------------------------------------

                    self.salvar_foto(
                        frame
                    )

                    # ----------------------------------------
                    # EXIBIR
                    # ----------------------------------------

                    self.apresentar(
                        frame
                    )

                # --------------------------------------------
                # TECLADO
                # --------------------------------------------

                tecla = cv2.waitKey(1) & 0xFF

                # ROTULAR FRENTE OU VERSO

                if tecla in ROTULOS_TECLAS:

                    rotulo = (
                        ROTULOS_TECLAS[tecla]
                    )

                    if self.etapa == "aguardando_frente":

                        self.rotular_frente(
                            rotulo
                        )

                    elif self.etapa == "aguardando_verso":

                        self.rotular_verso(
                            rotulo
                        )

                    else:

                        print(
                            "Captura em andamento; "
                            "aguarde a pausa."
                        )

                # CENTRALIZAR ROI

                elif (
                    tecla == ord("r")
                    and self.etapa in (
                        "aguardando_frente",
                        "aguardando_verso"
                    )
                ):

                    self.limitar_roi(
                        (
                            self.largura - ROI_LADO
                        ) // 2,
                        (
                            self.altura - ROI_LADO
                        ) // 2
                    )

                    self.salvar_roi()

                    self.aviso = (
                        "ROI centralizada. "
                        "Posicione sobre a peca."
                    )

                # SAIR

                elif tecla in (
                    ord("q"),
                    27
                ):

                    break

                # --------------------------------------------
                # ERRO
                # --------------------------------------------

                if self.etapa == "erro":

                    print(
                        "Coleta interrompida por erro. "
                        "Pasta parcial preservada."
                    )

                    break

                # --------------------------------------------
                # PROXIMO FRAME
                # --------------------------------------------

                ok, frame = self.cap.read()

                if not ok:

                    frame = None

        except KeyboardInterrupt:

            print(
                "Interrompido pelo teclado."
            )

        finally:

            self.cap.release()

            cv2.destroyAllWindows()

            if self.provisoria is not None:

                print(
                    "Coleta incompleta mantida em:",
                    self.provisoria
                )

            print(
                "Programa encerrado."
            )


# ============================================================
# INICIO
# ============================================================

def main():

    coletor = Coletor()
    coletor.executar()


if __name__ == "__main__":

    main()
