"""Painel web de inspeção visual - TCC.

Servidor LAN no PC. ESP32 mantém a câmera/NeoPixel; EfficientNet roda no PC.
Abra http://localhost:8765 ou, na rede, http://IP_DO_PC:8765.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# CONFIGURAÇÃO DO PROJETO
# ============================================================
BASE = Path(__file__).resolve().parent
# Se o painel estiver dentro de IA/painel_tcc, utiliza IA/modelos automaticamente.
MODELOS_LOCAL = BASE / "modelos"
MODELOS_PAI = BASE.parent / "modelos"
MODELOS = (MODELOS_PAI if MODELOS_PAI.is_dir() and any(MODELOS_PAI.glob("*.pt"))
           else MODELOS_LOCAL)
WEB = BASE / "web"
LOG = BASE / "resultados_tcc.csv"
ROI_FILE = (BASE.parent / "roi.json" if (BASE.parent / "roi.json").exists()
            else BASE / "roi.json")

# Ajuste caso o IP da ESP32 tenha mudado.
ESP_IP = os.environ.get("TCC_ESP_IP", "192.168.0.105")
ESP_HTTP = f"http://{ESP_IP}"
STREAM_URL = f"http://{ESP_IP}:81/stream"

HOST = os.environ.get("TCC_HOST", "0.0.0.0")
PORT = int(os.environ.get("TCC_PORT", "8765"))

ROI_LADO = 320                     # igual ao dataset
INFERENCIA_INTERVALO = 0.30       # segundos
DETECCAO_INTERVALO = 0.35        # segundos
FRAMES_POR_FACE = 4              # leituras de IA, não augmentation
SUAVIZACAO = 4
PREVIEW_LARGURA = 1050
PREVIEW_FPS = 8
TIMEOUT_FRAME = 3.0

# Detector pressupõe peça iluminada sobre bancada preta fosca.
MARGEM_DETECCAO_X = 0.13
MARGEM_DETECCAO_Y = 0.12
AREA_MINIMA_RELATIVA = 0.0015
AREA_MAXIMA_RELATIVA = 0.22
LADO_MINIMO_OBJETO = 35
LADO_MAXIMO_OBJETO = int(ROI_LADO * 0.98)

COLUNAS_CSV = [
    "horario", "modelo", "melhor_epoca", "limiar", "resultado",
    "frente_resultado", "verso_resultado", "prob_frente_nc",
    "prob_verso_nc", "prob_peca_nc", "leituras_frente",
    "leituras_verso", "duracao_total_s", "roi_modo",
    "roi_frente", "roi_verso",
]


# ============================================================
# CAMERA: UMA CONEXÃO PARA A ESP32
# ============================================================
class Camera:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.fps = 0.0
        self.last_time = 0.0
        self.connected = False
        self.stop_evt = threading.Event()
        self.reconnect_evt = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)

    def request_reconnect(self):
        # Solicita que a thread proprietaria do VideoCapture o
        # feche e abra novamente apos mudanca de qualidade/tamanho.
        self.reconnect_evt.set()
        with self.lock:
            self.connected = False
            self.frame = None
            self.last_time = 0.0

    def _open_capture(self):
        # Limites de leitura impedem que um MJPEG interrompido
        # prenda a thread indefinidamente no OpenCV/FFmpeg.
        try:
            cap = cv2.VideoCapture(
                STREAM_URL,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4500,
                 cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3500],
            )
            if cap.isOpened():
                return cap
            cap.release()
        except (cv2.error, TypeError, AttributeError):
            pass
        return cv2.VideoCapture(STREAM_URL)

    def start(self):
        self.worker.start()

    def stop(self):
        self.stop_evt.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2)

    def _run(self):
        while not self.stop_evt.is_set():
            self.reconnect_evt.clear()
            cap = self._open_capture()
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                cap.release()
                with self.lock:
                    self.connected = False
                self.stop_evt.wait(1.3)
                continue

            count, tick = 0, time.monotonic()
            while not self.stop_evt.is_set() and not self.reconnect_evt.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                now = time.monotonic()
                with self.lock:
                    self.frame = frame
                    self.seq += 1
                    self.last_time = now
                    self.connected = True
                count += 1
                if now - tick >= 1.5:
                    with self.lock:
                        self.fps = count / (now - tick)
                    count, tick = 0, now
            cap.release()
            with self.lock:
                self.connected = False
                self.fps = 0
            # Ajustes solicitados pela UI reiniciam mais rapido.
            was_requested = self.reconnect_evt.is_set()
            self.reconnect_evt.clear()
            self.stop_evt.wait(0.2 if was_requested else 0.8)

    def latest(self, copy=True):
        with self.lock:
            frame = self.frame.copy() if (copy and self.frame is not None) else self.frame
            return frame, self.seq, self.fps, self.last_time


# ============================================================
# DETECÇÃO GEOMÉTRICA (não usa a EfficientNet)
# ============================================================
def localizar_peca(frame):
    """Encontra contorno metálico em fundo escuro; None se incerto.

    Retorna bbox em pixels originais e motivo de falha.
    Método heurístico, deve ser calibrado na bancada real.
    """
    h, w = frame.shape[:2]
    x0, x1 = int(w * MARGEM_DETECCAO_X), int(w * (1 - MARGEM_DETECCAO_X))
    y0, y1 = int(h * MARGEM_DETECCAO_Y), int(h * (1 - MARGEM_DETECCAO_Y))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    inner = gray[y0:y1, x0:x1]
    if inner.size == 0:
        return None, "Imagem insuficiente"

    threshold, mask_inner = cv2.threshold(
        inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Rejeita imagens inteiras escuras ou com separação tonal muito baixa.
    if threshold < 38 or np.percentile(inner, 99) - np.median(inner) < 45:
        return None, "Sem contraste para localizar a peça"

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = mask_inner
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        rel = area / (w * h)
        if not (AREA_MINIMA_RELATIVA <= rel <= AREA_MAXIMA_RELATIVA):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if min(bw, bh) < LADO_MINIMO_OBJETO:
            continue
        if bw > LADO_MAXIMO_OBJETO or bh > LADO_MAXIMO_OBJETO:
            continue
        if x <= x0 + 2 or x + bw >= x1 - 2 or y <= y0 + 2 or y + bh >= y1 - 2:
            continue
        ratio = bw / max(bh, 1)
        if not (0.45 <= ratio <= 2.2):
            continue
        candidates.append((area, (x, y, bw, bh)))

    if not candidates:
        return None, "Peça não localizada; use ROI manual ou ajuste iluminação"

    _, bbox = max(candidates, key=lambda item: item[0])
    return bbox, None


def limitar_roi(x, y, w, h):
    return (max(0, min(int(x), w - ROI_LADO)),
            max(0, min(int(y), h - ROI_LADO)))


# ============================================================
# MODELOS: Ef-B0, dois rótulos, preprocess do checkpoint
# ============================================================
def listar_modelos():
    MODELOS.mkdir(parents=True, exist_ok=True)
    def key(path):
        hit = re.fullmatch(r"treino_(\d+)\.pt", path.name, re.I)
        return (1, int(hit.group(1)), path.name.lower()) if hit else (0, 0, path.name.lower())
    return sorted(MODELOS.glob("*.pt"), key=key)


def abrir_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        # Checkpoints antigos conhecidos podem conter torch.TorchVersion.
        if "torch.torch_version.TorchVersion" not in str(error):
            raise
        from torch.torch_version import TorchVersion
        with torch.serialization.safe_globals([TorchVersion]):
            return torch.load(path, map_location="cpu", weights_only=True)


class Classificador:
    def __init__(self, path, device):
        ckpt = abrir_checkpoint(path)
        if ckpt.get("class_names") != ["conforme", "nao_conforme"]:
            raise ValueError("Ordem de classes incompatível")
        if ckpt.get("architecture", "efficientnet_b0") != "efficientnet_b0":
            raise ValueError("Este painel requer EfficientNet-B0 convencional")
        if ckpt.get("resize_interpolation", "bicubic") != "bicubic":
            raise ValueError("Interpolação incompatível com o treinamento")
        network = efficientnet_b0(weights=None)
        features = network.classifier[1].in_features
        network.classifier[1] = nn.Linear(features, 2)
        network.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.network = network.to(device).eval()
        self.device = device
        size = int(ckpt.get("image_size", 224))
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(ckpt.get("mean", [.485, .456, .406]),
                                 ckpt.get("std", [.229, .224, .225])),
        ])
        self.name = path.name
        self.epoch = ckpt.get("best_epoch", "—")
        self.threshold = float(ckpt.get("threshold", .5))
        if not 0 < self.threshold < 1:
            raise ValueError("Limiar inválido")
        self.mode = str(ckpt.get("training_mode", "imagem"))

    @torch.inference_mode()
    def infer(self, roi):
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        prob = float(self.network(tensor).softmax(dim=1)[0, 1].item())
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return prob, (time.perf_counter() - start) * 1000


# ============================================================
# MOTOR DE INSPEÇÃO
# ============================================================
class Motor:
    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.RLock()
        self.stop_evt = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.status_worker = threading.Thread(target=self._poll_esp, daemon=True)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(min(os.cpu_count() or 4, 4))
        self.model = None
        self.last_error = ""
        self.roi_mode = "auto"
        self.roi_xy = None
        self.bbox = None
        self.detected = False
        self.detect_msg = "Aguardando imagem"
        self.fail_detect = 0
        self.success_detect = 0
        self.previous_candidate = None
        self.last_detect = 0.0
        self.last_infer = 0.0
        self.phase = "continuous"
        self.history = deque(maxlen=SUAVIZACAO)
        self.samples = {"front": [], "back": []}
        self.face_roi = {"front": None, "back": None}
        self.start_time = None
        self.last_prob = None
        self.smooth_prob = None
        self.infer_ms = 0.0
        self.face_mean = {"front": None, "back": None}
        self.piece_prob = None
        self.inspections = deque(maxlen=12)
        if LOG.exists():
            try:
                with LOG.open("r", newline="", encoding="utf-8-sig") as archive:
                    entries = list(csv.DictReader(archive, delimiter=";"))
                self.inspections.extend(reversed(entries[-12:]))
            except (OSError, csv.Error):
                pass
        self.status_esp = {}
        self.last_status_poll = 0
        self.neo = {"mode": "dual", "color1": "FF0000", "color2": "FA9696", "brightness": 50}
        self.last_frame_seq = -1
        self.restore_roi()

    def start(self):
        self.worker.start()
        self.status_worker.start()

    def stop(self):
        self.stop_evt.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2)
        if self.status_worker.is_alive():
            self.status_worker.join(timeout=2)

    def _poll_esp(self):
        while not self.stop_evt.is_set():
            try:
                r = requests.get(ESP_HTTP + "/status", timeout=1.6)
                r.raise_for_status()
                status = r.json()
                with self.lock:
                    self.status_esp = status
                    if "neo_mode" in status:
                        self.neo = {
                            "mode": str(status.get("neo_mode", "dual")),
                            "color1": str(status.get("color1", "FF0000")),
                            "color2": str(status.get("color2", "FA9696")),
                            "brightness": int(status.get("neo_brightness", 50)),
                        }
            except (requests.RequestException, ValueError, TypeError):
                pass
            self.stop_evt.wait(3.5)

    def restore_roi(self):
        try:
            data = json.loads(ROI_FILE.read_text(encoding="utf-8"))
            self.roi_xy = (int(data["x"]), int(data["y"]))
        except (OSError, KeyError, ValueError, TypeError):
            pass

    def save_roi(self, w, h):
        if self.roi_xy is not None:
            x, y = self.roi_xy
            ROI_FILE.write_text(json.dumps({
                "x": x, "y": y, "w": ROI_LADO, "h": ROI_LADO,
                "frame_width": w, "frame_height": h,
            }, indent=2), encoding="utf-8")

    def reset(self):
        self.phase = "continuous"
        self.samples = {"front": [], "back": []}
        self.face_mean = {"front": None, "back": None}
        self.face_roi = {"front": None, "back": None}
        self.piece_prob = None
        self.start_time = None
        self.history.clear()
        self.smooth_prob = None
        self.last_prob = None
        self.last_infer = 0.0

    def set_model(self, name):
        names = {p.name: p for p in listar_modelos()}
        if name not in names:
            raise ValueError("Modelo não encontrado")
        # Constrói novo modelo fora do lock; o atual segue em funcionamento.
        loaded = Classificador(names[name], self.device)
        with self.lock:
            self.model = loaded
            self.last_error = ""
            self.reset()
        return loaded.name

    def set_roi(self, mode, x=None, y=None):
        frame, _, _, _ = self.camera.latest(copy=False)
        with self.lock:
            if mode not in ("auto", "manual"):
                raise ValueError("Modo de ROI inválido")
            if self.phase in ("front", "back"):
                raise ValueError("Aguarde o término da face em inspeção")
            self.roi_mode = mode
            if frame is not None:
                h, w = frame.shape[:2]
                if x is not None and y is not None:
                    self.roi_xy = limitar_roi(x, y, w, h)
                if self.roi_xy is None:
                    self.roi_xy = limitar_roi((w - ROI_LADO)//2, (h - ROI_LADO)//2, w, h)
                if mode == "manual":
                    self.save_roi(w, h)
            self.reset()

    def inspection(self, action):
        with self.lock:
            if action == "reset" or action == "continuous":
                self.reset()
                return
            if self.model is None:
                raise ValueError("Selecione um modelo antes de inspecionar")
            if self.roi_mode == "auto" and not self.detected:
                raise ValueError("Peça não localizada: reposicione ou use ROI manual")
            if action == "front":
                self.reset()
                self.phase = "front"
                self.start_time = time.monotonic()
                self.face_roi["front"] = self.roi_xy
                return
            if action == "back" and self.phase == "wait_flip":
                self.phase = "back"
                self.face_roi["back"] = self.roi_xy
                self.last_infer = 0
                return
            raise ValueError("Ação indisponível nesta etapa")

    def _update_detector(self, frame, now):
        bbox, reason = localizar_peca(frame)
        h, w = frame.shape[:2]
        with self.lock:
            if bbox is None:
                self.fail_detect += 1
                if self.fail_detect >= 2:
                    self.detected = False
                    self.success_detect = 0
                    self.previous_candidate = None
                    self.bbox = None
                    self.detect_msg = reason or "Não localizada"
                    if self.roi_mode == "auto":
                        self.history.clear()
                        self.smooth_prob = None
                return

            self.fail_detect = 0
            candidate = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            if self.previous_candidate is not None and (
                abs(candidate[0] - self.previous_candidate[0]) < 65
                and abs(candidate[1] - self.previous_candidate[1]) < 65
            ):
                self.success_detect += 1
            else:
                self.success_detect = 1
            self.previous_candidate = candidate
            self.bbox = list(map(int, bbox))
            self.detected = self.success_detect >= 2
            self.detect_msg = "Peça localizada" if self.detected else "Confirmando localização..."
            if self.roi_mode == "auto":
                x, y, bw, bh = bbox
                next_xy = limitar_roi(x + bw/2 - ROI_LADO/2,
                                     y + bh/2 - ROI_LADO/2, w, h)
                if self.roi_xy is None:
                    self.roi_xy = next_xy
                else:
                    prev_x, prev_y = self.roi_xy
                    if abs(prev_x - next_xy[0]) > 8 or abs(prev_y - next_xy[1]) > 8:
                        self.history.clear()
                        self.smooth_prob = None
                    # Suaviza deslocamentos do quadrado sem modificar tamanho.
                    self.roi_xy = limitar_roi(
                        .55 * prev_x + .45 * next_xy[0],
                        .55 * prev_y + .45 * next_xy[1], w, h
                    )

    def _run(self):
        while not self.stop_evt.is_set():
            now = time.monotonic()
            frame, seq, _, last_frame = self.camera.latest()
            if frame is None or now - last_frame > TIMEOUT_FRAME or seq == self.last_frame_seq:
                self.stop_evt.wait(.07)
                continue
            self.last_frame_seq = seq
            h, w = frame.shape[:2]
            if w < ROI_LADO or h < ROI_LADO:
                with self.lock:
                    self.detected = False
                    self.detect_msg = "Resolução menor que 320x320"
                self.stop_evt.wait(.08)
                continue
            if self.roi_xy is None:
                with self.lock:
                    self.roi_xy = limitar_roi((w - ROI_LADO)//2, (h - ROI_LADO)//2, w, h)
            if now - self.last_detect >= DETECCAO_INTERVALO:
                self.last_detect = now
                self._update_detector(frame, now)
            if now - self.last_infer >= INFERENCIA_INTERVALO:
                with self.lock:
                    ai, mode, det, xy = self.model, self.roi_mode, self.detected, self.roi_xy
                    phase = self.phase
                if ai is not None and xy is not None and (mode == "manual" or det) and phase in ("continuous", "front", "back"):
                    x, y = limitar_roi(*xy, w, h)
                    crop = frame[y:y+ROI_LADO, x:x+ROI_LADO]
                    if crop.shape[:2] == (ROI_LADO, ROI_LADO):
                        try:
                            p, duration = ai.infer(crop)
                        except Exception as exc:
                            with self.lock:
                                self.last_error = f"Falha da IA: {exc}"
                        else:
                            with self.lock:
                                # Evita misturar previsões após troca de modelo/mode.
                                if ai is self.model and phase == self.phase:
                                    self.last_prob = p
                                    self.infer_ms = round(duration, 1)
                                    self.history.append(p)
                                    self.smooth_prob = sum(self.history)/len(self.history)
                                    if phase in ("front", "back"):
                                        self.samples[phase].append(p)
                                        if len(self.samples[phase]) >= FRAMES_POR_FACE:
                                            self.face_mean[phase] = sum(self.samples[phase])/len(self.samples[phase])
                                            if phase == "front":
                                                self.phase = "wait_flip"
                                            else:
                                                self.piece_prob = max(self.face_mean["front"], self.face_mean["back"])
                                                self.phase = "done"
                                                self._log_result_locked()
                self.last_infer = time.monotonic()
            self.stop_evt.wait(.03)

    def _log_result_locked(self):
        ai = self.model
        if ai is None:
            return
        p_front = self.face_mean["front"]
        p_back = self.face_mean["back"]
        p_piece = self.piece_prob
        elapsed = round(time.monotonic() - self.start_time, 2)
        row = {
            "horario": datetime.now().isoformat(timespec="seconds"),
            "modelo": ai.name,
            "melhor_epoca": ai.epoch,
            "limiar": ai.threshold,
            "resultado": "NAO CONFORME" if p_piece >= ai.threshold else "CONFORME",
            "frente_resultado": "NAO CONFORME" if p_front >= ai.threshold else "CONFORME",
            "verso_resultado": "NAO CONFORME" if p_back >= ai.threshold else "CONFORME",
            "prob_frente_nc": round(p_front, 6),
            "prob_verso_nc": round(p_back, 6),
            "prob_peca_nc": round(p_piece, 6),
            "leituras_frente": len(self.samples["front"]),
            "leituras_verso": len(self.samples["back"]),
            "duracao_total_s": elapsed,
            "roi_modo": self.roi_mode,
            "roi_frente": str(self.face_roi["front"]),
            "roi_verso": str(self.face_roi["back"]),
        }
        is_new = not LOG.exists() or LOG.stat().st_size == 0
        try:
            with LOG.open("a", newline="", encoding="utf-8-sig" if is_new else "utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUNAS_CSV, delimiter=";")
                if is_new:
                    writer.writeheader()
                writer.writerow(row)
            self.inspections.appendleft(row)
        except OSError as exc:
            self.last_error = f"Falha ao salvar CSV: {exc}"

    def snapshot(self):
        frame, seq, fps, last_frame = self.camera.latest(copy=False)
        now = time.monotonic()
        if frame is not None:
            h, w = frame.shape[:2]
        else:
            w, h = 0, 0
        connected = frame is not None and now - last_frame < TIMEOUT_FRAME
        with self.lock:
            ai = self.model
            t = ai.threshold if ai else .5
            p = self.smooth_prob if len(self.history) == SUAVIZACAO else None
            detection_ok = self.detected if self.roi_mode == "auto" else True
            return {
                "camera": {"connected": connected, "width": w, "height": h,
                           "fps": round(fps, 1), "age": round(now-last_frame, 1) if last_frame else None,
                           "esp": dict(self.status_esp)},
                "roi": {"mode": self.roi_mode, "x": self.roi_xy[0] if self.roi_xy else 0,
                        "y": self.roi_xy[1] if self.roi_xy else 0, "w": ROI_LADO, "h": ROI_LADO,
                        "detected": bool(detection_ok), "bbox": self.bbox,
                        "message": self.detect_msg},
                "ai": {"loaded": ai is not None, "model": ai.name if ai else None,
                       "epoch": ai.epoch if ai else None, "threshold": t,
                       "device": str(self.device), "mode": ai.mode if ai else None,
                       "instant": self.last_prob,
                       "average": p if detection_ok and connected else None,
                       "latency_ms": self.infer_ms},
                "inspection": {
                    "phase": self.phase,
                    "front_n": len(self.samples["front"]), "back_n": len(self.samples["back"]),
                    "target_n": FRAMES_POR_FACE, "front_p": self.face_mean["front"],
                    "back_p": self.face_mean["back"], "piece_p": self.piece_prob,
                    "elapsed_s": round(now-self.start_time, 1) if self.start_time else None,
                },
                "neo": dict(self.neo),
                "last_error": self.last_error,
                "recent": list(self.inspections)[:8],
            }


# ============================================================
# APP HTTP
# ============================================================
camera = Camera()
motor = Motor(camera)


@asynccontextmanager
async def lifespan(app):
    camera.start()
    motor.start()
    paths = listar_modelos()
    if paths:
        try:
            motor.set_model(paths[-1].name)
        except Exception as exc:
            motor.last_error = f"Não foi possível carregar {paths[-1].name}: {exc}"
    yield
    motor.stop()
    camera.stop()


app = FastAPI(title="Sistema de Inspeção Visual Industrial", lifespan=lifespan)
# Necessário quando a mesma interface HTML é servida pela ESP32, na LAN.
# Não use a API exposta à Internet; os comandos alteram hardware.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(?:localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?::\d{1,5})?$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ModeloEscolha(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class ROIAtualizacao(BaseModel):
    mode: Literal["auto", "manual"]
    x: int | None = None
    y: int | None = None


class InspecaoAcao(BaseModel):
    action: Literal["front", "back", "reset", "continuous"]


class CameraControle(BaseModel):
    name: Literal["framesize", "quality", "brightness", "contrast", "saturation", "hmirror", "vflip"]
    value: str | int


class NeoControle(BaseModel):
    mode: Literal["off", "solid", "dual", "chase"]
    color1: str = "FF0000"
    color2: str = "FA9696"
    # Suporte a versoes antigas da pagina que usam o campo color.
    color: str | None = None
    brightness: int = Field(ge=0, le=255)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/api/ping")
def ping():
    return {"ok": True, "service": "TCC-Painel", "esp_ip": ESP_IP}


@app.get("/api/status")
def status():
    return motor.snapshot()


@app.get("/api/models")
def models():
    return {"models": [p.name for p in listar_modelos()],
            "active": motor.model.name if motor.model else None}


@app.post("/api/models/select")
def model_select(body: ModeloEscolha):
    try:
        name = motor.set_model(body.name)
        return {"ok": True, "model": name}
    except (ValueError, RuntimeError, OSError, pickle.UnpicklingError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/roi")
def roi_set(body: ROIAtualizacao):
    try:
        motor.set_roi(body.mode, body.x, body.y)
        return {"ok": True, "roi": motor.snapshot()["roi"]}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/inspection")
def inspection_set(body: InspecaoAcao):
    try:
        motor.inspection(body.action)
        return {"ok": True, "inspection": motor.snapshot()["inspection"]}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def get_esp(path, params):
    try:
        response = requests.get(ESP_HTTP + path, params=params, timeout=3.0)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"ESP32 inacessível: {exc}") from exc


@app.post("/api/camera/reconnect")
def camera_reconnect():
    """Reabre a conexao MJPEG do Python sem alterar a camera."""
    camera.request_reconnect()
    with motor.lock:
        motor.reset()
        motor.detected = False
        motor.bbox = None
        motor.detect_msg = "Reconectando stream da ESP32"
    return {"ok": True, "message": "Reconexao solicitada"}


@app.post("/api/camera/control")
def camera_control(body: CameraControle):
    valid = {
        "framesize": {"VGA", "SVGA", "XGA", "SXGA", "UXGA"},
        "quality": range(4, 64),
        "brightness": range(-2, 3),
        "contrast": range(-2, 3),
        "saturation": range(-2, 3),
        "hmirror": {0, 1},
        "vflip": {0, 1},
    }
    value = body.value
    if body.name == "framesize":
        if not isinstance(value, str) or value not in valid[body.name]:
            raise HTTPException(status_code=400, detail="Resolução inválida")
    else:
        try:
            # Rejeita strings que não representam inteiros exatamente.
            value = int(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Valor inválido")
        if value not in valid[body.name]:
            raise HTTPException(status_code=400, detail="Valor fora do intervalo")
    result = get_esp("/control", {"var": body.name, "val": str(value)})
    # O sensor pode descartar frames quando muda o JPEG.
    # A thread de camera deve abrir um MJPEG novo, sem
    # exigir que o usuario reinicie o servidor Python.
    if body.name in ("quality", "framesize"):
        camera.request_reconnect()
    with motor.lock:
        motor.reset()
        motor.detected = False
        motor.bbox = None
        motor.fail_detect = 0
        motor.detect_msg = "Ajuste de câmera: aguardando atualização"
    return {"ok": True, "response": result}


@app.post("/api/neo/control")
def neo_control(body: NeoControle):
    color1 = (body.color if body.color is not None else body.color1).lstrip("#").upper()
    color2 = body.color2.lstrip("#").upper()
    if not all(re.fullmatch(r"[0-9A-F]{6}", c) for c in (color1, color2)):
        raise HTTPException(status_code=400, detail="Cor hexadecimal inválida")
    result = get_esp("/neo", {
        "mode": body.mode,
        "color1": color1,
        "color2": color2,
        "brightness": body.brightness,
    })
    with motor.lock:
        motor.neo = {
            "mode": body.mode, "color1": color1,
            "color2": color2, "brightness": body.brightness,
        }
        motor.reset()
    return {"ok": True, "response": result}


@app.get("/api/snapshot")
def snapshot_image():
    frame, _, _, timestamp = camera.latest()
    if frame is None or time.monotonic() - timestamp > TIMEOUT_FRAME:
        raise HTTPException(status_code=503, detail="Câmera sem imagem recente")
    ok, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao gerar JPEG")
    return Response(bytes(payload), media_type="image/jpeg",
                    headers={"Content-Disposition": "attachment; filename=inspecao_tcc.jpg"})


@app.get("/api/logs")
def logs():
    if not LOG.exists():
        return {"records": []}
    with LOG.open("r", newline="", encoding="utf-8-sig") as file:
        return {"records": list(csv.DictReader(file, delimiter=";"))[-15:][::-1]}


@app.get("/api/logs/download")
def download_logs():
    if not LOG.exists():
        raise HTTPException(status_code=404, detail="Nenhuma inspeção registrada")
    return FileResponse(LOG, media_type="text/csv", filename="resultados_tcc.csv")


@app.get("/api/stream")
def stream(req: Request):
    def packets():
        prev = -1
        interval = 1 / PREVIEW_FPS
        while True:
            frame, seq, _, t = camera.latest()
            if frame is not None and seq != prev and time.monotonic() - t < TIMEOUT_FRAME:
                prev = seq
                h, w = frame.shape[:2]
                if w > PREVIEW_LARGURA:
                    scale = PREVIEW_LARGURA / w
                    frame = cv2.resize(frame, (PREVIEW_LARGURA, round(h * scale)),
                                       interpolation=cv2.INTER_AREA)
                ok, img = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(img)).encode() + b"\r\n\r\n" + img.tobytes() + b"\r\n"
            time.sleep(interval)
    return StreamingResponse(packets(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    print("\nPainel TCC: http://localhost:%s" % PORT)
    print("ESP32: %s" % ESP_HTTP)
    print("Os modelos ficam em: %s" % MODELOS)
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
