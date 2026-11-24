
"""
TCC - Treinamento EfficientNet-B0 com rotulos por face.

Cada peca possui:
    coleta.json com rotulo_frente e rotulo_verso
    frente_01.png ... frente_04.png
    verso_01.png ... verso_04.png

Treino: quatro imagens de cada face.
Validacao/teste: somente frente_01.png e verso_01.png.
Divisao: por peca fisica, nunca por fotografia.

A cada execucao:
    relatorios/treino_N/
    modelos/treino_N.pt

"""

import csv
import hashlib
import json
import os
import random
import time

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


# ============================================================
# PARAMETROS DO EXPERIMENTO
# ============================================================

SEMENTE = 40

EPOCAS_CABECALHO = 15
EPOCAS_AJUSTE = 50

BATCH_SIZE = 8

LR_CABECALHO = 0.0005
LR_AJUSTE = 0.00001

WEIGHT_DECAY = 0.001
DROPOUT = 0.30

PACIENCIA_EARLY_STOPPING = 10
PACIENCIA_SCHEDULER = 3

USAR_PESOS_DE_CLASSE = False
ROTACAO_EXTRA_GRAUS = 0

LIMIAR = 0.50


# ============================================================
# CAMINHOS E CONSTANTES
# ============================================================

BASE = Path(__file__).resolve().parent

DATASET = BASE / "dataset"
RELATORIOS = BASE / "relatorios"
MODELOS = BASE / "modelos"

CLASSES = [
    "conforme",
    "nao_conforme",
]

MEDIA = [
    0.485,
    0.456,
    0.406,
]

DESVIO = [
    0.229,
    0.224,
    0.225,
]

TAMANHO_ENTRADA = 224
TAMANHO_COLETA = (320, 320)

FOTOS_POR_FACE = 4


# ============================================================
# SALVAR JSON
# ============================================================

def salvar_json(caminho, dados):

    caminho.write_text(
        json.dumps(
            dados,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# SALVAR CSV
# ============================================================

def salvar_csv(caminho, linhas, colunas=None):

    if colunas is None:

        colunas = (
            list(linhas[0])
            if linhas
            else []
        )

    with caminho.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as arquivo:

        writer = csv.DictWriter(
            arquivo,
            fieldnames=colunas,
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(linhas)


# ============================================================
# COMPARATIVO GERAL DOS TREINAMENTOS
# ============================================================

def atualizar_comparativo(registro):

    caminho = (
        RELATORIOS
        / "comparativo_treinamentos.csv"
    )

    anteriores = []
    colunas = []

    if caminho.exists():

        with caminho.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as arquivo:

            leitor = csv.DictReader(
                arquivo,
                delimiter=";",
            )

            anteriores = list(leitor)

            colunas = list(
                leitor.fieldnames or []
            )

    anteriores = [
        r
        for r in anteriores
        if r.get("treino") != registro["treino"]
    ]

    colunas = list(
        dict.fromkeys(
            colunas + list(registro)
        )
    )

    salvar_csv(
        caminho,
        anteriores + [registro],
        colunas,
    )


# ============================================================
# LOCALIZAR E VALIDAR PECAS
# ============================================================

def carregar_pecas():

    pecas = []

    for classe in CLASSES:

        raiz = DATASET / classe

        if not raiz.is_dir():

            raise RuntimeError(
                f"Pasta ausente: {raiz}"
            )

        for pasta in sorted(raiz.iterdir()):

            if not pasta.is_dir():
                continue

            meta_path = pasta / "coleta.json"

            if not meta_path.is_file():

                raise RuntimeError(
                    f"coleta.json ausente: {pasta}"
                )

            meta = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

            rotulos = {
                "frente": meta.get("rotulo_frente"),
                "verso": meta.get("rotulo_verso"),
            }

            if any(
                rotulo not in CLASSES
                for rotulo in rotulos.values()
            ):

                raise RuntimeError(
                    f"Rotulo de face invalido: {pasta}"
                )

            # -----------------------------------------------
            # VALIDAR CLASSE DA PECA
            # -----------------------------------------------

            classe_calculada = (
                "nao_conforme"
                if "nao_conforme" in rotulos.values()
                else "conforme"
            )

            if (
                classe_calculada != classe
                or meta.get("classe_peca") != classe
            ):

                raise RuntimeError(
                    "Pasta e rotulos contraditorios: "
                    f"{pasta}"
                )

            # -----------------------------------------------
            # VALIDAR FOTOGRAFIAS
            # -----------------------------------------------

            fotos = {}

            for face in ("frente", "verso"):

                caminhos = [
                    pasta / f"{face}_{i:02d}.png"
                    for i in range(
                        1,
                        FOTOS_POR_FACE + 1
                    )
                ]

                for caminho in caminhos:

                    if not caminho.is_file():

                        raise RuntimeError(
                            f"Imagem ausente: {caminho}"
                        )

                    with Image.open(caminho) as img:

                        if img.size != TAMANHO_COLETA:

                            raise RuntimeError(
                                "Imagem diferente de 320x320: "
                                f"{caminho}"
                            )

                fotos[face] = caminhos

            pecas.append({

                "id": (
                    f"{classe}/{pasta.name}"
                ),

                "classe_peca": classe,

                "rotulos": rotulos,

                "fotos": fotos,
            })

    # --------------------------------------------------------
    # VERIFICAR QUANTIDADE DE PECAS
    # --------------------------------------------------------

    for classe in CLASSES:

        quantidade = sum(
            p["classe_peca"] == classe
            for p in pecas
        )

        if quantidade < 6:

            raise RuntimeError(
                f"Poucas pecas da classe "
                f"{classe}: {quantidade}"
            )

    return pecas


# ============================================================
# DIVISAO POR PECA FISICA
#
# Reutiliza a divisao fixa do mesmo dataset.
#
# Nenhuma peca aparece em dois conjuntos.
# ============================================================

def dividir_pecas(pecas):

    RELATORIOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapa = {
        p["id"]: p
        for p in pecas
    }

    assinatura = hashlib.sha256(
        "\n".join(
            sorted(mapa)
        ).encode("utf-8")
    ).hexdigest()[:12]

    manifesto = (
        RELATORIOS
        / f"divisao_{assinatura}.json"
    )

    # --------------------------------------------------------
    # REUTILIZAR DIVISAO EXISTENTE
    # --------------------------------------------------------

    if manifesto.exists():

        grupos = json.loads(
            manifesto.read_text(
                encoding="utf-8"
            )
        )["divisao"]

        print(
            "Reutilizando divisao:",
            manifesto.name
        )

    else:

        rng = random.Random(SEMENTE)

        grupos = {
            "treino": [],
            "validacao": [],
            "teste": [],
        }

        for classe in CLASSES:

            itens = sorted(
                (
                    p
                    for p in pecas
                    if p["classe_peca"] == classe
                ),
                key=lambda p: p["id"],
            )

            rng.shuffle(itens)

            n = len(itens)

            n_treino = min(
                max(1, round(n * 0.70)),
                n - 2,
            )

            resto = n - n_treino

            if classe == "conforme":

                n_val = (
                    resto + 1
                ) // 2

            else:

                n_val = (
                    resto // 2
                )

            n_val = min(
                max(1, n_val),
                resto - 1,
            )

            grupos["treino"].extend(
                p["id"]
                for p in itens[:n_treino]
            )

            grupos["validacao"].extend(
                p["id"]
                for p in itens[
                    n_treino:n_treino + n_val
                ]
            )

            grupos["teste"].extend(
                p["id"]
                for p in itens[
                    n_treino + n_val:
                ]
            )

        for nome in grupos:

            rng.shuffle(
                grupos[nome]
            )

        salvar_json(
            manifesto,
            {
                "assinatura": assinatura,
                "semente": SEMENTE,
                "divisao": grupos,
            },
        )

        print(
            "Nova divisao fixa criada:",
            manifesto.name
        )

    # --------------------------------------------------------
    # VERIFICAR VAZAMENTO DE DADOS
    # --------------------------------------------------------

    nomes = (
        "treino",
        "validacao",
        "teste",
    )

    ids = [
        identificador
        for nome in nomes
        for identificador in grupos[nome]
    ]

    if (
        len(ids) != len(set(ids))
        or set(ids) != set(mapa)
    ):

        raise RuntimeError(
            "Divisao inconsistente: "
            "pecas duplicadas ou ausentes."
        )

    for nome in nomes:

        classes = {
            mapa[identificador]["classe_peca"]
            for identificador in grupos[nome]
        }

        if classes != set(CLASSES):

            raise RuntimeError(
                f"Conjunto {nome} sem as "
                "duas classes de pecas."
            )

    splits = {
        nome: [
            mapa[identificador]
            for identificador in grupos[nome]
        ]
        for nome in nomes
    }

    return splits, assinatura


# ============================================================
# CRIAR PASTA E MODELO NUMERADOS
# ============================================================

def criar_experimento():

    RELATORIOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    for numero in range(1, 100000):

        nome = f"treino_{numero}"

        pasta = (
            RELATORIOS / nome
        )

        caminho_modelo = (
            MODELOS / f"{nome}.pt"
        )

        if (
            pasta.exists()
            or caminho_modelo.exists()
        ):
            continue

        pasta.mkdir(
            exist_ok=False
        )

        return (
            pasta,
            caminho_modelo,
        )

    raise RuntimeError(
        "Nao foi possivel numerar um novo treino."
    )


# ============================================================
# DATASET PYTORCH
#
# TREINO:
#   quatro imagens por face
#
# VALIDACAO E TESTE:
#   somente imagem original *_01.png
# ============================================================

class DatasetFaces(Dataset):

    def __init__(self, pecas, treino):

        self.registros = []

        passos = [
            transforms.Resize(
                (
                    TAMANHO_ENTRADA,
                    TAMANHO_ENTRADA,
                ),
                interpolation=(
                    transforms.InterpolationMode.BICUBIC
                ),
            )
        ]

        # ----------------------------------------------------
        # ROTACAO EXTRA OPCIONAL
        # ----------------------------------------------------

        if (
            treino
            and ROTACAO_EXTRA_GRAUS > 0
        ):

            passos.append(
                transforms.RandomRotation(
                    ROTACAO_EXTRA_GRAUS,
                    fill=(0, 0, 0),
                )
            )

        # ----------------------------------------------------
        # NORMALIZACAO IMAGENET
        # ----------------------------------------------------

        passos.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                MEDIA,
                DESVIO,
            ),
        ])

        self.transform = transforms.Compose(
            passos
        )

        # ----------------------------------------------------
        # REGISTROS DAS FOTOGRAFIAS
        # ----------------------------------------------------

        for peca in pecas:

            for face in ("frente", "verso"):

                rotulo = CLASSES.index(
                    peca["rotulos"][face]
                )

                if treino:

                    caminhos = (
                        peca["fotos"][face]
                    )

                else:

                    caminhos = (
                        peca["fotos"][face][:1]
                    )

                for caminho in caminhos:

                    self.registros.append((
                        caminho,
                        rotulo,
                        peca["id"],
                        face,
                    ))

    def __len__(self):

        return len(
            self.registros
        )

    def __getitem__(self, indice):

        (
            caminho,
            rotulo,
            peca,
            face,
        ) = self.registros[indice]

        with Image.open(caminho) as imagem:

            tensor = self.transform(
                imagem.convert("RGB")
            )

        return (
            tensor,
            rotulo,
            peca,
            face,
            caminho.name,
        )


# ============================================================
# METRICAS
#
# Classe positiva = nao_conforme.
# ============================================================

def metricas_binarias(reais, previstos):

    tp = sum(
        a == 1 and b == 1
        for a, b in zip(
            reais,
            previstos
        )
    )

    tn = sum(
        a == 0 and b == 0
        for a, b in zip(
            reais,
            previstos
        )
    )

    fp = sum(
        a == 0 and b == 1
        for a, b in zip(
            reais,
            previstos
        )
    )

    fn = sum(
        a == 1 and b == 0
        for a, b in zip(
            reais,
            previstos
        )
    )

    precisao = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    especificidade = (
        tn / (tn + fp)
        if tn + fp
        else 0.0
    )

    f1 = (
        2 * precisao * recall
        / (precisao + recall)
        if precisao + recall
        else 0.0
    )

    return {
        "n": len(reais),

        "acuracia": (
            (tp + tn) / len(reais)
            if reais
            else 0.0
        ),

        "precisao_nc": precisao,
        "recall_nc": recall,
        "especificidade": especificidade,
        "f1_nc": f1,

        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,

        "matriz": [
            [tn, fp],
            [fn, tp],
        ],
    }


# ============================================================
# AVALIACAO DO MODELO
# ============================================================

@torch.inference_mode()
def avaliar(modelo, loader, device):

    modelo.eval()

    linhas = []

    soma_perda = 0.0
    n_total = 0

    for (
        imagens,
        rotulos,
        pecas,
        faces,
        arquivos,
    ) in loader:

        imagens = imagens.to(
            device
        )

        rotulos = rotulos.to(
            device
        )

        logits = modelo(
            imagens
        )

        soma_perda += (
            nn.functional.cross_entropy(
                logits,
                rotulos,
                reduction="sum",
            ).item()
        )

        probabilidades = (
            logits.softmax(
                dim=1
            )[:, 1]
            .cpu()
            .tolist()
        )

        for (
            peca,
            face,
            arquivo,
            real,
            prob,
        ) in zip(
            pecas,
            faces,
            arquivos,
            rotulos.cpu().tolist(),
            probabilidades,
        ):

            linhas.append({
                "peca": peca,
                "face": face,
                "arquivo": arquivo,
                "rotulo": int(real),
                "prob_nao_conforme": float(prob),
            })

            n_total += 1

    return {
        "perda": (
            soma_perda / max(n_total, 1)
        ),
        "linhas": linhas,
    }


# ============================================================
# RESUMIR METRICAS
#
# Imagens, frente, verso e peca fisica.
# ============================================================

def resumir_avaliacao(
    avaliacao,
    limiar=LIMIAR,
):

    linhas = avaliacao["linhas"]

    reais = [
        r["rotulo"]
        for r in linhas
    ]

    previstos = [
        int(
            r["prob_nao_conforme"] >= limiar
        )
        for r in linhas
    ]

    saida = {
        "perda": avaliacao["perda"]
    }

    # --------------------------------------------------------
    # METRICAS POR FACE
    # --------------------------------------------------------

    for face in ("frente", "verso"):

        pares = [
            (r, p)
            for r, p, linha in zip(
                reais,
                previstos,
                linhas
            )
            if linha["face"] == face
        ]

        saida[face] = metricas_binarias(
            [a for a, _ in pares],
            [b for _, b in pares],
        )

    saida["imagens"] = metricas_binarias(
        reais,
        previstos,
    )

    # --------------------------------------------------------
    # AGRUPAR POR PECA
    # --------------------------------------------------------

    grupos = defaultdict(dict)

    predicoes_imagens = []

    for linha, previsto in zip(
        linhas,
        previstos,
    ):

        grupos[
            linha["peca"]
        ][linha["face"]] = linha

        predicoes_imagens.append({
            "peca": linha["peca"],
            "face": linha["face"],
            "arquivo": linha["arquivo"],
            "real": CLASSES[linha["rotulo"]],

            "prob_nao_conforme": round(
                linha["prob_nao_conforme"],
                6,
            ),

            "previsto": CLASSES[previsto],

            "acertou": int(
                previsto == linha["rotulo"]
            ),
        })

    # --------------------------------------------------------
    # RESULTADO POR PECA
    #
    # Se qualquer face for nao conforme,
    # a peca inteira sera nao conforme.
    # --------------------------------------------------------

    reais_pecas = []
    previstos_pecas = []
    predicoes_pecas = []

    for chave, faces in sorted(
        grupos.items()
    ):

        if set(faces) != {
            "frente",
            "verso",
        }:

            raise RuntimeError(
                "Faces ausentes da peca "
                f"{chave} na avaliacao."
            )

        f = faces["frente"]
        v = faces["verso"]

        real = max(
            f["rotulo"],
            v["rotulo"],
        )

        prob = max(
            f["prob_nao_conforme"],
            v["prob_nao_conforme"],
        )

        previsto = int(
            prob >= limiar
        )

        reais_pecas.append(
            real
        )

        previstos_pecas.append(
            previsto
        )

        predicoes_pecas.append({
            "peca": chave,
            "real": CLASSES[real],

            "prob_frente": round(
                f["prob_nao_conforme"],
                6,
            ),

            "prob_verso": round(
                v["prob_nao_conforme"],
                6,
            ),

            "prob_peca": round(
                prob,
                6,
            ),

            "previsto": CLASSES[previsto],

            "acertou": int(
                real == previsto
            ),
        })

    saida["pecas"] = metricas_binarias(
        reais_pecas,
        previstos_pecas,
    )

    saida["predicoes_imagens"] = (
        predicoes_imagens
    )

    saida["predicoes_pecas"] = (
        predicoes_pecas
    )

    return saida


# ============================================================
# TREINAR UMA EPOCA
# ============================================================

def treinar_epoca(
    modelo,
    loader,
    criterio,
    otimizador,
    device,
):

    modelo.train()

    # Preserva estatisticas BatchNorm
    # durante fine-tuning com poucas pecas.

    for modulo in modelo.modules():

        if isinstance(
            modulo,
            nn.BatchNorm2d
        ):

            modulo.eval()

    perda_total = 0.0
    total = 0
    acertos = 0

    for (
        imagens,
        rotulos,
        _,
        _,
        _,
    ) in loader:

        imagens = imagens.to(
            device
        )

        rotulos = rotulos.to(
            device
        )

        otimizador.zero_grad(
            set_to_none=True
        )

        logits = modelo(
            imagens
        )

        perda = criterio(
            logits,
            rotulos
        )

        perda.backward()
        otimizador.step()

        n = rotulos.size(0)

        perda_total += (
            perda.item() * n
        )

        acertos += (
            (
                logits.argmax(dim=1)
                == rotulos
            )
            .sum()
            .item()
        )

        total += n

    return (
        perda_total / total,
        acertos / total,
    )


# ============================================================
# CONFIGURAR OTIMIZADOR E FASE
# ============================================================

def configurar_otimizador(
    modelo,
    fase,
):

    # --------------------------------------------------------
    # CONGELAR EXTRATOR
    # --------------------------------------------------------

    for parametro in (
        modelo.features.parameters()
    ):

        parametro.requires_grad = False

    # --------------------------------------------------------
    # LIBERAR ULTIMOS BLOCOS NO AJUSTE FINO
    # --------------------------------------------------------

    if fase == "ajuste":

        for parametro in (
            modelo.features[-2:].parameters()
        ):

            parametro.requires_grad = True

    # --------------------------------------------------------
    # CLASSIFICADOR SEMPRE TREINAVEL
    # --------------------------------------------------------

    for parametro in (
        modelo.classifier.parameters()
    ):

        parametro.requires_grad = True

    lr = (
        LR_CABECALHO
        if fase == "cabecalho"
        else LR_AJUSTE
    )

    otimizador = torch.optim.AdamW(
        (
            p
            for p in modelo.parameters()
            if p.requires_grad
        ),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            otimizador,
            mode="min",
            factor=0.5,
            patience=PACIENCIA_SCHEDULER,
        )
    )

    return (
        otimizador,
        scheduler,
    )


# ============================================================
# SALVAR CHECKPOINT
#
# Somente tensores e tipos Python nativos.
# Compativel com torch.load(weights_only=True).
# ============================================================

def salvar_checkpoint(
    modelo,
    destino,
    epoca,
    val_loss,
    parametros,
    assinatura,
):

    checkpoint = {

        "model_state_dict": {
            chave: tensor.detach().cpu().clone()
            for chave, tensor in (
                modelo.state_dict().items()
            )
        },

        "architecture": "efficientnet_b0",

        "class_names": CLASSES,

        "image_size": TAMANHO_ENTRADA,

        "mean": MEDIA,

        "std": DESVIO,

        "resize_interpolation": "bicubic",

        "threshold": float(LIMIAR),

        "best_epoch": int(epoca),

        "val_loss": float(val_loss),

        "training_mode": (
            "faces_rotuladas_individualmente"
        ),

        "piece_aggregation": (
            "max(prob_frente,prob_verso)"
        ),

        "dataset_signature": str(
            assinatura
        ),

        "parameters": parametros,
    }

    torch.save(
        checkpoint,
        destino
    )


# ============================================================
# GERAR GRAFICOS
# ============================================================

def gerar_graficos(
    pasta,
    historico,
    teste,
):

    epocas = [
        x["epoca"]
        for x in historico
    ]

    # --------------------------------------------------------
    # CURVAS
    # --------------------------------------------------------

    fig, eixos = plt.subplots(
        1,
        2,
        figsize=(12, 4),
    )

    eixos[0].plot(
        epocas,
        [
            x["loss_treino"]
            for x in historico
        ],
        label="Treino",
    )

    eixos[0].plot(
        epocas,
        [
            x["loss_validacao"]
            for x in historico
        ],
        label="Validacao",
    )

    eixos[0].set(
        title="Perda",
        xlabel="Epoca",
    )

    eixos[0].legend()

    for chave, nome in (
        ("acc_treino", "Treino"),
        ("acc_val_frente", "Validacao frente"),
        ("acc_val_verso", "Validacao verso"),
    ):

        eixos[1].plot(
            epocas,
            [
                x[chave]
                for x in historico
            ],
            label=nome,
        )

    eixos[1].set(
        title="Acuracia",
        xlabel="Epoca",
        ylim=(0, 1.05),
    )

    eixos[1].legend()

    fig.tight_layout()

    fig.savefig(
        pasta / "curvas.png",
        dpi=160,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # MATRIZES DE CONFUSAO
    # --------------------------------------------------------

    fig, eixos = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    for ax, nome in zip(
        eixos,
        (
            "frente",
            "verso",
            "pecas",
        ),
    ):

        matriz = (
            teste[nome]["matriz"]
        )

        ax.imshow(
            matriz,
            cmap="Blues",
            vmin=0,
        )

        ax.set(
            title=nome,
            xlabel="Previsto",
            ylabel="Real",
        )

        ax.set_xticks(
            [0, 1],
            CLASSES,
        )

        ax.set_yticks(
            [0, 1],
            CLASSES,
        )

        for i in (0, 1):

            for j in (0, 1):

                ax.text(
                    j,
                    i,
                    str(matriz[i][j]),
                    ha="center",
                    va="center",
                )

    fig.tight_layout()

    fig.savefig(
        pasta / "matrizes_confusao.png",
        dpi=160,
    )

    plt.close(fig)


# ============================================================
# TREINAMENTO PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # REPRODUTIBILIDADE
    # --------------------------------------------------------

    random.seed(
        SEMENTE
    )

    torch.manual_seed(
        SEMENTE
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEMENTE
        )

    # --------------------------------------------------------
    # DISPOSITIVO
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cpu":

        torch.set_num_threads(
            min(
                os.cpu_count() or 4,
                6,
            )
        )

    # --------------------------------------------------------
    # LER DATASET
    # --------------------------------------------------------

    pecas = carregar_pecas()

    RELATORIOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits, assinatura = (
        dividir_pecas(pecas)
    )

    # --------------------------------------------------------
    # NOVO EXPERIMENTO
    # --------------------------------------------------------

    pasta, caminho_modelo = (
        criar_experimento()
    )

    # --------------------------------------------------------
    # PARAMETROS
    # --------------------------------------------------------

    parametros = {

        "data": datetime.now().isoformat(
            timespec="seconds"
        ),

        "semente": SEMENTE,

        "epocas_cabecalho": EPOCAS_CABECALHO,

        "epocas_ajuste": EPOCAS_AJUSTE,

        "batch_size": BATCH_SIZE,

        "lr_cabecalho": LR_CABECALHO,

        "lr_ajuste": LR_AJUSTE,

        "weight_decay": WEIGHT_DECAY,

        "dropout": DROPOUT,

        "paciencia_early_stopping": (
            PACIENCIA_EARLY_STOPPING
        ),

        "paciencia_scheduler": (
            PACIENCIA_SCHEDULER
        ),

        "usar_pesos_de_classe": (
            USAR_PESOS_DE_CLASSE
        ),

        "rotacao_extra_graus": (
            ROTACAO_EXTRA_GRAUS
        ),

        "limiar": LIMIAR,

        "n_fotos_por_face": (
            FOTOS_POR_FACE
        ),

        "validacao_teste": (
            "somente *_01.png"
        ),

        "dataset_signature": assinatura,

        # Converte TorchVersion para string simples.
        "torch_version": str(
            torch.__version__
        ),

        "device": str(device),
    }

    salvar_json(
        pasta / "parametros.json",
        parametros,
    )

    # --------------------------------------------------------
    # SALVAR DIVISAO DAS PECAS
    # --------------------------------------------------------

    salvar_csv(
        pasta / "divisao_pecas.csv",
        [

            {

                "conjunto": conjunto,

                "peca": p["id"],

                "classe_peca": (
                    p["classe_peca"]
                ),

                "rotulo_frente": (
                    p["rotulos"]["frente"]
                ),

                "rotulo_verso": (
                    p["rotulos"]["verso"]
                ),
            }

            for conjunto, lista in splits.items()

            for p in lista
        ],
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    datasets = {

        nome: DatasetFaces(
            lista,
            treino=(nome == "treino"),
        )

        for nome, lista in splits.items()
    }

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    loaders = {

        nome: DataLoader(

            dataset,

            batch_size=BATCH_SIZE,

            shuffle=(nome == "treino"),

            num_workers=0,

            pin_memory=(
                device.type == "cuda"
            ),
        )

        for nome, dataset in datasets.items()
    }

    # --------------------------------------------------------
    # CONTAR CLASSES VISUAIS DO TREINO
    # --------------------------------------------------------

    contagens = [

        sum(
            registro[1] == classe
            for registro in (
                datasets["treino"].registros
            )
        )

        for classe in (0, 1)
    ]

    if min(contagens) == 0:

        raise RuntimeError(
            "O treino nao contem as "
            "duas classes visuais."
        )

    # --------------------------------------------------------
    # PESOS OPCIONAIS DE CLASSE
    # --------------------------------------------------------

    pesos = None

    if USAR_PESOS_DE_CLASSE:

        total = sum(
            contagens
        )

        pesos = torch.tensor(
            [
                total / (2 * n)
                for n in contagens
            ],
            dtype=torch.float32,
            device=device,
        )

    criterio = nn.CrossEntropyLoss(
        weight=pesos
    )

    # --------------------------------------------------------
    # INFORMACOES DO EXPERIMENTO
    # --------------------------------------------------------

    print("=" * 60)
    print(
        "TREINAMENTO EFFICIENTNET-B0 "
        "- ROTULOS POR FACE"
    )
    print("=" * 60)

    print(
        "Experimento:",
        pasta.name
    )

    print(
        "Modelo:",
        caminho_modelo
    )

    print(
        "Dispositivo:",
        device
    )

    for nome in (
        "treino",
        "validacao",
        "teste",
    ):

        print(
            f"{nome}: "
            f"{len(splits[nome])} pecas; "
            f"{len(datasets[nome])} imagens"
        )

    print(
        "Rotulos do treino "
        "[conforme, nao conforme]:",
        contagens
    )

    print(
        "Validacao e teste: "
        "somente imagens originais *_01.png"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # EFFICIENTNET-B0 PRE-TREINADA
    # --------------------------------------------------------

    modelo = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    entradas = (
        modelo.classifier[1].in_features
    )

    modelo.classifier[0].p = DROPOUT

    modelo.classifier[1] = nn.Linear(
        entradas,
        2,
    )

    modelo.to(
        device
    )

    # --------------------------------------------------------
    # PRIMEIRA FASE
    # --------------------------------------------------------

    otimizador, scheduler = (
        configurar_otimizador(
            modelo,
            "cabecalho",
        )
    )

    melhor_perda = float("inf")

    historico = []

    sem_melhora = 0
    melhor_epoca = 0

    total_epocas = (
        EPOCAS_CABECALHO
        + EPOCAS_AJUSTE
    )

    # --------------------------------------------------------
    # LOOP DE TREINAMENTO
    # --------------------------------------------------------

    for epoca in range(
        1,
        total_epocas + 1
    ):

        # ----------------------------------------------------
        # INICIAR FINE-TUNING
        # ----------------------------------------------------

        if epoca == (
            EPOCAS_CABECALHO + 1
        ):

            print()
            print(
                "--- INICIANDO AJUSTE FINO ---"
            )

            otimizador, scheduler = (
                configurar_otimizador(
                    modelo,
                    "ajuste",
                )
            )

            sem_melhora = 0

        fase = (
            "cabecalho"
            if epoca <= EPOCAS_CABECALHO
            else "ajuste"
        )

        inicio = time.perf_counter()

        # ----------------------------------------------------
        # TREINAMENTO
        # ----------------------------------------------------

        perda_treino, acc_treino = (
            treinar_epoca(
                modelo,
                loaders["treino"],
                criterio,
                otimizador,
                device,
            )
        )

        # ----------------------------------------------------
        # VALIDACAO
        # ----------------------------------------------------

        val_raw = avaliar(
            modelo,
            loaders["validacao"],
            device,
        )

        val = resumir_avaliacao(
            val_raw
        )

        scheduler.step(
            val_raw["perda"]
        )

        # ----------------------------------------------------
        # SALVAR MELHOR MODELO
        # ----------------------------------------------------

        marcador = ""

        if val_raw["perda"] < (
            melhor_perda - 0.0001
        ):

            melhor_perda = (
                val_raw["perda"]
            )

            melhor_epoca = epoca

            sem_melhora = 0

            salvar_checkpoint(
                modelo,
                caminho_modelo,
                epoca,
                melhor_perda,
                parametros,
                assinatura,
            )

            marcador = " [SALVO]"

        else:

            sem_melhora += 1

        # ----------------------------------------------------
        # HISTORICO DA EPOCA
        # ----------------------------------------------------

        historico.append({

            "epoca": epoca,

            "fase": fase,

            "loss_treino": round(
                perda_treino,
                6,
            ),

            "acc_treino": round(
                acc_treino,
                6,
            ),

            "loss_validacao": round(
                val_raw["perda"],
                6,
            ),

            "acc_val_frente": round(
                val["frente"]["acuracia"],
                6,
            ),

            "acc_val_verso": round(
                val["verso"]["acuracia"],
                6,
            ),

            "acc_val_pecas": round(
                val["pecas"]["acuracia"],
                6,
            ),

            "recall_nc_val": round(
                val["imagens"]["recall_nc"],
                6,
            ),

            "fp_val": val["imagens"]["FP"],

            "fn_val": val["imagens"]["FN"],

            "lr": (
                otimizador.param_groups[0]["lr"]
            ),
        })

        salvar_csv(
            pasta / "historico.csv",
            historico,
        )

        # ----------------------------------------------------
        # EXIBIR EPOCA
        # ----------------------------------------------------

        duracao = (
            time.perf_counter()
            - inicio
        )

        print(
            f"Ep {epoca:02d}/{total_epocas} "
            f"{fase:9s} "
            f"train={perda_treino:.4f} "
            f"val={val_raw['perda']:.4f} "
            f"frente="
            f"{val['frente']['acuracia']:.1%} "
            f"verso="
            f"{val['verso']['acuracia']:.1%} "
            f"recall_NC="
            f"{val['imagens']['recall_nc']:.1%} "
            f"tempo={duracao:.1f}s"
            f"{marcador}"
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            fase == "ajuste"
            and sem_melhora >= PACIENCIA_EARLY_STOPPING
        ):

            print(
                "Early stopping: "
                "validacao sem melhora."
            )

            break

    # ========================================================
    # RECARREGAR MELHOR MODELO
    #
    # Checkpoint sem objetos TorchVersion.
    # ========================================================

    checkpoint = torch.load(
        caminho_modelo,
        map_location="cpu",
        weights_only=True,
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    modelo.to(
        device
    )

    modelo.eval()

    # ========================================================
    # AVALIACAO FINAL
    # ========================================================

    resultados = {}

    for nome in (
        "validacao",
        "teste",
    ):

        raw = avaliar(
            modelo,
            loaders[nome],
            device,
        )

        met = resumir_avaliacao(
            raw,
            LIMIAR,
        )

        # ----------------------------------------------------
        # METRICAS
        # ----------------------------------------------------

        salvar_json(
            pasta / f"metricas_{nome}.json",
            {
                "perda": raw["perda"],

                "limiar": LIMIAR,

                **{
                    nivel: met[nivel]
                    for nivel in (
                        "imagens",
                        "frente",
                        "verso",
                        "pecas",
                    )
                },
            },
        )

        # ----------------------------------------------------
        # PREDICOES INDIVIDUAIS
        # ----------------------------------------------------

        salvar_csv(
            pasta / f"predicoes_{nome}.csv",
            met["predicoes_imagens"],
        )

        # ----------------------------------------------------
        # PREDICOES POR PECA
        # ----------------------------------------------------

        salvar_csv(
            pasta / f"predicoes_pecas_{nome}.csv",
            met["predicoes_pecas"],
        )

        resultados[nome] = met

    # ========================================================
    # GRAFICOS
    # ========================================================

    gerar_graficos(
        pasta,
        historico,
        resultados["teste"],
    )

    # ========================================================
    # RESUMO DO EXPERIMENTO
    # ========================================================

    val = resultados["validacao"]
    teste = resultados["teste"]

    resumo = {

        "treino": pasta.name,

        "dataset": assinatura,

        "modo": (
            "faces_rotuladas_individualmente"
        ),

        "modelo": str(caminho_modelo),

        "melhor_epoca": melhor_epoca,

        "val_loss": round(
            melhor_perda,
            6,
        ),

        "val_acc_frente": round(
            val["frente"]["acuracia"],
            6,
        ),

        "val_acc_verso": round(
            val["verso"]["acuracia"],
            6,
        ),

        "val_recall_nc": round(
            val["imagens"]["recall_nc"],
            6,
        ),

        "val_fp_faces": (
            val["imagens"]["FP"]
        ),

        "val_fn_faces": (
            val["imagens"]["FN"]
        ),

        "teste_acc_frente": round(
            teste["frente"]["acuracia"],
            6,
        ),

        "teste_acc_verso": round(
            teste["verso"]["acuracia"],
            6,
        ),

        "teste_acc_peca": round(
            teste["pecas"]["acuracia"],
            6,
        ),

        "teste_recall_nc": round(
            teste["imagens"]["recall_nc"],
            6,
        ),

        "teste_fp_faces": (
            teste["imagens"]["FP"]
        ),

        "teste_fn_faces": (
            teste["imagens"]["FN"]
        ),

        "lr_ajuste": LR_AJUSTE,

        "weight_decay": WEIGHT_DECAY,

        "dropout": DROPOUT,
    }

    salvar_json(
        pasta / "resumo.json",
        resumo,
    )

    atualizar_comparativo(
        resumo
    )

    # ========================================================
    # RESULTADOS NO TERMINAL
    # ========================================================

    print()
    print("=" * 60)

    print(
        f"RESULTADO DO TESTE - "
        f"{pasta.name} "
        f"(EPOCA {melhor_epoca})"
    )

    print("=" * 60)

    for nivel in (
        "imagens",
        "frente",
        "verso",
        "pecas",
    ):

        m = teste[nivel]

        print(
            f"{nivel:8s}: "
            f"acc={m['acuracia']:.1%} "
            f"recall_NC={m['recall_nc']:.1%} "
            f"F1={m['f1_nc']:.3f} "
            f"FP={m['FP']} "
            f"FN={m['FN']} "
            f"(N={m['n']})"
        )

    print()
    print(
        "Modelo salvo:",
        caminho_modelo
    )

    print(
        "Relatorios:",
        pasta
    )

    print()
    print(
        "Aviso: o conjunto de teste ainda "
        "possui poucas pecas fisicas."
    )


# ============================================================
# INICIO
# ============================================================

if __name__ == "__main__":
    main()
