"""INSPECTA / TCC - EfficientNet-B0, treino de faces e avaliacao por peca.

Dataset esperado:
  dataset/conforme/peca_.../coleta.json, frente_01..04.png, verso_01..04.png
  dataset/nao_conforme/peca_.../coleta.json, frente_01..04.png, verso_01..04.png

A cada execucao, cria relatorios/treino_N e modelos/treino_N.pt.
O checkpoint e compativel com o painel INSPECTA (EfficientNet-B0 padrao).

Avaliacao original: somente *_01.png, como nos treinos anteriores.
Avaliacao 4 fotos: todas as fotos por face, probabilidades agregadas por media.
As fotos 02/03/04 sao transformacoes e NAO sao observacoes independentes.
O conjunto de teste so e avaliado depois da escolha do melhor checkpoint.
"""

import csv
import hashlib
import json
import os
import random
import re
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
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


# ============================================================================
# CONFIGURACAO DO EXPERIMENTO
# ============================================================================

SEMENTE = 40
TAMANHO_ENTRADA = 320                # Antes: 224; arquivos reais: 320x320
TAMANHO_COLETA = (320, 320)
FOTOS_POR_FACE = 4

EPOCAS_CABECALHO = 15
EPOCAS_AJUSTE = 50
BATCH_SIZE = 8                        # Em 320 usa mais RAM; reduza para 4 se necessario
LR_CABECALHO = 0.0005
LR_AJUSTE = 0.00001
WEIGHT_DECAY = 0.001
DROPOUT = 0.30
PACIENCIA_EARLY_STOPPING = 10
PACIENCIA_SCHEDULER = 3
USAR_PESOS_DE_CLASSE = False

# Augmentation online LEVE, somente para o treino.
# Desative para comparar isoladamente 224 vs 320 com o treino anterior.
AUMENTO_LEVE = False              # Primeiro isole o efeito da resolucao; depois experimente True
BRILHO_AUMENTO = 0.08
CONTRASTE_AUMENTO = 0.08
SATURACAO_AUMENTO = 0.04
ROTACAO_EXTRA_GRAUS = 0            # Evita modificar a geometria dos defeitos

# Limiar fixo para comparar com treinamentos anteriores.
# O script faz um levantamento de limiares SOMENTE na validacao,
# mas nao muda o limiar do checkpoint automaticamente.
LIMIAR = 0.50

# Escolha do checkpoint exclusivamente pela perda da foto ORIGINAL na validacao.
# Mantem a comparabilidade e nao escolhe com base no teste nem nas fotos sinteticas.
CRITERIO_CHECKPOINT = "perda_validacao_foto_original"

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
RELATORIOS = BASE / "relatorios"
MODELOS = BASE / "modelos"
CLASSES = ["conforme", "nao_conforme"]
FACES = ("frente", "verso")
CONJUNTOS = ("treino", "validacao", "teste")
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]


# ============================================================================
# ARQUIVOS E VALIDACAO
# ============================================================================

def salvar_json(path, objeto):
    path.write_text(json.dumps(objeto, indent=2, ensure_ascii=False), encoding="utf-8")


def salvar_csv(path, linhas, colunas=None):
    colunas = colunas or (list(linhas[0].keys()) if linhas else [])
    with path.open("w", newline="", encoding="utf-8-sig") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=colunas, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(linhas)


def atualizar_comparativo(registro):
    path = RELATORIOS / "comparativo_treinamentos.csv"
    anteriores, colunas = [], []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as arquivo:
            reader = csv.DictReader(arquivo, delimiter=";")
            anteriores = list(reader)
            colunas = list(reader.fieldnames or [])
    anteriores = [r for r in anteriores if r.get("treino") != registro["treino"]]
    colunas = list(dict.fromkeys(colunas + list(registro)))
    salvar_csv(path, anteriores + [registro], colunas)


def carregar_pecas():
    pecas = []
    for classe in CLASSES:
        raiz = DATASET / classe
        if not raiz.is_dir():
            raise FileNotFoundError(f"Pasta do dataset ausente: {raiz}")

        for pasta in sorted(raiz.iterdir()):
            if not pasta.is_dir():
                continue
            meta_path = pasta / "coleta.json"
            if not meta_path.is_file():
                raise FileNotFoundError(f"Falta coleta.json em {pasta}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rotulos = {face: meta.get(f"rotulo_{face}") for face in FACES}
            if any(r not in CLASSES for r in rotulos.values()):
                raise ValueError(f"Rotulo invalido: {pasta}: {rotulos}")
            classe_calculada = "nao_conforme" if "nao_conforme" in rotulos.values() else "conforme"
            if classe_calculada != classe or meta.get("classe_peca") != classe:
                raise ValueError(f"Classe inconsistente com as faces: {pasta}")

            fotos = {}
            for face in FACES:
                fotos[face] = [pasta / f"{face}_{i:02d}.png" for i in range(1, FOTOS_POR_FACE + 1)]
                for imagem in fotos[face]:
                    if not imagem.is_file():
                        raise FileNotFoundError(f"Fotografia ausente: {imagem}")
                    with Image.open(imagem) as img:
                        if img.size != TAMANHO_COLETA:
                            raise ValueError(f"Imagem {imagem} tem {img.size}, esperado {TAMANHO_COLETA}")
                        img.verify()

            match = re.search(r"20\d{6}", pasta.name)
            sessao = (match.group(0) if match else "sem_data")
            if sessao != "sem_data":
                sessao = f"{sessao[0:4]}-{sessao[4:6]}-{sessao[6:8]}"

            pecas.append({
                "id": f"{classe}/{pasta.name}",
                "classe_peca": classe,
                "rotulos": rotulos,
                "fotos": fotos,
                "sessao": sessao,
                "meta": meta,
            })

    for classe in CLASSES:
        n = sum(p["classe_peca"] == classe for p in pecas)
        if n < 6:
            raise ValueError(f"Poucas pecas da classe {classe}: {n}")
    return pecas


def divisao_pecas(pecas):
    """Mesmo manifesto antigo por IDs para permitir comparacao com o Treino 11."""
    RELATORIOS.mkdir(parents=True, exist_ok=True)
    mapa = {p["id"]: p for p in pecas}
    assinatura = hashlib.sha256("\n".join(sorted(mapa)).encode("utf-8")).hexdigest()[:12]
    manifesto = RELATORIOS / f"divisao_{assinatura}.json"

    if manifesto.exists():
        registro = json.loads(manifesto.read_text(encoding="utf-8"))
        grupos = registro["divisao"]
        print(f"[DIVISAO] Reutilizando {manifesto.name}, semente original: {registro.get('semente', '?')}")
        print("[DIVISAO] Alterar SEMENTE nao altera esta divisao ja existente.")
    else:
        rng = random.Random(SEMENTE)
        grupos = {nome: [] for nome in CONJUNTOS}
        for classe in CLASSES:
            itens = sorted((p for p in pecas if p["classe_peca"] == classe), key=lambda p: p["id"])
            rng.shuffle(itens)
            n = len(itens)
            n_treino = min(max(1, round(n * 0.70)), n - 2)
            restante = n - n_treino
            n_val = (restante + 1) // 2 if classe == "conforme" else restante // 2
            n_val = min(max(1, n_val), restante - 1)
            grupos["treino"].extend(p["id"] for p in itens[:n_treino])
            grupos["validacao"].extend(p["id"] for p in itens[n_treino:n_treino + n_val])
            grupos["teste"].extend(p["id"] for p in itens[n_treino + n_val:])
        for nome in CONJUNTOS:
            rng.shuffle(grupos[nome])
        salvar_json(manifesto, {"assinatura": assinatura, "semente": SEMENTE, "divisao": grupos})
        print(f"[DIVISAO] Criada divisao fixa {manifesto.name}")

    todos = [identificador for nome in CONJUNTOS for identificador in grupos[nome]]
    if len(todos) != len(set(todos)) or set(todos) != set(mapa):
        raise ValueError("Divisao invalida: pecas repetidas ou faltando.")
    for nome in CONJUNTOS:
        classes = {mapa[identificador]["classe_peca"] for identificador in grupos[nome]}
        if classes != set(CLASSES):
            raise ValueError(f"Falta uma das classes no conjunto {nome}.")
    return ({nome: [mapa[i] for i in grupos[nome]] for nome in CONJUNTOS}, assinatura)


def hash_dataset(pecas):
    """Registra uma assinatura real do conteudo, nao so nomes de diretorios."""
    digest = hashlib.sha256()
    for p in sorted(pecas, key=lambda x: x["id"]):
        digest.update(p["id"].encode("utf-8"))
        digest.update(json.dumps(p["rotulos"], sort_keys=True).encode("utf-8"))
        for face in FACES:
            for path in p["fotos"][face]:
                digest.update(path.name.encode("utf-8"))
                with path.open("rb") as arquivo:
                    for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
                        digest.update(bloco)
    return digest.hexdigest()[:16]


def criar_experimento():
    RELATORIOS.mkdir(parents=True, exist_ok=True)
    MODELOS.mkdir(parents=True, exist_ok=True)
    for numero in range(1, 100000):
        nome = f"treino_{numero}"
        pasta = RELATORIOS / nome
        modelo = MODELOS / f"{nome}.pt"
        if pasta.exists() or modelo.exists():
            continue
        pasta.mkdir(exist_ok=False)
        return pasta, modelo
    raise RuntimeError("Nao foi possivel criar nova numeracao de treino.")


# ============================================================================
# DATASET E TRANSFORMACOES
# ============================================================================

class DatasetFaces(Dataset):
    def __init__(self, pecas, treinamento=False, todas_fotos=True):
        self.registros = []
        pipeline = [
            transforms.Resize(
                (TAMANHO_ENTRADA, TAMANHO_ENTRADA),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        ]
        if treinamento:
            if ROTACAO_EXTRA_GRAUS:
                pipeline.append(transforms.RandomRotation(ROTACAO_EXTRA_GRAUS, fill=(0, 0, 0)))
            if AUMENTO_LEVE:
                pipeline.append(transforms.ColorJitter(
                    brightness=BRILHO_AUMENTO,
                    contrast=CONTRASTE_AUMENTO,
                    saturation=SATURACAO_AUMENTO,
                ))
        pipeline += [transforms.ToTensor(), transforms.Normalize(MEDIA, DESVIO)]
        self.transform = transforms.Compose(pipeline)

        for peca in pecas:
            for face in FACES:
                caminhos = peca["fotos"][face] if todas_fotos else peca["fotos"][face][:1]
                for path in caminhos:
                    self.registros.append((
                        path, CLASSES.index(peca["rotulos"][face]), peca["id"], face, peca["sessao"]
                    ))

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, indice):
        path, rotulo, peca_id, face, sessao = self.registros[indice]
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, rotulo, peca_id, face, path.name, sessao


# ============================================================================
# METRICAS E AGREGACAO
# ============================================================================

def metricas_binarias(reais, previstos):
    if len(reais) != len(previstos):
        raise ValueError("Listas de rotulos com comprimentos diferentes.")
    tp = sum(a == 1 and b == 1 for a, b in zip(reais, previstos))
    tn = sum(a == 0 and b == 0 for a, b in zip(reais, previstos))
    fp = sum(a == 0 and b == 1 for a, b in zip(reais, previstos))
    fn = sum(a == 1 and b == 0 for a, b in zip(reais, previstos))
    precisao = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    espec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
    return {
        "n": len(reais), "acuracia": (tp + tn) / len(reais) if reais else 0.0,
        "precisao_nc": precisao, "recall_nc": recall, "especificidade": espec,
        "f1_nc": f1, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "matriz": [[tn, fp], [fn, tp]],
    }


@torch.inference_mode()
def avaliar(modelo, loader, device):
    """Armazena todas as previsoes; as duas estrategias usam a MESMA inferencia."""
    modelo.eval()
    linhas = []
    for imagens, rotulos, ids, faces, arquivos, sessoes in loader:
        imagens = imagens.to(device)
        rotulos = rotulos.to(device)
        logits = modelo(imagens)
        probs = logits.softmax(dim=1)[:, 1].cpu().tolist()
        losses = nn.functional.cross_entropy(logits, rotulos, reduction="none").cpu().tolist()
        for id_peca, face, arquivo, sessao, real, prob, perda in zip(
            ids, faces, arquivos, sessoes, rotulos.cpu().tolist(), probs, losses
        ):
            linhas.append({
                "peca": id_peca, "face": face, "arquivo": arquivo, "sessao": sessao,
                "rotulo": int(real), "prob_nao_conforme": float(prob), "perda": float(perda),
            })
    return linhas


def agregar_previsoes(linhas, usar_4_fotos, limiar=LIMIAR):
    """Retorna metricas por imagem, face fisica e peca fisica.

    4 fotos = media de probabilidades por face; peca = max(media_frente, media_verso).
    Original = usa *_01.png, tanto por face quanto por peca.
    """
    if not usar_4_fotos:
        linhas = [r for r in linhas if r["arquivo"].endswith("_01.png")]

    linhas = sorted(linhas, key=lambda x: (x["peca"], x["face"], x["arquivo"]))
    if not linhas:
        raise ValueError("Nenhuma imagem disponivel para avaliacao.")

    imagens_csv = []
    for r in linhas:
        previsto = int(r["prob_nao_conforme"] >= limiar)
        imagens_csv.append({
            "peca": r["peca"], "sessao": r["sessao"], "face": r["face"],
            "arquivo": r["arquivo"], "real": CLASSES[r["rotulo"]],
            "prob_nao_conforme": round(r["prob_nao_conforme"], 6),
            "previsto": CLASSES[previsto], "acertou": int(previsto == r["rotulo"]),
        })

    m_imagens = metricas_binarias(
        [r["rotulo"] for r in linhas],
        [int(r["prob_nao_conforme"] >= limiar) for r in linhas],
    )
    por_face = defaultdict(list)
    for r in linhas:
        por_face[(r["peca"], r["face"])].append(r)

    faces_csv = []
    for (id_peca, face), grupo in sorted(por_face.items()):
        if len(grupo) != (FOTOS_POR_FACE if usar_4_fotos else 1):
            raise ValueError(f"Quantidade de fotos incorreta em {id_peca}/{face}: {len(grupo)}")
        reais = {r["rotulo"] for r in grupo}
        if len(reais) != 1:
            raise ValueError(f"Rotulos inconsistentes em {id_peca}/{face}")
        probabilidades = [r["prob_nao_conforme"] for r in grupo]
        media = sum(probabilidades) / len(probabilidades)
        real = grupo[0]["rotulo"]
        previsto = int(media >= limiar)
        dispersao = max(probabilidades) - min(probabilidades)
        faces_csv.append({
            "peca": id_peca, "sessao": grupo[0]["sessao"], "face": face,
            "real": CLASSES[real], "real_codigo": real, "prob_media": media,
            "prob_min": min(probabilidades), "prob_max": max(probabilidades),
            "amplitude": dispersao,
            "predicoes_divergentes": int(min(probabilidades) < limiar <= max(probabilidades)),
            "previsto": CLASSES[previsto], "acertou": int(real == previsto),
            "n_fotos": len(probabilidades),
        })

    grupos_pecas = defaultdict(dict)
    for r in faces_csv:
        grupos_pecas[r["peca"]][r["face"]] = r
    pecas_csv = []
    for id_peca, faces in sorted(grupos_pecas.items()):
        if set(faces) != set(FACES):
            raise ValueError(f"Faltam faces da peca {id_peca}")
        frente, verso = faces["frente"], faces["verso"]
        real = max(frente["real_codigo"], verso["real_codigo"])
        prob = max(frente["prob_media"], verso["prob_media"])
        previsto = int(prob >= limiar)
        pecas_csv.append({
            "peca": id_peca, "sessao": frente["sessao"], "real": CLASSES[real],
            "real_codigo": real, "prob_frente": frente["prob_media"],
            "prob_verso": verso["prob_media"], "prob_peca": prob,
            "previsto": CLASSES[previsto], "acertou": int(real == previsto),
        })

    def met_faces(face):
        grupo = [r for r in faces_csv if r["face"] == face]
        return metricas_binarias(
            [r["real_codigo"] for r in grupo],
            [int(r["prob_media"] >= limiar) for r in grupo],
        )

    m_pecas = metricas_binarias(
        [r["real_codigo"] for r in pecas_csv],
        [int(r["prob_peca"] >= limiar) for r in pecas_csv],
    )
    perda = sum(r["perda"] for r in linhas) / len(linhas)
    return {
        "perda": perda, "imagens": m_imagens,
        "frente": met_faces("frente"), "verso": met_faces("verso"), "pecas": m_pecas,
        "predicoes_imagens": imagens_csv, "predicoes_faces": faces_csv,
        "predicoes_pecas": pecas_csv,
        "faces_com_previsoes_divergentes": sum(r["predicoes_divergentes"] for r in faces_csv),
    }


def resumir_sessoes(agregado):
    por_sessao = defaultdict(list)
    for p in agregado["predicoes_pecas"]:
        por_sessao[p["sessao"]].append(p)
    return [
        {"sessao": sessao, **metricas_binarias(
            [p["real_codigo"] for p in pecas],
            [int(p["previsto"] == "nao_conforme") for p in pecas],
        )}
        for sessao, pecas in sorted(por_sessao.items())
    ]


def tabela_limiares(agregado, metodo):
    """Apenas validacao. Nao escolhe limiar usando o teste."""
    linhas = []
    for k in range(20, 81, 5):
        limiar = k / 100
        reais = [p["real_codigo"] for p in agregado["predicoes_pecas"]]
        previstos = [int(p["prob_peca"] >= limiar) for p in agregado["predicoes_pecas"]]
        m = metricas_binarias(reais, previstos)
        linhas.append({
            "metodo": metodo, "limiar": limiar, "acuracia_pecas": m["acuracia"],
            "recall_nc": m["recall_nc"], "especificidade": m["especificidade"],
            "f1_nc": m["f1_nc"], "FP": m["FP"], "FN": m["FN"],
        })
    return linhas


# ============================================================================
# TREINO E CHECKPOINT
# ============================================================================

def configurar_otimizador(modelo, fase):
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False
    if fase == "ajuste":
        for parametro in modelo.features[-2:].parameters():
            parametro.requires_grad = True
    for parametro in modelo.classifier.parameters():
        parametro.requires_grad = True
    lr = LR_CABECALHO if fase == "cabecalho" else LR_AJUSTE
    otimizador = torch.optim.AdamW(
        (p for p in modelo.parameters() if p.requires_grad),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="min", factor=0.5, patience=PACIENCIA_SCHEDULER,
    )
    return otimizador, scheduler


def treinar_epoca(modelo, loader, criterio, otimizador, device):
    modelo.train()
    # BatchNorm nao atualiza estatisticas com batch pequeno.
    for modulo in modelo.modules():
        if isinstance(modulo, nn.BatchNorm2d):
            modulo.eval()
    total, soma_perda, acertos = 0, 0.0, 0
    for imagens, rotulos, *_ in loader:
        imagens, rotulos = imagens.to(device), rotulos.to(device)
        otimizador.zero_grad(set_to_none=True)
        logits = modelo(imagens)
        loss = criterio(logits, rotulos)
        loss.backward()
        otimizador.step()
        n = len(rotulos)
        soma_perda += loss.item() * n
        acertos += (logits.argmax(1) == rotulos).sum().item()
        total += n
    return soma_perda / total, acertos / total


def salvar_checkpoint(modelo, destino, epoca, perda_val, parametros, assinatura, conteudo_hash):
    """O mesmo formato do INSPECTA; somente tipos aceitos por weights_only=True."""
    checkpoint = {
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()},
        "architecture": "efficientnet_b0",
        "class_names": list(CLASSES),
        "image_size": int(TAMANHO_ENTRADA),
        "mean": list(MEDIA), "std": list(DESVIO),
        "resize_interpolation": "bicubic",
        "threshold": float(LIMIAR),
        "best_epoch": int(epoca),
        "val_loss": float(perda_val),
        "training_mode": "faces_rotuladas_individualmente",
        "piece_aggregation": "max(prob_frente,prob_verso)",
        "dataset_signature": str(assinatura),
        "dataset_content_hash": str(conteudo_hash),
        "validation_strategy": "foto_01_checkpoint_4_fotos_relatorio",
        "parameters": dict(parametros),
    }
    temporario = destino.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporario)
    temporario.replace(destino)


def graficos(pasta, historico, original, quatro):
    epocas = [r["epoca"] for r in historico]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for chave, titulo in (("loss_treino", "Treino"), ("loss_validacao", "Validacao original"),
                          ("loss_validacao_4fotos", "Validacao 4 fotos")):
        axs[0].plot(epocas, [r[chave] for r in historico], label=titulo)
    axs[0].set(title="Perda", xlabel="Epoca")
    axs[0].legend()
    for chave, titulo in (("acc_treino", "Treino"), ("acc_val_pecas", "Pecas original"),
                          ("acc_val_pecas_4fotos", "Pecas 4 fotos")):
        axs[1].plot(epocas, [r[chave] for r in historico], label=titulo)
    axs[1].set(title="Acuracia", xlabel="Epoca", ylim=(0, 1.05))
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(pasta / "curvas.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for i, (nome_modo, agregado) in enumerate((("Original 01", original), ("Media 4 fotos", quatro))):
        for j, face in enumerate(("frente", "verso", "pecas")):
            matriz = agregado[face]["matriz"]
            ax = axs[i, j]
            ax.imshow(matriz, cmap="Blues", vmin=0)
            ax.set(title=f"{nome_modo} - {face}", xlabel="Previsto", ylabel="Real")
            ax.set_xticks([0, 1], CLASSES)
            ax.set_yticks([0, 1], CLASSES)
            for linha in (0, 1):
                for coluna in (0, 1):
                    ax.text(coluna, linha, str(matriz[linha][coluna]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(pasta / "matrizes_confusao.png", dpi=160)
    plt.close(fig)


def guardar_resultados(pasta, nome, original, quatro):
    def metricas(agregado):
        return {"perda": agregado["perda"], "limiar": LIMIAR,
                **{k: agregado[k] for k in ("imagens", "frente", "verso", "pecas")}}
    # Estrutura antiga preservada na raiz para scripts de comparacao.
    salvar_json(pasta / f"metricas_{nome}.json", {
        **metricas(original), "avaliacao_4fotos": {
            **metricas(quatro), "faces_com_previsoes_divergentes": quatro["faces_com_previsoes_divergentes"]
        }
    })
    salvar_csv(pasta / f"predicoes_{nome}.csv", original["predicoes_imagens"])
    salvar_csv(pasta / f"predicoes_pecas_{nome}.csv", original["predicoes_pecas"])
    salvar_csv(pasta / f"predicoes_4fotos_{nome}.csv", quatro["predicoes_imagens"])
    salvar_csv(pasta / f"predicoes_faces_4fotos_{nome}.csv", quatro["predicoes_faces"])
    salvar_csv(pasta / f"predicoes_pecas_4fotos_{nome}.csv", quatro["predicoes_pecas"])
    salvar_csv(pasta / f"sessoes_{nome}.csv", [
        {"modo": modo, **sessao}
        for modo, conjunto in (("original", original), ("4fotos", quatro))
        for sessao in resumir_sessoes(conjunto)
    ])


def main():
    random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMENTE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(os.cpu_count() or 4, 6)))

    pecas = carregar_pecas()
    splits, assinatura = divisao_pecas(pecas)
    conteudo_hash = hash_dataset(pecas)
    pasta, caminho_modelo = criar_experimento()
    parametros = {
        "data": datetime.now().isoformat(timespec="seconds"),
        "semente": SEMENTE, "epocas_cabecalho": EPOCAS_CABECALHO,
        "epocas_ajuste": EPOCAS_AJUSTE, "batch_size": BATCH_SIZE,
        "lr_cabecalho": LR_CABECALHO, "lr_ajuste": LR_AJUSTE,
        "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
        "paciencia_early_stopping": PACIENCIA_EARLY_STOPPING,
        "paciencia_scheduler": PACIENCIA_SCHEDULER,
        "usar_pesos_de_classe": USAR_PESOS_DE_CLASSE,
        "rotacao_extra_graus": ROTACAO_EXTRA_GRAUS,
        "aumento_leve": AUMENTO_LEVE, "brilho_aumento": BRILHO_AUMENTO,
        "contraste_aumento": CONTRASTE_AUMENTO, "saturacao_aumento": SATURACAO_AUMENTO,
        "image_size": TAMANHO_ENTRADA, "limiar": LIMIAR,
        "fotos_por_face": FOTOS_POR_FACE,
        "validacao_teste": "foto 01 + media das 4 fotografias por face",
        "criterio_checkpoint": CRITERIO_CHECKPOINT,
        "dataset_signature": assinatura, "dataset_content_hash": conteudo_hash,
        "torch_version": str(torch.__version__), "device": str(device),
    }
    salvar_json(pasta / "parametros.json", parametros)
    salvar_csv(pasta / "divisao_pecas.csv", [
        {"conjunto": conjunto, "peca": p["id"], "classe_peca": p["classe_peca"],
         "rotulo_frente": p["rotulos"]["frente"], "rotulo_verso": p["rotulos"]["verso"],
         "sessao": p["sessao"]}
        for conjunto, lista in splits.items() for p in lista
    ])
    salvar_csv(pasta / "distribuicao_sessoes.csv", [
        {"conjunto": conjunto, "sessao": sessao, "classe_peca": classe, "n_pecas": n}
        for conjunto in CONJUNTOS
        for (sessao, classe), n in sorted(
            ((s, c), sum(p["sessao"] == s and p["classe_peca"] == c for p in splits[conjunto]))
            for s in {p["sessao"] for p in splits[conjunto]}
            for c in CLASSES
        )
    ])

    treino = DatasetFaces(splits["treino"], treinamento=True, todas_fotos=True)
    validacao = DatasetFaces(splits["validacao"], treinamento=False, todas_fotos=True)
    teste = DatasetFaces(splits["teste"], treinamento=False, todas_fotos=True)
    pin = device.type == "cuda"
    loader_treino = DataLoader(treino, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    loader_val = DataLoader(validacao, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    loader_teste = DataLoader(teste, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    contagens = [sum(r[1] == classe for r in treino.registros) for classe in (0, 1)]
    if min(contagens) == 0:
        raise ValueError("Treino nao contem duas classes visuais.")
    pesos = None
    if USAR_PESOS_DE_CLASSE:
        pesos = torch.tensor([sum(contagens) / (2 * n) for n in contagens], device=device, dtype=torch.float32)
    criterio = nn.CrossEntropyLoss(weight=pesos)

    print("=" * 65)
    print(f"INSPECTA - {pasta.name} | EfficientNet-B0 | {TAMANHO_ENTRADA}x{TAMANHO_ENTRADA}")
    print(f"Dispositivo: {device} | checkpoint: {caminho_modelo}")
    for nome, ds in (("treino", treino), ("validacao", validacao), ("teste", teste)):
        print(f"{nome}: {len(splits[nome])} pecas / {len(ds)} fotografias (4 por face)")
    print(f"Fotos por classe visual no treino: {contagens}")
    print("Validacao: 1 foto original + analise das 4 fotos (sem augmentation aleatoria)")
    print("Selecao do melhor modelo: SOMENTE loss original da validacao; teste isolado")
    print("=" * 65)

    modelo = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    entradas = modelo.classifier[1].in_features
    modelo.classifier[0].p = DROPOUT
    modelo.classifier[1] = nn.Linear(entradas, 2)
    modelo.to(device)
    otimizador, scheduler = configurar_otimizador(modelo, "cabecalho")
    melhor_perda = float("inf")
    melhor_epoca = 0
    sem_melhora = 0
    historico = []
    total_epocas = EPOCAS_CABECALHO + EPOCAS_AJUSTE

    for epoca in range(1, total_epocas + 1):
        if epoca == EPOCAS_CABECALHO + 1:
            print("\n--- AJUSTE FINO: ultimos blocos da EfficientNet ---")
            otimizador, scheduler = configurar_otimizador(modelo, "ajuste")
            sem_melhora = 0

        fase = "cabecalho" if epoca <= EPOCAS_CABECALHO else "ajuste"
        inicio = time.perf_counter()
        loss_treino, acc_treino = treinar_epoca(modelo, loader_treino, criterio, otimizador, device)
        todas_linhas = avaliar(modelo, loader_val, device)
        original = agregar_previsoes(todas_linhas, usar_4_fotos=False)
        quatro = agregar_previsoes(todas_linhas, usar_4_fotos=True)
        scheduler.step(original["perda"])

        marcador = ""
        if original["perda"] < melhor_perda - 0.0001:
            melhor_perda, melhor_epoca = original["perda"], epoca
            sem_melhora = 0
            salvar_checkpoint(modelo, caminho_modelo, epoca, melhor_perda, parametros, assinatura, conteudo_hash)
            marcador = " [SALVO]"
        else:
            sem_melhora += 1

        historico.append({
            "epoca": epoca, "fase": fase,
            "loss_treino": round(loss_treino, 6), "acc_treino": round(acc_treino, 6),
            "loss_validacao": round(original["perda"], 6),
            "loss_validacao_4fotos": round(quatro["perda"], 6),
            "acc_val_frente": round(original["frente"]["acuracia"], 6),
            "acc_val_verso": round(original["verso"]["acuracia"], 6),
            "acc_val_pecas": round(original["pecas"]["acuracia"], 6),
            "recall_nc_val": round(original["imagens"]["recall_nc"], 6),
            "fp_val": original["imagens"]["FP"], "fn_val": original["imagens"]["FN"],
            "acc_val_frente_4fotos": round(quatro["frente"]["acuracia"], 6),
            "acc_val_verso_4fotos": round(quatro["verso"]["acuracia"], 6),
            "acc_val_pecas_4fotos": round(quatro["pecas"]["acuracia"], 6),
            "recall_nc_pecas_val": round(original["pecas"]["recall_nc"], 6),
            "recall_nc_pecas_val_4fotos": round(quatro["pecas"]["recall_nc"], 6),
            "fp_pecas_val": original["pecas"]["FP"], "fn_pecas_val": original["pecas"]["FN"],
            "fp_pecas_val_4fotos": quatro["pecas"]["FP"], "fn_pecas_val_4fotos": quatro["pecas"]["FN"],
            "faces_divergentes_4fotos": quatro["faces_com_previsoes_divergentes"],
            "lr": otimizador.param_groups[0]["lr"],
        })
        salvar_csv(pasta / "historico.csv", historico)
        duracao = time.perf_counter() - inicio
        print(
            f"Ep {epoca:02d}/{total_epocas} {fase:9s} | "
            f"loss treino={loss_treino:.4f} val01={original['perda']:.4f} | "
            f"pecas val01={original['pecas']['acuracia']:.1%} "
            f"val4={quatro['pecas']['acuracia']:.1%} | "
            f"FN_pecas={original['pecas']['FN']}/{quatro['pecas']['FN']} | "
            f"{duracao:.1f}s{marcador}"
        )
        if fase == "ajuste" and sem_melhora >= PACIENCIA_EARLY_STOPPING:
            print("Early stopping: validacao sem melhora.")
            break

    # Restaurar o MELHOR modelo antes de avaliar o teste.
    checkpoint = torch.load(caminho_modelo, map_location="cpu", weights_only=True)
    modelo.load_state_dict(checkpoint["model_state_dict"], strict=True)
    modelo.to(device).eval()

    resultados = {}
    for nome, loader in (("validacao", loader_val), ("teste", loader_teste)):
        linhas = avaliar(modelo, loader, device)
        orig = agregar_previsoes(linhas, usar_4_fotos=False)
        multi = agregar_previsoes(linhas, usar_4_fotos=True)
        guardar_resultados(pasta, nome, orig, multi)
        resultados[nome] = (orig, multi)

    val_orig, val_4 = resultados["validacao"]
    teste_orig, teste_4 = resultados["teste"]
    salvar_csv(pasta / "analise_limiares_validacao.csv", (
        tabela_limiares(val_orig, "original") + tabela_limiares(val_4, "media_4fotos")
    ))
    graficos(pasta, historico, teste_orig, teste_4)

    resumo = {
        "treino": pasta.name, "dataset": assinatura, "dataset_content_hash": conteudo_hash,
        "modo": "faces_rotuladas_individualmente", "modelo": str(caminho_modelo),
        "image_size": TAMANHO_ENTRADA, "melhor_epoca": melhor_epoca,
        "val_loss": round(melhor_perda, 6), "limiar": LIMIAR,
        "val_acc_frente": val_orig["frente"]["acuracia"],
        "val_acc_verso": val_orig["verso"]["acuracia"],
        "val_acc_peca": val_orig["pecas"]["acuracia"],
        "val_acc_peca_4fotos": val_4["pecas"]["acuracia"],
        "val_recall_nc_pecas": val_orig["pecas"]["recall_nc"],
        "val_recall_nc_pecas_4fotos": val_4["pecas"]["recall_nc"],
        "teste_acc_frente": teste_orig["frente"]["acuracia"],
        "teste_acc_verso": teste_orig["verso"]["acuracia"],
        "teste_acc_peca": teste_orig["pecas"]["acuracia"],
        "teste_acc_peca_4fotos": teste_4["pecas"]["acuracia"],
        "teste_recall_nc": teste_orig["imagens"]["recall_nc"],
        "teste_recall_nc_pecas": teste_orig["pecas"]["recall_nc"],
        "teste_recall_nc_pecas_4fotos": teste_4["pecas"]["recall_nc"],
        "teste_fp_pecas": teste_orig["pecas"]["FP"],
        "teste_fn_pecas": teste_orig["pecas"]["FN"],
        "teste_fp_pecas_4fotos": teste_4["pecas"]["FP"],
        "teste_fn_pecas_4fotos": teste_4["pecas"]["FN"],
        "teste_faces_divergentes_4fotos": teste_4["faces_com_previsoes_divergentes"],
        "lr_ajuste": LR_AJUSTE, "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
        "aumento_leve": AUMENTO_LEVE,
    }
    salvar_json(pasta / "resumo.json", resumo)
    atualizar_comparativo(resumo)

    print("\n" + "=" * 75)
    print(f"RESULTADO FINAL | {pasta.name} | melhor epoca {melhor_epoca}")
    for nome, (orig, multi) in resultados.items():
        print(f"\n{nome.upper()} - {len(orig['predicoes_pecas'])} pecas fisicas")
        for metodo, met in (("FOTO 01", orig), ("MEDIA DE 4", multi)):
            p = met["pecas"]
            print(f"  {metodo:10s} | acuracia peca={p['acuracia']:.1%} | "
                  f"recall NC={p['recall_nc']:.1%} | FP={p['FP']} FN={p['FN']} | "
                  f"frente={met['frente']['acuracia']:.1%} verso={met['verso']['acuracia']:.1%}")
    print(f"\nModelo (compatível com INSPECTA): {caminho_modelo}")
    print(f"Relatorios: {pasta}")
    print("Observacao: quatro fotos da mesma face nao equivalem a quatro pecas independentes.")
    print("Observacao: fotos 02/03/04 sao transformadas; teste final precisa de pecas novas.")


if __name__ == "__main__":
    main()
