"""
extract_frames.py
-----------------
Extrai um frame representativo por janela temporal de vídeos de vinhedo,
priorizando nitidez (variância do Laplaciano) e luminosidade adequada.

Uso:
    python extract_frames.py

Os vídeos devem estar na pasta raiz com os nomes video1.mp4 ... video7.mp4.
Os frames extraídos são salvos em dataset_uvas_estruturado/<nome_do_video>/.
"""

import cv2
import os
import numpy as np


# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------

PASTA_SAIDA_BASE = "dataset_uvas_estruturado"
VIDEOS = [f"video{i}.mp4" for i in range(1, 8)]
FRAMES_POR_SEGUNDO = 4       # janelas por segundo → 1 frame por janela
LIMIAR_LUMINOSIDADE = 40.0   # abaixo disso o frame é considerado escuro


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def calcular_nitidez(frame_cinza: np.ndarray) -> float:
    """
    Retorna a variância do Laplaciano do frame em escala de cinza.
    Quanto maior o valor, mais nítido é o frame.
    """
    return cv2.Laplacian(frame_cinza, cv2.CV_64F).var()


def calcular_luminosidade(frame_cinza: np.ndarray) -> float:
    """
    Retorna a luminosidade média do frame (valor entre 0 e 255).
    """
    return float(np.mean(frame_cinza))


def selecionar_melhor_frame(janela: list, limiar: float) -> tuple:
    """
    Dado uma lista de frames de uma janela temporal, seleciona o mais adequado.

    Critério principal: entre os frames com luminosidade >= limiar,
    escolhe o de maior nitidez.

    Fallback: se todos estiverem escuros, escolhe o mais nítido do lote.

    Retorna (frame_selecionado, status) onde status é 'ideal' ou 'fallback'.
    """
    frames_claros = [f for f in janela if f["luminosidade"] >= limiar]

    if frames_claros:
        melhor = max(frames_claros, key=lambda f: f["nitidez"])
        return melhor, "ideal"

    melhor = max(janela, key=lambda f: f["nitidez"])
    return melhor, "fallback"


def salvar_frame(frame: np.ndarray, caminho: str) -> None:
    """
    Salva o frame como JPEG no caminho especificado.
    """
    cv2.imwrite(caminho, frame)


# ---------------------------------------------------------------------------
# Função principal de extração
# ---------------------------------------------------------------------------

def extrair_frames(caminho_video: str, pasta_saida: str,
                   frames_por_seg: int = 4, limiar_luz: float = 40.0) -> int:
    """
    Processa um vídeo e extrai exatamente um frame por janela temporal.

    Parâmetros
    ----------
    caminho_video : str
        Caminho para o arquivo de vídeo.
    pasta_saida : str
        Pasta onde os frames serão salvos.
    frames_por_seg : int
        Número de janelas por segundo (equivale a quantos frames extrair por segundo).
    limiar_luz : float
        Valor mínimo de luminosidade média para considerar um frame bem iluminado.

    Retorna
    -------
    int
        Número de frames salvos.
    """
    os.makedirs(pasta_saida, exist_ok=True)

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"  [ERRO] Não foi possível abrir: {caminho_video}")
        return 0

    fps_original = cap.get(cv2.CAP_PROP_FPS)
    tamanho_janela = max(1, int(round(fps_original / frames_por_seg)))

    print(f"  FPS do vídeo: {fps_original:.2f} | Tamanho da janela: {tamanho_janela} frames")

    janela = []
    indice_frame = 0
    frames_salvos = 0
    fallbacks = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        janela.append({
            "imagem": frame,
            "nitidez": calcular_nitidez(frame_cinza),
            "luminosidade": calcular_luminosidade(frame_cinza),
            "indice_original": indice_frame,
        })

        if len(janela) == tamanho_janela:
            melhor, status = selecionar_melhor_frame(janela, limiar_luz)

            nome_arquivo = (
                f"frame_{frames_salvos:06d}"
                f"_orig_{melhor['indice_original']:06d}.jpg"
            )
            salvar_frame(melhor["imagem"], os.path.join(pasta_saida, nome_arquivo))

            if status == "fallback":
                fallbacks += 1

            frames_salvos += 1
            janela = []

        indice_frame += 1

    # Processa os frames restantes caso o vídeo não termine em janela completa
    if janela:
        melhor, status = selecionar_melhor_frame(janela, limiar_luz)
        nome_arquivo = (
            f"frame_{frames_salvos:06d}"
            f"_orig_{melhor['indice_original']:06d}.jpg"
        )
        salvar_frame(melhor["imagem"], os.path.join(pasta_saida, nome_arquivo))
        if status == "fallback":
            fallbacks += 1
        frames_salvos += 1

    cap.release()

    print(f"  Frames extraídos: {frames_salvos} | Fallbacks (janelas escuras): {fallbacks}")
    return frames_salvos


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    total_geral = 0

    for nome_video in VIDEOS:
        pasta_destino = os.path.join(
            PASTA_SAIDA_BASE,
            os.path.splitext(nome_video)[0]
        )
        print(f"\n=== Processando {nome_video} ===")

        if not os.path.isfile(nome_video):
            print(f"  [AVISO] Arquivo não encontrado, pulando.")
            continue

        total = extrair_frames(
            caminho_video=nome_video,
            pasta_saida=pasta_destino,
            frames_por_seg=FRAMES_POR_SEGUNDO,
            limiar_luz=LIMIAR_LUMINOSIDADE,
        )
        total_geral += total

    print(f"\n{'='*50}")
    print(f"Extração concluída. Total de frames gerados: {total_geral}")
    print(f"Saída em: {os.path.abspath(PASTA_SAIDA_BASE)}")


if __name__ == "__main__":
    main()
