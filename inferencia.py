import serial
import time
import threading
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rede import AttentionCNN

PORTA = "COM3"          # Sua porta COM
BAUD = 115200
JANELA = 128
ARQUIVO_PESOS = "pesos_imu.pth"

buffer_imu = deque([np.zeros(6) for _ in range(JANELA)], maxlen=JANELA)
historico_prob = deque([0.0 for _ in range(JANELA)], maxlen=JANELA)
historico_thresh = deque([50.0 for _ in range(JANELA)], maxlen=JANELA)
rodando = True

# Estado atual com Histerese (False = Parado, True = Em Movimento)
em_movimento = False

def thread_serial(ser):
    while rodando:
        try:
            linha = ser.readline().decode('ascii', errors='ignore').strip()
            partes = linha.split(',')
            if len(partes) == 6:
                vals = np.array([float(x) for x in partes])
                if np.all(np.abs(vals[:3]) < 45.0) and np.all(np.abs(vals[3:]) < 15.0):
                    buffer_imu.append(vals)
        except ValueError:
            pass

def obter_janela_limpa():
    janela = np.array(buffer_imu, dtype=np.float32)
    return janela - np.mean(janela, axis=0, keepdims=True)

def calcular_threshold_dinamico(dados_janela, estado_movimento_atual):
    """
    Calcula o Threshold Dinâmico da CNN (entre 25% e 85%) baseado em:
    1. Histerese de Estado (evita oscilação em torno de 50%).
    2. Energia de Alta Frequência (Jitter): se houver trepidação rápida, 
       eleva o rigor do limiar para proteger o ZUPT contra falsos positivos.
    """
    # 1. Mede apenas o ruído de alta frequência (diferença amostra a amostra no acelerômetro)
    # Movimentos reais suaves têm derivada baixa; vibrações e batidas têm derivada alta!
    jitter_alta_freq = np.std(np.diff(dados_janela[:, :3], axis=0))
    
    # 2. Base da Histerese:
    # Se está PARADO, exige mais certeza para sair do repouso (base = 55%)
    # Se já está EM MOVIMENTO, facilita continuar em movimento (base = 35%)
    base_histerese = 35.0 if estado_movimento_atual else 55.0
    
    # 3. Penalidade adaptativa de vibração (sobe o limiar se o chassi/mão tremer rápido)
    ajuste_vibracao = np.clip(jitter_alta_freq * 40.0, 0.0, 30.0)
    
    # Suavização exponencial com o último valor do histórico para a linha não dar saltos secos
    novo_alvo = np.clip(base_histerese + ajuste_vibracao, 25.0, 85.0)
    thresh_suavizado = 0.8 * historico_thresh[-1] + 0.2 * novo_alvo
    
    return thresh_suavizado

if __name__ == "__main__":
    modelo = AttentionCNN()
    modelo.load_state_dict(torch.load(ARQUIVO_PESOS, weights_only=True))
    modelo.eval()
    print(f"[OK] Pesos carregados de '{ARQUIVO_PESOS}'!")

    ser = serial.Serial(PORTA, BAUD, timeout=0.1)
    time.sleep(1.5)
    ser.reset_input_buffer()
    threading.Thread(target=thread_serial, args=(ser,), daemon=True).start()

    plt.style.use('dark_background')
    fig, (ax_sinal, ax_cnn) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.canvas.manager.set_window_title("ATTENTION-CNN COM THRESHOLD DINÂMICO")
    x_eixo = np.arange(JANELA)

    # Gráfico 1: Sinais IMU
    l_ax, = ax_sinal.plot(x_eixo, np.zeros(JANELA), label="Acc X", color="#00d4ff")
    l_ay, = ax_sinal.plot(x_eixo, np.zeros(JANELA), label="Acc Y", color="#00ff88")
    l_az, = ax_sinal.plot(x_eixo, np.zeros(JANELA), label="Acc Z", color="#ff3366")
    l_gz, = ax_sinal.plot(x_eixo, np.zeros(JANELA), label="Gyro Z", color="#ffcc00", ls="--")
    ax_sinal.set_ylim(-4.0, 4.0)
    ax_sinal.set_ylabel("Sinal IMU (Sem Gravidade)")
    ax_sinal.legend(loc="upper left", ncol=4)
    ax_sinal.grid(True, alpha=0.2)
    titulo = ax_sinal.set_title("Inicializando...", fontsize=11, fontweight="bold")

    # Gráfico 2: Probabilidade da CNN vs. Threshold Dinâmico
    l_prob, = ax_cnn.plot(x_eixo, np.zeros(JANELA), color="#ff9900", lw=2.2, label="Probabilidade CNN (%)")
    l_thresh, = ax_cnn.plot(x_eixo, np.ones(JANELA)*50.0, color="#00ffff", lw=1.8, ls="--", label="Threshold Dinâmico Adaptativo (%)")
    ax_cnn.set_ylim(-5, 105)
    ax_cnn.set_ylabel("Probabilidade vs. Limiar (%)")
    ax_cnn.set_xlabel("Amostras no Tempo")
    ax_cnn.legend(loc="upper left")
    ax_cnn.grid(True, alpha=0.2)

    def atualizar(frame):
        global em_movimento
        dados = obter_janela_limpa()

        l_ax.set_ydata(dados[:, 0])
        l_ay.set_ydata(dados[:, 1])
        l_az.set_ydata(dados[:, 2])
        l_gz.set_ydata(dados[:, 5])

        # 1. Inferência da Attention-CNN
        with torch.no_grad():
            entrada = torch.tensor(dados, dtype=torch.float32).T.unsqueeze(0)
            prob = torch.sigmoid(modelo(entrada)).item() * 100.0

        # 2. Atualiza o Threshold Dinâmico para a janela atual
        limiar_atual = calcular_threshold_dinamico(dados, em_movimento)

        historico_prob.append(prob)
        historico_thresh.append(limiar_atual)

        l_prob.set_ydata(historico_prob)
        l_thresh.set_ydata(historico_thresh)

        # 3. Decisão baseada no cruzamento entre a CNN e o Threshold Dinâmico
        if prob >= limiar_atual:
            em_movimento = True
            titulo.set_text(f"EM MOVIMENTO | CNN: {prob:4.1f}% >= Limiar Dinâmico: {limiar_atual:4.1f}%")
            titulo.set_color("#ff3366")
        else:
            em_movimento = False
            titulo.set_text(f"ESTACIONÁRIO (ZUPT) | CNN: {prob:4.1f}% < Limiar Dinâmico: {limiar_atual:4.1f}%")
            titulo.set_color("#00ff88")

        return l_ax, l_ay, l_az, l_gz, l_prob, l_thresh, titulo

    ani = FuncAnimation(fig, atualizar, interval=40, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    rodando = False
    ser.close()