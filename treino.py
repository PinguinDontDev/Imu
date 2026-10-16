import serial
import time
import threading
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rede import AttentionCNN

PORTA = "COM3"          # Sua porta COM
BAUD = 115200
JANELA = 128
ARQUIVO_PESOS = "pesos_imu.pth"

buffer_imu = deque([np.zeros(6) for _ in range(JANELA)], maxlen=JANELA)
rodando = True

# Estado da gravação interativa: None (Pausado), 0 (Gravando Parado), 1 (Gravando Movimento)
modo_gravacao = None
X_dados, Y_dados = [], []
cont_parado, cont_movimento = 0, 0

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

def executar_treinamento():
    global modo_gravacao
    modo_gravacao = None

    if cont_parado < 10 or cont_movimento < 10:
        print("\n[AVISO] Colete pelo menos 10 amostras de cada classe ('0' e '1') antes de treinar!")
        return

    print(f"\nIniciando treinamento com {cont_parado} amostras PARADO e {cont_movimento} MOVIMENTO...")
    modelo = AttentionCNN()
    x_tensor = torch.tensor(np.array(X_dados), dtype=torch.float32).transpose(1, 2)
    y_tensor = torch.tensor(np.array(Y_dados), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.003)
    criterio = nn.BCEWithLogitsLoss()

    modelo.train()
    for epoca in range(50):
        perda_total = 0.0
        for bx, by in loader:
            otimizador.zero_grad()
            perda = criterio(modelo(bx), by)
            perda.backward()
            otimizador.step()
            perda_total += perda.item()
        if (epoca + 1) % 10 == 0:
            print(f" -> Época {epoca+1}/50 | Loss: {perda_total/len(loader):.4f}")

    torch.save(modelo.state_dict(), ARQUIVO_PESOS)
    print(f"[SUCESSO] Modelo treinado e salvo em '{ARQUIVO_PESOS}'! Você já pode fechar a janela.")

def ao_pressionar_tecla(event):
    global modo_gravacao
    if event.key == '0':
        modo_gravacao = 0
    elif event.key == '1':
        modo_gravacao = 1
    elif event.key == ' ':
        modo_gravacao = None
    elif event.key in ['t', 'T']:
        executar_treinamento()

if __name__ == "__main__":
    ser = serial.Serial(PORTA, BAUD, timeout=0.1)
    time.sleep(1.5)
    ser.reset_input_buffer()
    threading.Thread(target=thread_serial, args=(ser,), daemon=True).start()

    plt.style.use('dark_background')
    fig, (ax_acc, ax_gyr) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.canvas.manager.set_window_title("TREINADOR INTERATIVO - Use as teclas [0], [1], [Espaço] e [T]")
    fig.canvas.mpl_connect('key_press_event', ao_pressionar_tecla)
    x_eixo = np.arange(JANELA)

    l_ax, = ax_acc.plot(x_eixo, np.zeros(JANELA), label="Acc X", color="#00d4ff")
    l_ay, = ax_acc.plot(x_eixo, np.zeros(JANELA), label="Acc Y", color="#00ff88")
    l_az, = ax_acc.plot(x_eixo, np.zeros(JANELA), label="Acc Z", color="#ff3366")
    ax_acc.set_ylim(-4.0, 4.0)
    ax_acc.set_ylabel("Accel Dinâmica (m/s²)")
    ax_acc.legend(loc="upper left", ncol=3)
    ax_acc.grid(True, alpha=0.2)
    titulo = ax_acc.set_title("Aguardando comando...", fontsize=11, fontweight="bold")

    l_gx, = ax_gyr.plot(x_eixo, np.zeros(JANELA), label="Gyro X", color="#00d4ff")
    l_gy, = ax_gyr.plot(x_eixo, np.zeros(JANELA), label="Gyro Y", color="#00ff88")
    l_gz, = ax_gyr.plot(x_eixo, np.zeros(JANELA), label="Gyro Z", color="#ffcc00")
    ax_gyr.set_ylim(-2.0, 2.0)
    ax_gyr.set_ylabel("Gyro (rad/s)")
    ax_gyr.set_xlabel("Teclas: [0] Gravar PARADO | [1] Gravar MOVIMENTO | [Espaço] Pausar | [T] Treinar e Salvar", fontsize=10, color="#ffcc00")
    ax_gyr.legend(loc="upper left", ncol=3)
    ax_gyr.grid(True, alpha=0.2)

    def atualizar(frame):
        global cont_parado, cont_movimento
        dados = obter_janela_limpa()

        l_ax.set_ydata(dados[:, 0])
        l_ay.set_ydata(dados[:, 1])
        l_az.set_ydata(dados[:, 2])
        l_gx.set_ydata(dados[:, 3])
        l_gy.set_ydata(dados[:, 4])
        l_gz.set_ydata(dados[:, 5])

        # Se a gravação estiver ativa, salva a janela atual a cada frame
        if modo_gravacao == 0:
            X_dados.append(dados)
            Y_dados.append([0.0])
            cont_parado += 1
            titulo.set_text(f"🔴 GRAVANDO [0: PARADO] | Amostras Parado: {cont_parado} | Movimento: {cont_movimento}")
            titulo.set_color("#00ff88")
        elif modo_gravacao == 1:
            X_dados.append(dados)
            Y_dados.append([1.0])
            cont_movimento += 1
            titulo.set_text(f"🔴 GRAVANDO [1: MOVIMENTO] | Amostras Parado: {cont_parado} | Movimento: {cont_movimento}")
            titulo.set_color("#ff3366")
        else:
            titulo.set_text(f"⏸ PAUSADO (Pressione 0, 1 ou T) | Amostras Parado: {cont_parado} | Movimento: {cont_movimento}")
            titulo.set_color("white")

        return l_ax, l_ay, l_az, l_gx, l_gy, l_gz, titulo

    ani = FuncAnimation(fig, atualizar, interval=40, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    rodando = False
    ser.close()