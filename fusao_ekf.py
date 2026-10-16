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

# =====================================================================
# 1. FILTRO DE KALMAN INERCIAL (ESTIMA VELOCIDADE 3D + BIAS 3D)
# =====================================================================
class InertialKalmanFilter:
    """
    Vetor de Estado x (6x1): [vx, vy, vz, bax, bay, baz]^T
    Estima a velocidade real 3D e aprende o drift/bias residual em tempo real.
    """
    def __init__(self):
        self.x = np.zeros((6, 1))       # Estado inicial: velocidade zero, bias zero
        self.P = np.eye(6) * 0.1        # Covariância inicial da incerteza

        # Ruído de Processo Q (quanto confiamos na integração física vs variação do bias)
        self.Q = np.diag([1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4])

        # Matriz de Observação H (No ZUPT, medimos diretamente a velocidade: z = H * x)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Ruído de Medida Base R (incerteza da observação de velocidade zero)
        self.R_base = np.eye(3) * 1e-3

    def predict(self, acc_dinamica, dt):
        """Integra a aceleração descontando o bias estimado pelo próprio Kalman."""
        F = np.eye(6)
        F[0, 3] = -dt
        F[1, 4] = -dt
        F[2, 5] = -dt

        acc = acc_dinamica.reshape(3, 1)
        # v_novo = v_antigo + (acc_medida - bias_estimado) * dt
        self.x[0:3] = self.x[0:3] + (acc - self.x[3:6]) * dt
        self.P = F @ self.P @ F.T + self.Q

    def update_zupt(self, prob_cnn):
        """
        Acionado quando a Attention-CNN detecta estado ESTACIONÁRIO.
        Corrige a velocidade para 0 e recalibra o bias residual do acelerômetro.
        """
        # Modula a confiança R pela probabilidade da CNN (quanto menor a prob, mais forte o freio)
        R = self.R_base * np.exp(prob_cnn * 0.05)

        # Inovação (Erro): y = z_medido (0,0,0) - v_predito
        y = np.zeros((3, 1)) - (self.H @ self.x)

        # Ganho de Kalman: K = P * H^T * (H * P * H^T + R)^-1
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Atualiza o vetor de estado (zera velocidade e ajusta bias!) e a covariância
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_speed_magnitude(self):
        """Retorna a velocidade escalar total ||v|| em m/s."""
        return float(np.linalg.norm(self.x[0:3]))


# =====================================================================
# 2. INTEGRAÇÃO EM TEMPO REAL (SERIAL + ATTENTION-CNN + KALMAN)
# =====================================================================
buffer_imu = deque([np.zeros(6) for _ in range(JANELA)], maxlen=JANELA)
hist_prob = deque([0.0 for _ in range(JANELA)], maxlen=JANELA)
hist_thresh = deque([50.0 for _ in range(JANELA)], maxlen=JANELA)

# Históricos para comparar Velocidade Sem CNN (Com Drift) vs. Com CNN+Kalman (Sem Drift)
hist_vel_kalman = deque([0.0 for _ in range(JANELA)], maxlen=JANELA)
hist_vel_sem_zupt = deque([0.0 for _ in range(JANELA)], maxlen=JANELA)

kf = InertialKalmanFilter()
vel_bruta_acumulada = np.zeros(3) # Integrador cego (para mostrar o drift explosivo)
em_movimento = False
rodando = True

def obter_janela_limpa():
    janela = np.array(buffer_imu, dtype=np.float32)
    return janela - np.mean(janela, axis=0, keepdims=True)

def calcular_threshold_dinamico(dados_janela, estado_mov_atual):
    jitter = np.std(np.diff(dados_janela[:, :3], axis=0))
    base = 35.0 if estado_mov_atual else 55.0
    alvo = np.clip(base + np.clip(jitter * 40.0, 0.0, 30.0), 25.0, 85.0)
    return 0.8 * hist_thresh[-1] + 0.2 * alvo

def thread_serial(ser):
    global vel_bruta_acumulada
    ultimo_t = time.time()
    
    while rodando:
        try:
            linha = ser.readline().decode('ascii', errors='ignore').strip()
            partes = linha.split(',')
            if len(partes) == 6:
                vals = np.array([float(x) for x in partes])
                if np.all(np.abs(vals[:3]) < 45.0) and np.all(np.abs(vals[3:]) < 15.0):
                    agora = time.time()
                    dt = np.clip(agora - ultimo_t, 0.002, 0.05)
                    ultimo_t = agora

                    buffer_imu.append(vals)

                    # Extrai a aceleração dinâmica instantânea (última amostra sem gravidade)
                    janela_limpa = obter_janela_limpa()
                    acc_atual = janela_limpa[-1, :3]

                    # 1. Passo de Predição do Filtro de Kalman
                    kf.predict(acc_atual, dt)

                    # 2. Integrador Cego (Sem CNN/ZUPT) apenas para provar o Drift no gráfico!
                    vel_bruta_acumulada += acc_atual * dt

        except ValueError:
            pass

def resetar_drift_cego(event):
    """Pressione 'R' se a linha vermelha sem ZUPT for para o infinito e quiser trazê-la de volta."""
    global vel_bruta_acumulada
    if event.key in ['r', 'R']:
        vel_bruta_acumulada = np.zeros(3)
        print("[RESET] Drift do integrador cego zerado manualmente!")

if __name__ == "__main__":
    modelo = AttentionCNN()
    modelo.load_state_dict(torch.load(ARQUIVO_PESOS, weights_only=True))
    modelo.eval()
    print(f"[OK] Attention-CNN carregada de '{ARQUIVO_PESOS}'!")

    ser = serial.Serial(PORTA, BAUD, timeout=0.1)
    time.sleep(1.5)
    ser.reset_input_buffer()
    threading.Thread(target=thread_serial, args=(ser,), daemon=True).start()

    plt.style.use('dark_background')
    fig, (ax_sinal, ax_cnn, ax_vel) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title("FUSÃO ATTENTION-CNN + FILTRO DE KALMAN (ZUPT)")
    fig.canvas.mpl_connect('key_press_event', resetar_drift_cego)
    x_eixo = np.arange(JANELA)

    # Gráfico 1: Aceleração Dinâmica
    l_ax, = ax_sinal.plot(x_eixo, np.zeros(JANELA), label="Acc X", color="#00d4ff")
    l_ay, = ax_sinal.plot(x_eixo, np.zeros(JANELA), label="Acc Y", color="#00ff88")
    l_az, = ax_sinal.plot(x_eixo, np.zeros(JANELA), label="Acc Z", color="#ff3366")
    ax_sinal.set_ylim(-4.0, 4.0)
    ax_sinal.set_ylabel("Accel (m/s²)")
    ax_sinal.legend(loc="upper left", ncol=3, fontsize=8)
    ax_sinal.grid(True, alpha=0.2)
    titulo = ax_sinal.set_title("Inicializando...", fontsize=11, fontweight="bold")

    # Gráfico 2: Attention-CNN + Threshold Dinâmico
    l_prob, = ax_cnn.plot(x_eixo, np.zeros(JANELA), color="#ff9900", lw=2, label="Prob. Movimento CNN (%)")
    l_thresh, = ax_cnn.plot(x_eixo, np.ones(JANELA)*50.0, color="#00ffff", lw=1.5, ls="--", label="Threshold Dinâmico (%)")
    ax_cnn.set_ylim(-5, 105)
    ax_cnn.set_ylabel("Gatilho CNN (%)")
    ax_cnn.legend(loc="upper left", fontsize=8)
    ax_cnn.grid(True, alpha=0.2)

    # Gráfico 3: O Duelo de Velocidades (Integração Pura vs. CNN + Kalman)
    l_vel_crua, = ax_vel.plot(x_eixo, np.zeros(JANELA), color="#ff3366", ls="--", lw=1.5, alpha=0.8, label="Velocidade SEM ZUPT (Drift Explosivo)")
    l_vel_kf, = ax_vel.plot(x_eixo, np.zeros(JANELA), color="#00ff88", lw=2.5, label="Velocidade COM CNN + Kalman (Corrigida)")
    ax_vel.set_ylim(-0.05, 1.5)
    ax_vel.set_ylabel("Velocidade ||v|| (m/s)")
    ax_vel.set_xlabel("Tempo (Amostras) — Dica: Pressione [R] para zerar a linha vermelha de Drift quando ela explodir", color="#ffcc00")
    ax_vel.legend(loc="upper left", fontsize=9)
    ax_vel.grid(True, alpha=0.2)

    def atualizar(frame):
        global em_movimento
        dados = obter_janela_limpa()

        l_ax.set_ydata(dados[:, 0])
        l_ay.set_ydata(dados[:, 1])
        l_az.set_ydata(dados[:, 2])

        # 1. Inferência da Attention-CNN
        with torch.no_grad():
            entrada = torch.tensor(dados, dtype=torch.float32).T.unsqueeze(0)
            prob = torch.sigmoid(modelo(entrada)).item() * 100.0

        # 2. Threshold Dinâmico
        limiar = calcular_threshold_dinamico(dados, em_movimento)
        em_movimento = (prob >= limiar)

        # 3. SE ESTIVER PARADO, A CNN ACIONA O ZUPT NO FILTRO DE KALMAN!
        if not em_movimento:
            kf.update_zupt(prob)

        # Salva nos históricos para plotagem
        v_kalman = kf.get_speed_magnitude()
        v_cego = float(np.linalg.norm(vel_bruta_acumulada))

        hist_prob.append(prob)
        hist_thresh.append(limiar)
        hist_vel_kalman.append(v_kalman)
        hist_vel_sem_zupt.append(v_cego)

        l_prob.set_ydata(hist_prob)
        l_thresh.set_ydata(hist_thresh)
        l_vel_kf.set_ydata(hist_vel_kalman)
        l_vel_crua.set_ydata(hist_vel_sem_zupt)

        # Auto-escala suave no gráfico de velocidade
        max_v = max(0.8, np.max(hist_vel_kalman) * 1.3, min(3.0, np.max(hist_vel_sem_zupt) * 1.1))
        ax_vel.set_ylim(-0.05, max_v)

        if em_movimento:
            titulo.set_text(f"DINÂMICO (INTEGRANDO FÍSICA) | Vel. Real Kalman: {v_kalman:.3f} m/s | Drift Sem CNN: {v_cego:.2f} m/s")
            titulo.set_color("#ff3366")
        else:
            titulo.set_text(f"ESTACIONÁRIO (ZUPT KALMAN ATIVO) | Vel. Real Kalman: {v_kalman:.3f} m/s | Drift Sem CNN: {v_cego:.2f} m/s")
            titulo.set_color("#00ff88")

        return l_ax, l_ay, l_az, l_prob, l_thresh, l_vel_crua, l_vel_kf, titulo

    ani = FuncAnimation(fig, atualizar, interval=40, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    rodando = False
    ser.close()