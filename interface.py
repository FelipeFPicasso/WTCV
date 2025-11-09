import tkinter as tk
import sys
from tkinter import messagebox, ttk
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "resultados_faces.csv"
CAPTADOR_SCRIPT = "captador_face.py"

# Função para rodar o captador_face.py
def iniciar_analise():
    try:
        messagebox.showinfo("Iniciando", "Análise sendo executada... Isso pode levar alguns segundos.")
        subprocess.run([sys.executable, CAPTADOR_SCRIPT], check=True)
        messagebox.showinfo("Concluído", "Análise finalizada com sucesso! O arquivo CSV foi gerado.")
    except subprocess.CalledProcessError:
        messagebox.showerror("Erro", "Erro ao executar o script de análise.")
    except Exception as e:
        messagebox.showerror("Erro inesperado", str(e))

# Função para abrir e exibir os dados do CSV
def mostrar_resultados():
    if not os.path.exists(CSV_PATH):
        messagebox.showwarning("Aviso", "Nenhum resultado encontrado. Rode a análise primeiro!")
        return

    df = pd.read_csv(CSV_PATH)

    # nova janela
    janela_resultado = tk.Toplevel(root)
    janela_resultado.title("Resultados da Análise")
    janela_resultado.geometry("700x500")

    label = tk.Label(janela_resultado, text="Dados coletados:", font=("Arial", 12, "bold"))
    label.pack(pady=5)

    # Tabela
    frame = ttk.Frame(janela_resultado)
    frame.pack(fill=tk.BOTH, expand=True)

    tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=150)
    for _, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))
    tree.pack(fill=tk.BOTH, expand=True)

    # Botão gráfico
    def mostrar_grafico():
        plt.figure(figsize=(6, 4))
        plt.bar(df["ID Pessoa"], df["Tempo olhando (s)"], color="skyblue")
        plt.xlabel("Rosto detectado")
        plt.ylabel("Tempo olhando (segundos)")
        plt.title("Tempo de atenção por rosto")
        plt.tight_layout()
        plt.show()

    btn_grafico = tk.Button(janela_resultado, text="Mostrar gráfico", command=mostrar_grafico, bg="#0078D7", fg="white")
    btn_grafico.pack(pady=10)


# Tela Principal
root = tk.Tk()
root.title("Reconhecimento Facial - Rieder Scanner")
root.geometry("400x300")
root.configure(bg="#F0F0F0")

titulo = tk.Label(root, text="Rieder Face Scanner", font=("Arial", 16, "bold"), bg="#F0F0F0")
titulo.pack(pady=20)

descricao = tk.Label(root, text="Analise tempo de atenção em vídeos.", bg="#F0F0F0")
descricao.pack(pady=5)

btn_iniciar = tk.Button(root, text="▶️ Iniciar Análise", command=iniciar_analise, bg="#28A745", fg="white", font=("Arial", 12, "bold"))
btn_iniciar.pack(pady=15)

btn_resultados = tk.Button(root, text="📈 Ver Resultados", command=mostrar_resultados, bg="#0078D7", fg="white", font=("Arial", 12, "bold"))
btn_resultados.pack(pady=10)

btn_sair = tk.Button(root, text="❌ Sair", command=root.quit, bg="#DC3545", fg="white", font=("Arial", 11))
btn_sair.pack(pady=15)

root.mainloop()
