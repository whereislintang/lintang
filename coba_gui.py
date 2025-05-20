import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib
import datetime

# --- Variabel Global ---
dataset_path = None
model = None
mse_value = None
r2_value = None
df_global = None
eval_scores = {}
label_stok_tersedia = None
label_estimasi_habis = None
label_user_aktif = None
label_barang_terbanyak = None


# --- Fungsi Navigasi Frame ---
def show_frame(frame_name):
    frames[frame_name].tkraise()
    create_navbar(frame_name)

# --- Fungsi Training ---
def pilih_dataset():
    global dataset_path
    filename = filedialog.askopenfilename(title="Pilih Dataset", filetypes=[("Excel Files", "*.xlsx")])
    if filename:
        dataset_path = filename
        dataset_label.config(text=filename.split("/")[-1])

def mulai_training():
    global model, df_global, mse_value, r2_value, eval_scores
    try:
        df = pd.read_excel(dataset_path)
        df_global = df.copy()

        numerical_df = df.select_dtypes(include=['float64', 'int64']).copy()

        if 'jumlah_barang_diambil' not in numerical_df.columns:
            log_text.insert(tk.END, "Kolom 'jumlah_barang_diambil' tidak ditemukan.\n")
            return

        # Klasifikasi biner berdasarkan rata-rata
        X = numerical_df.drop(columns=['jumlah_barang_diambil'])
        y = (numerical_df['jumlah_barang_diambil'] > numerical_df['jumlah_barang_diambil'].mean()).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if selected_model.get() == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if selected_model.get() == "Linear Regression":
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        mse_value = mean_squared_error(y_test, y_pred)
        r2_value = r2_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        eval_scores = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'loss': mse_value,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'roc_auc': roc_auc_score(y_test, y_prob) if selected_model.get() == "Logistic Regression" else None
        }

        log_text.insert(tk.END, f"Training selesai.\nMSE: {mse_value:.2f} | R2: {r2_value:.2f}\n")

    except Exception as e:
        log_text.insert(tk.END, f"Terjadi kesalahan saat training: {e}\n")

def save_model():
    if model:
        joblib.dump(model, "model_klasifikasi.joblib")
        log_text.insert(tk.END, "Model berhasil disimpan ke 'model_klasifikasi.joblib'.\n")
    else:
        log_text.insert(tk.END, "Belum ada model yang dilatih.\n")

# --- Fungsi Evaluasi ---
def evaluasi_model():
    if model is None or df_global is None:
        return

    label_mse.config(text=f"{mse_value:.2f}")
    label_r2.config(text=f"{r2_value:.2f}")

    try:
        df = df_global.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['no_kartu'] = df['no_kartu'].astype(str)
        df['barang'] = df['barang'].astype(str)

        kehabisan_df = df[df['berat_akhir'] == 0]
        prediksi_kehabisan = kehabisan_df.groupby('barang')['timestamp'].max().reset_index()
        prediksi_kehabisan.columns = ['Barang', 'Waktu Terakhir Habis']

        pengguna_teraktif = df['no_kartu'].value_counts().reset_index()
        pengguna_teraktif.columns = ['ID Kartu', 'Jumlah Akses']

        barang_populer = df['barang'].value_counts().reset_index()
        barang_populer.columns = ['Nama Barang', 'Frekuensi Diambil']

        frekuensi_pengambilan = len(df)

        visual_text.delete("1.0", tk.END)

        visual_text.insert(tk.END, "\nHasil Evaluasi Model:\n")
        for k, v in eval_scores.items():
            visual_text.insert(tk.END, f"{k}: {v:.2f}\n")

        visual_text.insert(tk.END, "\n\nPrediksi Waktu Kehabisan Barang:\n")
        if not prediksi_kehabisan.empty:
            visual_text.insert(tk.END, prediksi_kehabisan.to_string(index=False))
        else:
            visual_text.insert(tk.END, "Tidak ada data barang yang habis.\n")

        visual_text.insert(tk.END, "\n\nPengguna Teraktif:\n")
        visual_text.insert(tk.END, pengguna_teraktif.head().to_string(index=False))

        visual_text.insert(tk.END, "\n\nBarang Paling Sering Diambil:\n")
        visual_text.insert(tk.END, barang_populer.head().to_string(index=False))

        visual_text.insert(tk.END, f"\n\nTotal Frekuensi Pengambilan: {frekuensi_pengambilan}")

    except Exception as e:
        visual_text.insert(tk.END, f"Gagal menampilkan evaluasi: {e}\n")

# --- GUI Setup ---
root = tk.Tk()
root.title("GUI AI")
root.geometry("10000x10000")
root.configure(bg="#fde9f3")

container = tk.Frame(root)
container.pack(fill="both", expand=True)

frames = {}
for F in ("Dashboard", "Evaluasi", "Training"):
    frame = tk.Frame(container, bg="#fde9f3")
    frame.grid(row=0, column=0, sticky="nsew")
    frames[F] = frame

def create_navbar(current_page):
    for widget in root.winfo_children():
        if isinstance(widget, tk.Frame) and str(widget) != str(container):
            widget.destroy()

    navbar = tk.Frame(root, bg="#880e4f")
    navbar.place(x=0, y=0, width=10000, height=60)

    btns = [
        ("Dashboard", lambda: show_frame("Dashboard")),
        ("Evaluasi", lambda: show_frame("Evaluasi")),
        ("Training", lambda: show_frame("Training")),
    ]

    for name, cmd in btns:
        tk.Button(
            navbar, text=name, font=("Arial", 12, "bold"),
            fg="#fde9f3" if current_page != name else "#880e4f",
            bg="#880e4f" if current_page != name else "#fde9f3",
            bd=0, padx=20, pady=5, relief="flat",
            command=cmd
        ).pack(side="left", padx=15, pady=10)

# --- Dashboard Baru ---
frame_dashboard = tk.Frame(frames["Dashboard"], bg="#fde9f3", padx=50, pady=20)
frame_dashboard.pack(fill="both", expand=True)

header_frame = tk.Frame(frame_dashboard, bg="#c2185b")
header_frame.pack(fill="x")

tk.Label(header_frame, text="DASHBOARD", font=("Arial", 20, "bold"), bg="#c2185b", fg="#fde9f3").pack(side="left", padx=20)
tk.Label(header_frame, text=f"Last Update: {datetime.datetime.now().strftime('%d/%m/%Y')}", font=("Arial", 12, "bold"), bg="#c2185b", fg="#fde9f3").pack(side="right", padx=20)

tk.Label(frame_dashboard, text="DASHBOARD", font=("Arial", 18, "bold"), bg="#fde9f3", fg="#d81b60").pack(anchor="w", pady=(20, 10))

summary_frame = tk.Frame(frame_dashboard, bg="#fde9f3")
summary_frame.pack(fill="x")

# Kartu Ringkasan
card1 = tk.Frame(summary_frame, bg="#f48fb1", width=300, height=150)
card1.pack(side="left", padx=10, pady=10)
card1.pack_propagate(False)

card2 = tk.Frame(summary_frame, bg="#f06292", width=300, height=150)
card2.pack(side="left", padx=10, pady=10)
card2.pack_propagate(False)

card3 = tk.Frame(summary_frame, bg="#f48fb1", width=300, height=150)
card3.pack(side="left", padx=10, pady=10)
card3.pack_propagate(False)

card4 = tk.Frame(summary_frame, bg="#f06292", width=300, height=150)
card4.pack(side="left", padx=10, pady=10)
card4.pack_propagate(False)

# Kartu 1 - Stok Tersedia
label_stok_tersedia = tk.Label(card1, text="Stok Tersedia", font=("Arial", 20, "bold"), bg="#f48fb1", fg="white")
label_stok_tersedia.pack()

# Kartu 2 - Estimasi Habis
label_estimasi_habis = tk.Label(card2, text="Estimasi Habis Stok", font=("Arial", 18, "bold"), bg="#f06292", fg="white")
label_estimasi_habis.pack()


# Kartu 3 - User Aktif
label_user_aktif = tk.Label(card3, text="Pengguna Paling Aktif", font=("Arial", 18, "bold"), bg="#f48fb1", fg="white")
label_user_aktif.pack()


# Kartu 4 - Barang Terbanyak
label_barang_terbanyak = tk.Label(card4, text="Pengambilan Barang Terbanyak", font=("Arial", 18, "bold"), bg="#f06292", fg="white")
label_barang_terbanyak.pack()

#update dashboard
def update_dashboard():
    if df_global is None:
        return

    try:
        df = df_global.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['no_kartu'] = df['no_kartu'].astype(str)
        df['barang'] = df['barang'].astype(str)

        stok_tersedia = df[df['berat_akhir'] > 0]['barang'].nunique()
        label_stok_tersedia.config(text=f"{stok_tersedia} item")

        df_kehabisan = df[df['berat_akhir'] == 0]
        if not df_kehabisan.empty:
            max_tanggal = df_kehabisan['timestamp'].max()
            delta_hari = (datetime.datetime.now() - max_tanggal).days
            estimasi = f"{max(0, 7 - delta_hari)} Hari Lagi"
        else:
            estimasi = "Tidak Teridentifikasi"
        label_estimasi_habis.config(text=estimasi)

        user_aktif = df[df['timestamp'].dt.date == datetime.datetime.now().date()]['no_kartu'].nunique()
        label_user_aktif.config(text=f"{user_aktif} User")

        barang_terbanyak = df['barang'].value_counts().idxmax()
        label_barang_terbanyak.config(text=barang_terbanyak)

    except Exception as e:
        print(f"Gagal update dashboard: {e}")

update_dashboard()

# --- Evaluasi ---
tab_evaluasi = tk.Frame(frames["Evaluasi"], bg="#fde9f3")
tab_evaluasi.pack(fill="both", expand=True)

frame_header = tk.Frame(tab_evaluasi, bg="#fbb8cc", padx=20, pady=20)
frame_header.pack(fill="x", pady=(100, 20))

tk.Label(frame_header, text="Model: ", font=("Arial", 14), bg="#fbb8cc", fg="#932b4d").pack(side="left", padx=(0, 10))

# Dropdown for model selection
selected_model = tk.StringVar(value="Logistic Regression")
model_selector = ttk.Combobox(frame_header, textvariable=selected_model, values=["Logistic Regression", "Linear Regression"], state="readonly")
model_selector.pack(side="left", padx=(0, 10))

btn_eval = tk.Button(frame_header, text="Evaluasi Model", font=("Arial", 12, "bold"), bg="#880e4f", fg="#fde9f3", command=evaluasi_model)
btn_eval.pack(side="left")

frame_body = tk.Frame(tab_evaluasi, bg="#fde9f3", padx=20, pady=20)
frame_body.pack(expand=True, fill="both")

frame_hasil = tk.LabelFrame(frame_body, text="Hasil Evaluasi", font=("Arial", 14, "bold"), fg="#880e4f", bg="#ffc1e3", padx=10, pady=10)
frame_hasil.pack(side="left", expand=True, fill="both", padx=(0, 10))

tk.Label(frame_hasil, text="Mean Squared Error (MSE):", font=("Arial", 12), bg="#ffc1e3", fg="#880e4f").pack(anchor="w", pady=5)
label_mse = tk.Label(frame_hasil, text="-", font=("Arial", 12, "bold"), bg="#ffc1e3", fg="#880e4f")
label_mse.pack(anchor="w", pady=5)

tk.Label(frame_hasil, text="R-Square (R2):", font=("Arial", 12), bg="#ffc1e3", fg="#880e4f").pack(anchor="w", pady=5)
label_r2 = tk.Label(frame_hasil, text="-", font=("Arial", 12, "bold"), bg="#ffc1e3", fg="#880e4f")
label_r2.pack(anchor="w", pady=5)

frame_visual = tk.LabelFrame(frame_body, text="Visualisasi Evaluasi", font=("Arial", 14, "bold"), fg="#880e4f", bg="#ffc1e3", padx=10, pady=10)
frame_visual.pack(side="left", expand=True, fill="both")

visual_text = tk.Text(frame_visual, bg="#fff0f6", fg="#880e4f", font=("Arial", 11), wrap="word")
visual_text.pack(fill="both", expand=True)

# --- Training ---
dataset_section = tk.LabelFrame(frames["Training"], text="DATASET", font=("Arial", 14, "bold"), fg="#f48fb1", bg="#fde9f3", bd=0)
dataset_section.pack(pady=80, padx=50, fill="x")

dataset_inner = tk.Frame(dataset_section, bg="#ffc1e3")
dataset_inner.pack(padx=10, pady=10, fill="x")

tk.Button(dataset_inner, text="Pilih Dataset", font=("Arial", 12, "bold"), bg="#fde9f3", command=pilih_dataset).pack(side="left", padx=10, pady=10)
dataset_label = tk.Label(dataset_inner, text="Belum ada file", font=("Arial", 12, "bold"), bg="#ffc1e3")
dataset_label.pack(side="left", padx=10)

training_section = tk.LabelFrame(frames["Training"], text="TRAINING", font=("Arial", 14, "bold"), fg="#f48fb1", bg="#fde9f3", bd=0)
training_section.pack(padx=50, fill="both", expand=True)

training_inner = tk.Frame(training_section, bg="#ffc1e3")
training_inner.pack(padx=10, pady=10, fill="both", expand=True)

btns_frame = tk.Frame(training_inner, bg="#ffc1e3")
btns_frame.pack(side="left", padx=10, pady=10)

tk.Button(btns_frame, text="Mulai Training", font=("Arial", 12, "bold"), bg="#fde9f3", command=mulai_training).pack(pady=10)
tk.Button(btns_frame, text="Save Model", font=("Arial", 12, "bold"), bg="#fde9f3", command=save_model).pack(pady=10)

log_text = tk.Text(training_inner, bg="#ba4b74", fg="#fde9f3", font=("Arial", 12), wrap="word")
log_text.pack(side="left", padx=20, pady=10, fill="both", expand=True)

# --- Inisialisasi ---
create_navbar("Dashboard")
show_frame("Dashboard")

root.mainloop()
