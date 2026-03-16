import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SymulatorStacji:
    """
    Klasa odpowiadająca za warstwę logiczną i matematyczną symulacji.
    Implementuje model obsługi masowej (kolejkowy) typu M/M/S/S:
    - M: Markowowski proces wejścia (rozkład Poissona dla zgłoszeń),
    - M: Markowowski proces obsługi (w naszym przypadku zaimplementowano Gaussa wg wytycznych),
    - S: Liczba równoległych kanałów obsługi,
    - S: Maksymalna pojemność systemu (kanały + bufor/kolejka).
    """
    def __init__(self, liczba_kanalow, lambd, N, sigma, min_czas, maks_czas, max_kolejka, czas_symulacji):
        # --- Inicjalizacja podstawowych parametrów konfiguracyjnych systemu --- [cite: 603-612]
        self.liczba_kanalow = liczba_kanalow      # Liczba dostępnych linii/kanałów stacji bazowej
        self.lambd = lambd                        # Natężenie ruchu (średnia liczba zgłoszeń w jednostce czasu)
        self.N = N                                # Średni czas trwania rozmowy (wartość oczekiwana dla rozkładu Gaussa)
        self.sigma = sigma                        # Odchylenie standardowe dla czasu rozmowy
        self.min_czas = min_czas                  # Dolny limit czasu trwania połączenia (w sekundach)
        self.maks_czas = maks_czas                # Górny limit czasu trwania połączenia (w sekundach)
        self.max_kolejka = max_kolejka            # Maksymalna liczba klientów oczekujących na wolny kanał
        self.czas_symulacji = czas_symulacji      # Całkowity czas, przez jaki system ma działać
        
        # --- Zmienne przechowujące bieżący stan systemu ---
        self.kanaly = []        # Przechowuje czasy pozostałe do końca obsługi dla zajętych kanałów
        self.kolejka = []       # Bufor przechowujący czasy obsługi dla klientów oczekujących
        self.odrzuceni = 0      # Licznik zgłoszeń odrzuconych (gdy kanały i kolejka są pełne)
        self.obsluzeni = 0      # Licznik zgłoszeń, które pomyślnie zakończyły rozmowę
        self.aktualny_krok = 0  # Aktualna sekunda symulacji
        
        # --- Listy zbierające historię danych na potrzeby generowania wykresów --- [cite: 617-622]
        self.historia_rho = []  # Historia wskaźnika zajętości kanałów (Ro)
        self.historia_Q = []    # Historia długości kolejki (Q)
        self.historia_W = []    # Historia średniego czasu oczekiwania (W)
        
        # --- Bufor wygenerowanych, przyszłych zdarzeń --- [cite: 626-627]
        self.lista_lambda = []  # Odstępy czasowe między kolejnymi zgłoszeniami
        self.lista_mi = []      # Wylosowane czasy trwania rozmowy dla kolejnych zgłoszeń
        self.klienci_dane = []  # Słowniki ze szczegółowymi danymi klientów (do tabeli w GUI)

        # --- Zmienne przechowujące ostatnie wartości generatorów (na potrzeby widoku na żywo w GUI) ---
        self.u_poisson = 0
        self.t_poisson = 0
        self.u1_gauss = 0
        self.u2_gauss = 0
        self.x_gauss = 0

        # Przed startem symulacji, z góry generujemy pełną pulę zgłoszeń, które nadejdą w badanym czasie [cite: 626]
        self.generuj_zdarzenia()
        
        # --- Inicjalizacja i formatowanie nagłówka w pliku tekstowym Wyniki.txt --- [cite: 618, 623-624]
        with open("Wyniki.txt", "w", encoding="utf-8") as f:
            f.write("=== PARAMETRY SYMULACJI ===\n")
            f.write(f"Liczba kanalow: {self.liczba_kanalow}\n")
            f.write(f"Dlugosc kolejki: {self.max_kolejka}\n")
            f.write(f"Natezenie ruchu (lambda): {self.lambd}\n")
            f.write(f"Srednia dlugosc rozmowy (N): {self.N}, Sigma: {self.sigma}\n")
            f.write(f"Min. dlugosc rozmowy: {self.min_czas}, Maks: {self.maks_czas}\n")
            f.write(f"Czas symulacji: {self.czas_symulacji} s\n")
            f.write("===========================\n\n")
            f.write(f"{'Krok (s)':<10} | {'Kolejka (Q)':<12} | {'Czas Oczek. (W)':<16} | {'Ro (Zajetosc)':<14} | {'Obsluzeni':<10} | {'Odrzuceni':<10}\n")
            f.write("-" * 85 + "\n")

    def generuj_zdarzenia(self):
        """
        Metoda realizująca KROK 1 i 2 algorytmu:
        Generuje z góry ciąg zdarzeń (przyjść klientów i czasów ich obsługi).
        Wykorzystuje wbudowane generatory liczb pseudolosowych z przedziału (0,1).
        """
        id_klienta = 1
        suma_calkowita = 0
        
        # Generujemy zdarzenia z podwójnym zapasem czasu, aby upewnić się, że nie zabraknie nam 
        # zgłoszeń w ostatnich sekundach działania algorytmu.
        while suma_calkowita < self.czas_symulacji * 2:
            
            # --- GENERATOR POISSONA --- [cite: 626]
            # Odstępy między zgłoszeniami w procesie Poissona mają rozkład wykładniczy.
            # Wzór: t = -ln(u) / lambda
            u = np.random.rand()
            odstep_lambda = -np.log(u) / self.lambd
            self.lista_lambda.append(odstep_lambda)
            suma_calkowita += odstep_lambda
            
            # Zapisanie wartości dla pierwszego klienta, aby wyświetlić je startowo w polach GUI
            if id_klienta == 1:
                self.u_poisson = u
                self.t_poisson = odstep_lambda
            
            # --- GENERATOR GAUSSA (Metoda Boxa-Mullera) --- [cite: 627]
            # Służy do wygenerowania czasu obsługi (rozmowy) o zadanym rozkładzie normalnym.
            u1, u2 = np.random.rand(), np.random.rand()
            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            czas_rozmowy = self.N + self.sigma * z0
            
            # Ograniczenie wartości wygenerowanego czasu do bezpiecznych widełek (Min, Maks) podanych w GUI
            czas_rozmowy = max(self.min_czas, min(czas_rozmowy, self.maks_czas))
            
            if id_klienta == 1:
                self.u1_gauss = u1
                self.u2_gauss = u2
                self.x_gauss = czas_rozmowy
                
            self.lista_mi.append(int(czas_rozmowy))
            
            # Obliczenie kumulatywnych, uśrednionych wartości parametrów dla Dziennika Zdarzeń [cite: 522]
            b_lambda_i = id_klienta / suma_calkowita
            b_mi_i = id_klienta / sum(self.lista_mi) if sum(self.lista_mi) > 0 else 0
            b_ro_i = b_lambda_i / b_mi_i if b_mi_i > 0 else 0
            
            # Zapisanie struktury danych klienta do logów
            self.klienci_dane.append({
                "id": id_klienta,
                "poisson": round(u, 4),
                "odstep_lambda": round(odstep_lambda, 4),
                "mi_i": int(czas_rozmowy),
                "lambdai": round(b_lambda_i, 3),
                "mii": round(b_mi_i, 3),
                "roi": round(b_ro_i, 3)
            })
            id_klienta += 1

    def wykonaj_sekunde(self):
        """
        Metoda realizująca KROK 3 algorytmu:
        Wykonywana cyklicznie co 1 sekundę. Zarządza ruchem zgłoszeń w systemie,
        aktualizuje stany kanałów i kolejki oraz zapisuje statystyki bieżące.
        """
        # Jeśli osiągnięto zadany limit czasu, przerywamy symulację
        if self.aktualny_krok >= self.czas_symulacji:
            return False, [], 0

        # --- Krok 3a: Analiza napływu zgłoszeń w danej sekundzie --- [cite: 629]
        # Pobieramy kolejne wartości lambda_i z bufora, dopóki ich suma nie przekroczy 1 sekundy.
        # Wartość 'k' oznacza, ilu klientów faktycznie nadeszło w tym jednostkowym czasie.
        k = 0
        suma_lambda = 0
        for lam in self.lista_lambda:
            k += 1
            suma_lambda += lam
            if suma_lambda >= 1.0:
                break

        # Brak zgłoszeń -> zabezpieczenie przed błędem
        if k == 0 or len(self.lista_lambda) == 0:
            return False, [], 0

        nowi_klienci = []
        
        # --- Krok 3b: Umieszczenie pobranych elementów w symulatorze --- [cite: 630]
        for i in range(k):
            mi = self.lista_mi[i]
            nowi_klienci.append(self.klienci_dane[i])

            # W pierwszej kolejności sprawdzamy, czy są wolne kanały
            if len(self.kanaly) < self.liczba_kanalow:
                self.kanaly.append(mi)
            # Jeśli kanały zajęte, sprawdzamy czy jest miejsce w kolejce (buforze)
            elif len(self.kolejka) < self.max_kolejka:
                self.kolejka.append(mi)
            # Jeśli zarówno kanały jak i kolejka są pełne - zgłoszenie zostaje definitywnie odrzucone
            else:
                self.odrzuceni += 1 

        # --- Krok 3c: Obliczenie wskaźników statystycznych (Ro, Q, W) --- [cite: 631]
        # Ro (Intensywność / Zajętość): Stosunek zajętych kanałów do wszystkich dostępnych
        rho = len(self.kanaly) / self.liczba_kanalow if self.liczba_kanalow > 0 else 0
        
        # Q: Bieżąca długość kolejki
        Q = len(self.kolejka)
        
        # W: Średni czas oczekiwania obliczony na podstawie uproszczonego Prawa Little'a (W = Q / lambda)
        W = Q / self.lambd if self.lambd > 0 else 0

        self.historia_rho.append(rho)
        self.historia_Q.append(Q)
        self.historia_W.append(W)
        
        # Zapis wyników dla tej sekundy do pliku tekstowego [cite: 632]
        with open("Wyniki.txt", "a", encoding="utf-8") as f:
            f.write(f"{self.aktualny_krok:<10} | {Q:<12} | {round(W,4):<16} | {round(rho,4):<14} | {self.obsluzeni:<10} | {self.odrzuceni:<10}\n")

        # --- Krok 3d: Usunięcie przetworzonych zgłoszeń z początków list --- [cite: 633]
        self.lista_lambda = self.lista_lambda[k:]
        self.lista_mi = self.lista_mi[k:]
        self.klienci_dane = self.klienci_dane[k:]

        # --- Krok 3e: Obsługa trwających połączeń i przesuwanie kolejki --- [cite: 634]
        # Dekrementacja (zmniejszenie o 1 sekundę) czasu trwania rozmów we wszystkich zajętych kanałach
        for i in range(len(self.kanaly) - 1, -1, -1):
            self.kanaly[i] -= 1
            # Gdy czas spadnie do 0, klient opuszcza system (zakończył rozmowę)
            if self.kanaly[i] <= 0:
                self.kanaly.pop(i) 
                self.obsluzeni += 1

        # Przeniesienie pierwszych klientów z kolejki do nowo zwolnionych kanałów
        while len(self.kanaly) < self.liczba_kanalow and len(self.kolejka) > 0:
            self.kanaly.append(self.kolejka.pop(0))

        self.aktualny_krok += 1
        return True, nowi_klienci, k


class AplikacjaGUI:
    """
    Klasa odpowiedzialna za warstwę wizualną (Graficzny Interfejs Użytkownika - GUI).
    Wdrożono "Responsive Design" wykorzystujący układ typu Grid, co pozwala aplikacji 
    płynnie dostosowywać się do dowolnej rozdzielczości i rozmiaru okna.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("aplikacja mateusz konachowicz")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)  # Ustalenie minimalnego rozmiaru, by zapobiec zepsuciu interfejsu
        self.root.configure(bg="#eef2f5") 
        
        # Maksymalizacja okna na starcie, o ile system operacyjny na to pozwala
        try:
            self.root.state('zoomed')
        except:
            pass
        
        self.symulator = None
        self.dziala = False
        
        self.buduj_interfejs()
        
    def buduj_interfejs(self):
        """Inicjalizacja i pozycjonowanie wszystkich kontrolek wizualnych (pól, przycisków, wykresów)."""
        # Ustawienie nowoczesnego stylu dla biblioteki Tkinter
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#eef2f5")
        style.configure("TLabelframe", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background="#ffffff", font=("Segoe UI", 10, "bold"), foreground="#333333")
        style.configure("TLabel", background="#ffffff", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5)

        # ====== GŁÓWNA SIATKA OKNA ======
        # Ustawienie "wag" dla wierszy. Wiersz 0 (góra z wykresami) dostaje 60% miejsca, wiersz 1 (dół z logami) 40%.
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=6) 
        self.root.rowconfigure(1, weight=4) 

        frame_top = tk.Frame(self.root, bg="#eef2f5")
        frame_top.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        frame_bot = tk.Frame(self.root, bg="#eef2f5")
        frame_bot.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # --- SIATKA DLA GÓRNEGO PANELU ---
        # Podział na 3 kolumny: Parametry, Kanały na żywo, Wykresy.
        frame_top.rowconfigure(0, weight=1)
        frame_top.columnconfigure(0, weight=0, minsize=280) 
        frame_top.columnconfigure(1, weight=1, minsize=250) 
        frame_top.columnconfigure(2, weight=2, minsize=350) 

        frame_lewa = tk.Frame(frame_top, bg="#ffffff", bd=1, relief="ridge", padx=10, pady=10)
        frame_lewa.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        frame_srodek = tk.Frame(frame_top, bg="#ffffff", bd=1, relief="ridge", padx=10, pady=10)
        frame_srodek.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        
        frame_prawa = tk.Frame(frame_top, bg="#ffffff", bd=1, relief="ridge", padx=5, pady=5)
        frame_prawa.grid(row=0, column=2, sticky="nsew")

        # ================= KOLUMNA LEWA: POLA PARAMETRÓW =================
        tk.Label(frame_lewa, text="⚙️ Konfiguracja Systemu", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#0056b3").pack(pady=(0, 15))
        
        frame_pola = tk.Frame(frame_lewa, bg="#ffffff")
        frame_pola.pack(fill=tk.X)
        
        self.pola = {}
        # Słownik definiujący strukturę pól formularza [cite: 603-612]
        parametry_def = [
            ("Liczba kanałów", 10, ""), ("Długość kolejki", 10, ""), 
            ("Natężenie ruchu [λ]", 1.0, ""), ("Śr. długość rozmowy (N)", 20, "s"), 
            ("Odchylenie stand. (σ)", 5, ""), ("Min. czas połączenia", 10, "s"), 
            ("Maks. czas połączenia", 30, "s"), ("Czas symulacji", 30, "s")
        ]
        
        # Pętla generująca etykiety i pola wprowadzania tekstu (Entry)
        for i, (nazwa, wart, jednostka) in enumerate(parametry_def):
            tk.Label(frame_pola, text=nazwa, bg="#ffffff", font=("Segoe UI", 10)).grid(row=i, column=0, sticky=tk.W, pady=6)
            var = tk.StringVar(value=str(wart))
            ent = ttk.Entry(frame_pola, textvariable=var, width=6, justify="center")
            ent.grid(row=i, column=1, padx=5, sticky="ew")
            if jednostka:
                tk.Label(frame_pola, text=jednostka, bg="#ffffff").grid(row=i, column=2, sticky=tk.W)
            self.pola[nazwa] = var
        frame_pola.columnconfigure(1, weight=1)

        # Sekcja przycisków sterujących
        frame_przyciski = tk.Frame(frame_lewa, bg="#ffffff")
        frame_przyciski.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        btn_start = tk.Button(frame_przyciski, text="▶ START SYMULACJI", font=("Segoe UI", 12, "bold"), bg="#28a745", fg="white", relief="flat", command=self.start_symulacji)
        btn_start.pack(fill=tk.X, pady=5, ipady=5)
        
        frame_pp = tk.Frame(frame_przyciski, bg="#ffffff")
        frame_pp.pack(fill=tk.X, pady=5)
        tk.Button(frame_pp, text="⏸ Pauza", font=("Segoe UI", 10), bg="#ffc107", relief="flat", command=self.pauza).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        tk.Button(frame_pp, text="⏵ Wznów", font=("Segoe UI", 10), bg="#17a2b8", fg="white", relief="flat", command=self.wznow).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2,0))
        
        self.var_pokaz = tk.BooleanVar(value=True)
        tk.Checkbutton(frame_przyciski, text="Aktualizuj wykresy na żywo", variable=self.var_pokaz, bg="#ffffff", font=("Segoe UI", 9)).pack(pady=10)

        # ================= KOLUMNA ŚRODKOWA: WIDOK NA ŻYWO (KANAŁY I KONTROLKI) =================
        tk.Label(frame_srodek, text="📡 Stan Kanałów (Live)", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#0056b3").pack(pady=(0, 10))
        
        # Pasek postępu kolejki i liczniki bieżące [cite: 614]
        frame_status = tk.Frame(frame_srodek, bg="#ffffff")
        frame_status.pack(fill=tk.X, pady=5)
        
        self.lbl_czas = tk.Label(frame_status, text="Czas: 0 / 30 s", font=("Segoe UI", 13, "bold"), bg="#ffffff", fg="#dc3545")
        self.lbl_czas.pack(side=tk.RIGHT, padx=5)
        
        self.lbl_kolejka = tk.Label(frame_status, text="Kolejka: 0 / 10", font=("Segoe UI", 10, "bold"), bg="#ffffff")
        self.lbl_kolejka.pack(side=tk.LEFT, padx=5)
        
        self.pb_kolejka = ttk.Progressbar(frame_srodek, orient="horizontal", mode="determinate")
        self.pb_kolejka.pack(fill=tk.X, padx=10, pady=5)

        # Kontener zarezerwowany dla dynamicznie wyliczanej i rysowanej siatki kanałów
        self.frame_kanaly_container = tk.Frame(frame_srodek, bg="#ffffff")
        self.frame_kanaly_container.pack(fill=tk.BOTH, expand=True, pady=10)
        self.frame_kanaly = None
        self.kanaly_lbl = []

        # Tabela w formie Grid zawierająca liczniki sumaryczne (obsłużeni, odrzuceni, k)
        frame_stats_live = tk.Frame(frame_srodek, bg="#f8f9fa", bd=1, relief="solid")
        frame_stats_live.pack(fill=tk.X, pady=10, ipady=5)
        
        frame_stats_live.columnconfigure(0, weight=1)
        frame_stats_live.columnconfigure(1, weight=1)
        
        self.lbl_obsluzone = tk.Label(frame_stats_live, text="Obsłużone: 0", font=("Segoe UI", 10), bg="#f8f9fa", fg="#28a745")
        self.lbl_obsluzone.grid(row=0, column=0, pady=(5, 2), sticky="ew")
        
        self.lbl_odrzucone = tk.Label(frame_stats_live, text="Odrzucone: 0", font=("Segoe UI", 10), bg="#f8f9fa", fg="#dc3545")
        self.lbl_odrzucone.grid(row=0, column=1, pady=(5, 2), sticky="ew")
        
        # Wartość z kroku 3a ("k" zgłoszeń na sekundę) jest wyśrodkowana na dole [cite: 629]
        self.lbl_wartosc_k = tk.Label(frame_stats_live, text="Przyjęcia w tej sek (k): 0", font=("Segoe UI", 10, "bold"), bg="#f8f9fa")
        self.lbl_wartosc_k.grid(row=1, column=0, columnspan=2, pady=(2, 5), sticky="ew")

        # ================= KOLUMNA PRAWA: WYKRESY (MATPLOTLIB) ================= [cite: 617-622]
        self.fig = Figure(dpi=100) 
        self.fig.patch.set_facecolor('#ffffff')
        self.ax_q = self.fig.add_subplot(311)
        self.ax_w = self.fig.add_subplot(312)
        self.ax_ro = self.fig.add_subplot(313)
        
        # Osadzenie płótna Matplotlib (FigureCanvasTkAgg) wewnątrz interfejsu graficznego Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_prawa)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Etykiety ze średnimi arytmetycznymi pod każdym wykresem
        frame_etykiety = tk.Frame(frame_prawa, bg="#ffffff")
        frame_etykiety.pack(fill=tk.X, pady=2)
        self.lbl_avg_q = tk.Label(frame_etykiety, text="Średnia Q: 0.00", bg="#ffffff", font=("Segoe UI", 9, "bold"), fg="#dc3545")
        self.lbl_avg_q.pack(side=tk.LEFT, expand=True)
        self.lbl_avg_w = tk.Label(frame_etykiety, text="Średnie W: 0.00", bg="#ffffff", font=("Segoe UI", 9, "bold"), fg="#007bff")
        self.lbl_avg_w.pack(side=tk.LEFT, expand=True)
        self.lbl_avg_ro = tk.Label(frame_etykiety, text="Średnie Ro: 0.00", bg="#ffffff", font=("Segoe UI", 9, "bold"), fg="#28a745")
        self.lbl_avg_ro.pack(side=tk.LEFT, expand=True)

        # ================= PANEL DOLNY (GENERATORY I LOGI) =================
        frame_bot.rowconfigure(0, weight=1)
        frame_bot.columnconfigure(0, weight=0, minsize=280)
        frame_bot.columnconfigure(1, weight=1)

        # Lewa strona dołu - pola wyświetlające aktualne wyniki z generatorów pseudolosowych
        frame_geny = ttk.LabelFrame(frame_bot, text="Wartości Generatorów")
        frame_geny.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        tk.Label(frame_geny, text="Rozkład Poissona", font=("Segoe UI", 9, "bold"), bg="#ffffff").grid(row=0, column=0, columnspan=2, pady=(5,2))
        tk.Label(frame_geny, text="u:", bg="#ffffff").grid(row=1, column=0, sticky=tk.E, padx=2)
        self.var_poisson_x = tk.StringVar()
        ttk.Entry(frame_geny, textvariable=self.var_poisson_x, width=8, state='readonly').grid(row=1, column=1, padx=2)
        tk.Label(frame_geny, text="Odstęp (\u03BB_i):", bg="#ffffff").grid(row=2, column=0, sticky=tk.E, padx=2)
        self.var_poisson_l = tk.StringVar()
        ttk.Entry(frame_geny, textvariable=self.var_poisson_l, width=8, state='readonly').grid(row=2, column=1, padx=2)

        tk.Label(frame_geny, text="Rozkład Gaussa", font=("Segoe UI", 9, "bold"), bg="#ffffff").grid(row=0, column=2, columnspan=2, pady=(5,2), padx=10)
        tk.Label(frame_geny, text="u1:", bg="#ffffff").grid(row=1, column=2, sticky=tk.E, padx=2)
        self.var_gauss_x1 = tk.StringVar()
        ttk.Entry(frame_geny, textvariable=self.var_gauss_x1, width=8, state='readonly').grid(row=1, column=3, padx=2)
        tk.Label(frame_geny, text="u2:", bg="#ffffff").grid(row=2, column=2, sticky=tk.E, padx=2)
        self.var_gauss_x2 = tk.StringVar()
        ttk.Entry(frame_geny, textvariable=self.var_gauss_x2, width=8, state='readonly').grid(row=2, column=3, padx=2)
        tk.Label(frame_geny, text="Czas (\u03BC_i):", bg="#ffffff").grid(row=3, column=2, sticky=tk.E, padx=2)
        self.var_gauss_x = tk.StringVar()
        ttk.Entry(frame_geny, textvariable=self.var_gauss_x, width=8, state='readonly').grid(row=3, column=3, padx=2, pady=(0,5))

        # Prawa strona dołu - Szeroki widok na Dziennik Zdarzeń (Tabela typu Treeview)
        lf_tabela = ttk.LabelFrame(frame_bot, text="Dziennik Zdarzeń z wytycznych")
        lf_tabela.grid(row=0, column=1, sticky="nsew")
        
        scroll = ttk.Scrollbar(lf_tabela)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Kolumny odpowiadają zadanym parametrom matematycznym z PDF-a [cite: 631]
        kolumny = ("ID Klienta", "Pois (u)", "Odstęp (\u03BB_i)", "Rozmowa (\u03BC_i)", "Średnie \u03BB", "Średnie \u03BC", "Średnie Ro")
        self.tabela = ttk.Treeview(lf_tabela, columns=kolumny, show='headings', yscrollcommand=scroll.set, height=4)
        for col in kolumny:
            self.tabela.heading(col, text=col)
            self.tabela.column(col, width=60, minwidth=40, anchor=tk.CENTER)
        self.tabela.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.config(command=self.tabela.yview)

    def start_symulacji(self):
        """Zdarzenie wywoływane po kliknięciu przycisku START. Inicjuje obiekt logiki na nowo."""
        try:
            # Rzutowanie i walidacja wartości pobranych z GUI
            l_kanalow = int(self.pola["Liczba kanałów"].get())
            l_kolejka = int(self.pola["Długość kolejki"].get())
            lambd = float(self.pola["Natężenie ruchu [λ]"].get())
            N = float(self.pola["Śr. długość rozmowy (N)"].get())
            sigma = float(self.pola["Odchylenie stand. (σ)"].get())
            min_c = float(self.pola["Min. czas połączenia"].get())
            max_c = float(self.pola["Maks. czas połączenia"].get())
            czas_sym = int(self.pola["Czas symulacji"].get())
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadzono niepoprawne wartości parametrów!")
            return

        # --- DYNAMICZNA REKONSTRUKCJA SIATKI KANAŁÓW ---
        if self.frame_kanaly:
            self.frame_kanaly.destroy() # Usunięcie starej konfiguracji kanałów
        
        self.frame_kanaly = tk.Frame(self.frame_kanaly_container, bg="#ffffff")
        self.frame_kanaly.pack(expand=True)

        self.kanaly_lbl = []
        
        # Algorytm ustalania idealnej, niemal kwadratowej siatki wyświetlania (za pomocą pierwiastka).
        # Zapobiega generowaniu "wysokich słupów" zniekształcających ułożenie GUI przy dużej liczbie kanałów.
        liczba_kolumn = max(2, int(np.ceil(np.sqrt(l_kanalow))))
        liczba_wierszy = int(np.ceil(l_kanalow / float(liczba_kolumn)))

        # Przypisanie "wag" elementom siatki – to w tym miejscu uaktywniamy rozciągliwość kafelków
        for c in range(liczba_kolumn):
            self.frame_kanaly.columnconfigure(c, weight=1)
        for r in range(liczba_wierszy):
            self.frame_kanaly.rowconfigure(r, weight=1)
        
        # Fizyczne wygenerowanie etykiet reprezentujących wolne/zajęte linie [cite: 614]
        for i in range(l_kanalow):
            lbl = tk.Label(self.frame_kanaly, text="", bg="#28a745", fg="white", font=("Segoe UI", 11, "bold"), width=3, height=1, relief="flat")
            lbl.grid(row=i // liczba_kolumn, column=i % liczba_kolumn, sticky="nsew", padx=2, pady=2)
            self.kanaly_lbl.append(lbl)

        # Resetowanie stanu paska kolejki oraz czyszczenie starej tabeli logów
        self.pb_kolejka["maximum"] = l_kolejka
        for item in self.tabela.get_children(): self.tabela.delete(item)

        # Utworzenie nowej instancji maszyny logicznej SymulatorStacji
        self.symulator = SymulatorStacji(l_kanalow, lambd, N, sigma, min_c, max_c, l_kolejka, czas_sym)
        
        # Przypisanie do GUI pierwszych wygenerowanych przez logikę wartości pseudolosowych
        self.var_poisson_x.set(round(self.symulator.u_poisson, 5))
        self.var_poisson_l.set(round(self.symulator.t_poisson, 5))
        self.var_gauss_x1.set(round(self.symulator.u1_gauss, 5))
        self.var_gauss_x2.set(round(self.symulator.u2_gauss, 5))
        self.var_gauss_x.set(int(self.symulator.x_gauss))

        # Wymuszenie wyczyszczenia obszaru rysowania Matplotlib na czysty biały obszar
        self.dziala = True
        self.ax_q.clear(); self.ax_w.clear(); self.ax_ro.clear()
        self.canvas.draw()
        
        # Asynchroniczny skok do uruchomienia symulacji upływu czasu
        self.petla_symulacji()

    def pauza(self):
        """Flaga False zatrzymuje rekurencyjne wywoływanie pętli czasu."""
        self.dziala = False

    def wznow(self):
        """Wznawia symulację, odrzucając kliknięcia jeśli symulacja dobiegła już końca."""
        if not self.dziala and self.symulator is not None and self.symulator.aktualny_krok < self.symulator.czas_symulacji:
            self.dziala = True
            self.petla_symulacji()

    def petla_symulacji(self):
        """
        Mechanizm silnika czasu rzeczywistego wykorzystujący metodę '.after()'.
        Omija blokowanie głównego wątku aplikacji graficznej.
        """
        if not self.dziala: return
        
        # Wywołaj logikę, która wewnątrz przesuwa zegar o 1 sekundę symulacji do przodu
        trwa, nowi_klienci, k_wartosc = self.symulator.wykonaj_sekunde()
        
        if trwa:
            self.aktualizuj_gui(nowi_klienci, k_wartosc)
            # Parametr opóźnienia: 1000 ms oznacza idealne odwzorowanie 1 sekundy czasu w świecie rzeczywistym
            self.root.after(1000, self.petla_symulacji) 
        else:
            self.dziala = False
            self.lbl_czas.config(text="Koniec symulacji")

    def aktualizuj_gui(self, nowi_klienci, k_wartosc):
        """
        Synchronizuje zmiany, które zaszły na poziomie logiki z interfejsem użytkownika.
        Metoda jest odpowiedzialna za kolorowanie kanałów, przesuwanie pasków i rysowanie wykresów.
        """
        
        # Wypisywanie nowo przyjętych zgłoszeń w Dzienniku Zdarzeń na bieżąco
        for k in nowi_klienci:
            self.tabela.insert('', tk.END, values=(k["id"], k["poisson"], k["odstep_lambda"], k["mi_i"], k["lambdai"], k["mii"], k["roi"]))
            self.tabela.yview_moveto(1) # Autoscroll do najnowszego wpisu

        # Odświeżenie podglądu kanałów (czerwony z odliczaniem = zajęty, zielony = wolny) [cite: 614]
        for i in range(self.symulator.liczba_kanalow):
            if i < len(self.symulator.kanaly):
                self.kanaly_lbl[i].config(bg="#dc3545", text=str(int(self.symulator.kanaly[i]))) 
            else:
                self.kanaly_lbl[i].config(bg="#28a745", text="") 

        # Odświeżenie wizualnego paska określającego zajętość kolejki
        q_len = len(self.symulator.kolejka)
        self.pb_kolejka["value"] = q_len
        self.lbl_kolejka.config(text=f"Kolejka: {q_len} / {self.symulator.max_kolejka}")
        
        # Nadpisywanie tekstowych wskaźników postępu [cite: 615]
        self.lbl_obsluzone.config(text=f"Obsłużone: {self.symulator.obsluzeni}")
        self.lbl_odrzucone.config(text=f"Odrzucone: {self.symulator.odrzuceni}")
        self.lbl_czas.config(text=f"Czas: {self.symulator.aktualny_krok} / {self.symulator.czas_symulacji} s")
        self.lbl_wartosc_k.config(text=f"Przyjęcia w tej sek (k): {k_wartosc}")

        # Moduł renderowania wizualizacji matematycznej
        if self.var_pokaz.get():
            self.ax_q.clear(); self.ax_w.clear(); self.ax_ro.clear()
            
            x_data = list(range(len(self.symulator.historia_Q)))
            
            # W środowisku dyskretnym i modelach kolejkowych zmiany stanów następują skokowo. 
            # Wykorzystujemy tutaj '.step()' z parametrem 'where=post', co jest wymagane dla poprawnego 
            # odwzorowania natury takich zjawisk z punktu widzenia ścisłej matematyki [cite: 617-622].
            self.ax_q.step(x_data, self.symulator.historia_Q, color='#dc3545', label='Kolejka (Q)', where='post')
            self.ax_w.step(x_data, self.symulator.historia_W, color='#007bff', label='Czas oczek. (W)', where='post')
            self.ax_ro.step(x_data, self.symulator.historia_rho, color='#28a745', label='Zajętość (Ro)', where='post')
            
            # Autokalibracja i formatowanie siatek wykresów
            for ax in (self.ax_q, self.ax_w, self.ax_ro):
                ax.set_xlim(left=0, right=max(10, len(self.symulator.historia_Q)))
                ax.set_ylim(bottom=0)
                ax.grid(True, linestyle=':', alpha=0.7)
                ax.legend(loc='upper right', fontsize=8)
                
            # Dynamiczne zapobieganie obcinaniu wykresów po zmianie szerokości ekranu
            self.fig.tight_layout() 
            self.canvas.draw()
            
            # Zastosowanie klasycznej średniej arytmetycznej dla statystyk bieżących [cite: 620-622]
            avg_q = round(np.mean(self.symulator.historia_Q), 4) if self.symulator.historia_Q else 0
            avg_w = round(np.mean(self.symulator.historia_W), 4) if self.symulator.historia_W else 0
            avg_ro = round(np.mean(self.symulator.historia_rho), 4) if self.symulator.historia_rho else 0
            
            self.lbl_avg_q.config(text=f"Średnia Q: {avg_q}")
            self.lbl_avg_w.config(text=f"Średnie W: {avg_w}")
            self.lbl_avg_ro.config(text=f"Średnie Ro: {avg_ro}")

if __name__ == "__main__":
    # Punkt wejścia omijający błędy związane z wywołaniami w tle
    root = tk.Tk()
    app = AplikacjaGUI(root)
    root.mainloop()