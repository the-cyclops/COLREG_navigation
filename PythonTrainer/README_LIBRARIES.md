# Documentazione delle Librerie Python (Progetto COLREG-XAI)

Questo documento fornisce una descrizione tecnica delle librerie incluse nell'ambiente virtuale Conda `colreg_xai`. La selezione dei pacchetti è finalizzata a garantire la stabilità su architetture **Apple Silicon M4** e **Windows NVIDIA**, supportando l'integrazione tra logica formale (XAI) e apprendimento per rinforzo vincolato (Robotica).

**Versione Python**: `3.10.12` (Scelta per garantire la compatibilità con ML-Agents 1.1.0).

---

### 1. Librerie per la Logica Formale e XAI

#### **rtamt (Runtime Analysis and Monitoring Tool)**
`rtamt` è il motore logico principale del progetto per il calcolo della robustezza rispetto alle specifiche STL.
* **Monitoraggio COLREG**: Traduce le regole marittime definite in YAML in monitor che analizzano le traiettorie della nave in tempo reale.
* **Generazione dei Costi**: Fornisce il valore di robustezza ($\rho$) che, se negativo, indica una violazione. Questo valore viene trasformato in un segnale di costo per l'agente.
* **Spiegabilità (XAI)**: Permette di quantificare matematicamente il rispetto o la violazione di una specifica regola, rendendo interpretabili le decisioni dell'agente.

---

### 2. Infrastruttura di Comunicazione e Simulazione

#### **mlagents (v1.1.0)**
* Interfaccia di comunicazione tra l'ambiente di simulazione navale in Unity e lo script di controllo Python.
* La versione è bloccata alla 1.1.0 per coerenza con la Release 23 di Unity.

#### **protobuf (v3.20.3)**
* Gestisce la serializzazione dei dati. La versione specifica risolve i problemi di compatibilità noti sulle architetture ARM64 (M4) e garantisce l'allineamento dei messaggi binari con il simulatore.

#### **grpcio**
* Protocollo di trasporto per ML-Agents. Installato tramite il canale Conda per evitare errori di compilazione locale su macOS.

---