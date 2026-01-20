# Documentazione delle Librerie Python (Progetto COLREG-XAI)

Questo documento fornisce una descrizione tecnica delle librerie incluse nell'ambiente virtuale Conda. La selezione dei pacchetti è finalizzata a garantire la compatibilità tra diverse architetture hardware (macOS Apple Silicon e Windows NVIDIA CUDA) e a supportare l'integrazione tra logica formale e apprendimento per rinforzo.

Versione 3.10.12 di Python per garantire compatibilità con le librerie.
---

### 1. Librerie per la Logica Formale e XAI

#### stlcg (Signal Temporal Logic Computation Graph)
La libreria `stlcg` rappresenta il componente principale per l'implementazione della Explainable AI (XAI) nel progetto. 
* Consente la traduzione di specifiche espresse in Signal Temporal Logic (STL) in grafi computazionali differenziabili.
* Viene utilizzata per calcolare il grado di robustezza ($\rho$) rispetto alle regole COLREG, trasformando vincoli logici in funzioni di costo utilizzabili per l'addestramento della rete neurale.

#### rtamt (Runtime Analysis and Monitoring Tool)
`rtamt` è uno strumento dedicato al monitoraggio a tempo di esecuzione delle proprietà temporali.
* Viene impiegato per la verifica formale del comportamento dell'agente durante le fasi di test.
* Fornisce un'analisi quantitativa della conformità alle regole marittime in tempo reale, segnalando eventuali violazioni dei margini di sicurezza stabiliti.

---

### 2. Infrastruttura di Comunicazione e Simulazione

#### mlagents-envs
Il pacchetto `mlagents-envs` costituisce l'interfaccia di comunicazione tra l'ambiente di simulazione Unity e lo script di controllo Python.
* Gestisce lo scambio di dati relativi allo stato dei sensori e alle azioni dei motori tramite una connessione socket.
* È strettamente vincolato alla versione 1.1.0 per garantire l'allineamento con la Release 23 di Unity ML-Agents.

#### protobuf
La libreria `protobuf` (Protocol Buffers) definisce il formato di serializzazione dei dati utilizzato per la comunicazione tra C# e Python.
* La versione è bloccata alla 3.20.3 per risolvere i problemi di compatibilità noti sulle architetture ARM64 (Apple Silicon) e per mantenere la coerenza dei messaggi binari con il simulatore.

---

### 3. Gestione dei Dati e Calcolo Scientifico

#### numpy
`numpy` è la libreria fondamentale per il calcolo numerico.
* La versione è limitata a pacchetti inferiori alla 2.0.0 per mantenere la retrocompatibilità con ML-Agents 1.1.0, che non supporta le modifiche strutturali introdotte nelle versioni più recenti.

#### pytorch
`pytorch` è il framework di deep learning utilizzato per l'implementazione delle reti neurali.
* La configurazione dell'ambiente permette l'utilizzo dell'accelerazione hardware MPS (Metal Performance Shaders) su sistemi macOS M4 e l'accelerazione CUDA su sistemi Windows dotati di GPU NVIDIA.

---
*Fine della documentazione tecnica.*