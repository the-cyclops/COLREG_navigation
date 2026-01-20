# Progetto di Navigazione Autonoma COLREG con XAI

Questo repository contiene l'integrazione tra un ambiente di simulazione navale in Unity e un sistema di controllo basato su Logica Temporale (STL) sviluppato in Python.

## Algoritmi di Controllo e Apprendimento

Il progetto implementa un approccio di **Constraint-Controlled RL** basato su obiettivi Lagrangiani, che si distingue dal classico reward shaping.

* **Priorità Lessicografica**: L'agente alterna tra l'ottimizzazione dell'obiettivo (missione) e il recupero della sicurezza. Se i vincoli STL sono soddisfatti ($\rho \ge 0$), l'agente massimizza il reward; se violati ($\rho < 0$), entra in *Constraint Recovery*.
* **CAGRAD (Conflict-Averse Gradient Descent)**: Algoritmo multi-obiettivo che risolve i conflitti tra gradienti quando più regole COLREG sono violate simultaneamente, garantendo che la correzione di una regola non ne danneggi un'altra.
* **Differenza dallo Shaping**: Invece di sommare reward e costi, questo metodo mantiene i gradienti separati, evitando che reward elevati "nasccondano" violazioni di sicurezza critiche.

## Configurazione del Sistema

### 1. Gestione dei file grandi (Git LFS)
Il progetto utilizza Git LFS per gestire modelli neurali e asset 3D pesanti.
* **Installazione**: `brew install git-lfs` (Mac) o scaricare da git-lfs.com (Windows).
* **Setup**: Eseguire `git lfs install` e poi `git lfs pull` nella cartella del progetto.

### 2. Ambiente Conda
Creare l'ambiente:
```conda env create -f PythonTrainer/environment.yml```
```conda activate colreg_xai```