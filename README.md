# Progetto di Navigazione Autonoma COLREG con XAI

Questo repository contiene l'integrazione tra un ambiente di simulazione Unity e un sistema di controllo basato su Logica Temporale Segnale (STL) sviluppato in Python.

## Configurazione del Sistema

Per garantire la corretta sincronizzazione dei file e la riproducibilità degli esperimenti tra diverse piattaforme (macOS M4 e Windows), è necessario seguire i passaggi di configurazione riportati di seguito.

### 1. Gestione dei file grandi (Git LFS)

Il progetto utilizza Git LFS (Large File Storage) per la gestione di modelli neurali, asset 3D e file binari pesanti. Senza questa estensione, i file binari scaricati risulteranno non validi.

**Installazione**:
   - Su macOS: `brew install git-lfs`
   - Su Windows: Scaricare l'eseguibile da git-lfs.com

Eseguire il comando globale una sola volta sul proprio sistema:
```git lfs install```

Dopo di che eseguire ```git lfs pull``` all'interno della cartella COLREG_NAVIGATION

### 2. Ambiente conda

Lanciare ```conda env create -f PythonTrainer/environment.yml``` per l'ambiente. 
Attivare l'ambiente con ```conda activate colreg_xai```
Lanciare il seguente comando per vedere se mps o cuda sono in fuzione ```python -c "import torch; print('GPU Disponibile:', torch.backends.mps.is_available() or torch.cuda.is_available())"``` 