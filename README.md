# Offensive-Language-Detector-Threads
Progetto per l'esame di Fondamenti di Data Science e Machine Learning. È stato sviluppato un modello per la classificazione automatica del linguaggio offensivo su Threads e Tiktok.

La documentazione completa in formato PDF, contenente informazioni dettagliate sui dataset utilizzati nel progetto, è disponibile su GitHub al seguente link (https://github.com/grauso-t/Offensive-Language-Detector/blob/main//doc/Documentazione.pdf).

## Setup del progetto

Per eseguire il progetto è necessario creare un ambiente virtuale Python e installare le dipendenze specifiche in base alla configurazione hardware disponibile (con o senza GPU).

### 1. Creazione di un ambiente virtuale

Apri il terminale nella directory del progetto e crea l’ambiente virtuale:

```bash
python -m venv venv
```

Attiva l’ambiente virtuale:
```bash
(Windows) venv\Scripts\activate
(Mac/Linux) source venv/bin/activate
```

### 2. Installa le dipendenze
```bash
(se hai una GPU compatibile con CUDA) pip install -r requirements_gpu.txt
(uso solo CPU) pip install -r requirements_cpu.txt

```

### 3. Addestramento dei modelli

I modelli per la classificazione del linguaggio offensivo sono organizzati all'interno della cartella `models/`. Ogni modello ha una propria sottocartella dedicata, contenente uno script `train.py` per l'addestramento.

Assicurati che l'ambiente virtuale sia attivo e che tutte le dipendenze siano installate, poi esegui i seguenti comandi da terminale:

- **Logistic Regression**:
  ```bash
  models/logistic_regression/train.py
   ```

- **Linear SVM**:
  ```bash
  models/linear_svm/train.py
   ```

- **BERT**:
  ```bash
  models/bert_trainer/train.py
   ```

### 4. Esecuzione della classificazione

Per utilizzare il modello di classificazione e analizzare un testo offensivo tramite un link Threads (solo testo) o Tiktok (audio, speech-to-text), puoi eseguire lo script principale presente in `backend/script.py`.

Assicurati che l’ambiente virtuale sia attivo e che tutte le dipendenze siano installate, poi esegui il comando:

```bash
python backend/script.py
```

## Problematiche note

- Per scaricare i link da TikTok è necessario essere **obbligatoriamente loggati** in un browser.  
- Si consiglia di utilizzare **preferibilmente Firefox** per garantire una maggiore compatibilità e ridurre problemi di autenticazione durante il processo di scraping o download.  
- Senza il login attivo, il sistema non sarà in grado di accedere ai contenuti protetti e quindi non potrà scaricare i dati correttamente.
