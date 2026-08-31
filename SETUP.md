# Running the VLF (Vision-Based Line Following) Project — macOS & Windows

This guide gets the whole `vision-based-line-following` folder running on a new machine (macOS or Windows) from a clean install, no guesswork.

**Project stack:** Webots (robot simulator) + Python 3.12 + TensorFlow/Keras + OpenCV.

---

## 1. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| **Webots** | R2025a (recommended) | Download from [cyberbotics.com/download](https://cyberbotics.com/download) |
| **Python** | 3.12 (64-bit) | [python.org/downloads](https://www.python.org/downloads/) — on Windows, tick "Add python.exe to PATH" during install |

> The trained CNN (`models_trained/line_error_cnn.keras`) uses TensorFlow 2.21 with `tf.keras` (see `requirements.txt`). Stick to Python 3.12 for the tested project environment and compatible TensorFlow setup.

**macOS note (Apple Silicon):** TensorFlow 2.21 provides native macOS ARM64 wheels for supported Python versions, including Python 3.12. Install the standard dependencies with `pip install -r requirements.txt`. GPU acceleration through Apple's Metal backend is optional and can be added separately with `tensorflow-metal`.

---

## 2. Unzip the project

Extract `vision-based-line-following.zip` anywhere with **no special characters or excessively long paths** — e.g.:

- macOS: `~/Projects/vision-based-line-following`
- Windows: `C:\Projects\vision-based-line-following`

Avoid OneDrive/iCloud-synced folders if possible — live syncing while Webots writes log/cache files can cause file-lock errors.

---

## 3. Create the Python virtual environment

Open a terminal **inside the extracted project folder** (the one containing `requirements.txt`).

### macOS / Linux (bash/zsh)
```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
> If PowerShell blocks the activation script with an execution-policy error, run once as admin:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Verify the environment
With the venv still active:
```bash
python scripts/check_environment.py
```
This confirms every required package is installed **and** that the trained model, dataset manifest, and base world file are all present. Fix anything it flags before moving on.

---

## 4. Point Webots at this Python environment

Webots runs the controller scripts with whatever Python it's configured to use — this must be the venv you just created, not your system Python.

1. Open **Webots**.
2. Go to **Tools → Preferences → General → Python command**.
3. Set it to the full path of the venv's Python interpreter:
   - macOS: `/absolute/path/to/vision-based-line-following/.venv312/bin/python`
   - Windows: `C:\absolute\path\to\vision-based-line-following\.venv312\Scripts\python.exe`
4. Restart Webots so the setting takes effect.

---

## 5. Open and run the simulation

1. In Webots: **File → Open World...**
2. Open the base test world:
   ```
   worlds/line_following_curve_test.wbt
   ```
3. Press the **Play** (▶) button in the Webots toolbar.

The robot (e-puck) will start line-following using whichever controller is attached in the world file.

### The three final controllers
| Controller | What it does |
|---|---|
| `controllers/p_line_follower` | Classical vision + P-only steering |
| `controllers/pd_line_follower` | Classical vision + PD steering |
| `controllers/cnn_pd_hybrid_final` | CNN estimates line error → PD steers, using `models_trained/line_error_cnn.keras` |

To switch controllers, select the robot node in the Webots scene tree, open its `controller` field, and pick the desired controller name — or open one of the many pre-built experiment worlds in `worlds/` (e.g. `exp_baseline__pd_line_follower_offp000.wbt`, `exp_robust_c12000__cnn_pd_hybrid_final.wbt`) which already have a specific controller + track offset/curvature wired up.

---

## 6. Re-running analysis / training scripts (optional)

All standalone scripts live in `scripts/` and run with the venv Python (no Webots needed for these):

```bash
# with .venv312 activated
python scripts/analyze_results.py
python scripts/analyze_robustness.py
python scripts/train_line_error_cnn.py      # retrains the CNN from data/cnn_dataset
python scripts/make_dissertation_outputs.py
```

Check each script's top-of-file docstring for expected inputs — most read from `data/cnn_dataset/`, `evaluation_results_final/`, or `models_trained/`, all of which are already included in the zip.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Webots says controller can't start / `ModuleNotFoundError` | Python command in Webots preferences isn't pointing at `.venv312`. Redo step 4. |
| `pip install` fails on `tensorflow==2.21.0` (macOS ARM) | Use `tensorflow-macos` + `tensorflow-metal` as noted in step 1. |
| Model file missing (`line_error_cnn.keras` not found) | Confirm `models_trained/line_error_cnn.keras` was included when you unzipped — `check_environment.py` will flag this explicitly. |
| PowerShell won't run `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, as noted in step 3. |
| Simulation runs but robot doesn't move / camera errors | Confirm you opened `worlds/line_following_curve_test.wbt` (or another world in `worlds/`) rather than a blank Webots project. |

---

## 8. Quick reference — full command sequence

```bash
# macOS
cd "path/to/vision-based-line-following"
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
# then set Webots Python command → .venv312/bin/python, open worlds/line_following_curve_test.wbt, press Play
```

```powershell
# Windows
cd "path\to\vision-based-line-following"
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
# then set Webots Python command → .venv312\Scripts\python.exe, open worlds\line_following_curve_test.wbt, press Play
```
