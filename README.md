# Tree Predictors for Binary Classification

This project implements decision trees and random forests from scratch in Python to classify mushroom edibility using the UCI Mushroom dataset.

## Files

- `tree_predictor_mushroom.py`: Python script containing the full implementation and experiments.
- `tree_predictor_report.pdf`: Final project report.
- `README.md`: Project description and usage instructions.

## Installation

Install the required Python libraries using `pip`:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib graphviz ucimlrepo
```

Then install the Graphviz system binary (required for tree visualization):

- **macOS (Homebrew):**

  ```bash
  brew install graphviz
  ```

- **Ubuntu/Debian:**

  ```bash
  sudo apt-get install graphviz
  ```

- **Windows:**

  Download and install from [https://graphviz.org/download](https://graphviz.org/download),
  and add the `bin/` folder (e.g., `C:\Program Files\Graphviz\bin`) to your system’s `PATH`.
