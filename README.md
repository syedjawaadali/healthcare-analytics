# Healthcare / Clinical-Trial Analytics

Analyzes a synthetic two-arm clinical trial for **efficacy** (change from
baseline) and **safety** (adverse-event rates), and runs a Welch's t-test to
assess whether the treatment effect is statistically significant — reflecting
the healthcare & clinical-research analytics I've delivered.

## Run it
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/analyze.py
```

## Stack
`pandas` · `scipy.stats` (hypothesis testing)

---
Part of my analytics portfolio — [github.com/syedjawaadali](https://github.com/syedjawaadali)
