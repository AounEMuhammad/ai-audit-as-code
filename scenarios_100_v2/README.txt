# Audit-as-Code: 100 Scenario Pack (v2)

This bundle contains 100 scenarios across multiple domains with **rich evidence**:
- `dataset.json` (with `hash`)
- `model_card.json`, `data_card.json`
- `logs/train.log`
- `audit_trail.json`, `replication.json`
- `local_fidelity.json`, `global_stability.json`, `faithfulness.json`, `robustness.json`, `coverage.json`, `human_comprehensibility.json`
- `shap.json`, `counterfactuals.json`, `saliency.json`
- `fairness.json`, `pii_scan.json`, `redteam.json`
- `risk.json`
- `gate_log.json`

Use the included **policy**: `tracex.policy.v2.yaml` (drop into `policies/`) to require these evidences.

## How to use with your Simulator
1. Open the **Policy Gate Simulator** page.
2. Paste the YAML from `tracex.policy.v2.yaml` into the policy box.
3. Upload a scenario's JSON artifacts (you can ignore `logs/train.log` for the GUI; it's referenced by `audit_trail`).
4. Check the computed TI/XI/ARS/R, PASS/FAIL, and recommendation band.

> In CI, point the CLI to a scenario folder to evaluate all artifacts automatically.
