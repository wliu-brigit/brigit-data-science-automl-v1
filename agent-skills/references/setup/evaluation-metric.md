# Evaluation metric — projects/<project_name>/config.py EVAL

The project evaluation recipe is the objective for AutoML. The default starter
is binary-classification AUC.

## Contract

`config.py` must define an `EVAL` constant:

```python
from automl.eval import EvalSpec
from automl.eval.metrics import Auc

EVAL = EvalSpec(primary=Auc())
```

The task type comes from `TASK`, not `EVAL`:

```python
from automl.project import BinaryClassification

TASK = BinaryClassification(target="TARGET")
```

Use one of:

- `BinaryClassification(target=...)`
- `Regression(target=...)`
- `Multiclass(target=..., n_classes=...)`

Minimal example combining both:

```python
TASK = BinaryClassification(target="TARGET")
EVAL = EvalSpec(primary=Auc())
```

The primary metric is always ranked as higher-is-better. For loss-style metrics,
use unary minus:

```python
from automl.eval import EvalSpec
from automl.eval.metrics import LogLoss

EVAL = EvalSpec(primary=-LogLoss())
```

That resolves to a primary record named `negative_log_loss` with a signed value.

## Built-ins

```python
from automl.eval import EvalSpec
from automl.eval.metrics import Auc, LogLoss, ThresholdSweep

EVAL = EvalSpec(
    primary=Auc(),
    metrics=[
        LogLoss(),
        ThresholdSweep(thresholds=[0.2, 0.5, 0.8]),
    ],
)
```

Every result record has the same shape:

```json
{"name": "auc", "value": 0.7123}
```

Scalar records are eligible for MLflow metric logging. Non-scalar records remain
inside `eval/<label>/report.json`.

## Custom Business Metrics

Create a metric by subclassing `Metric`. Set `required_columns` for evaluation
inputs that must exist in `df_test`; the data pipeline validates those columns
before trials run.

```python
from automl.eval import Metric

class ProjectedRevenue(Metric):
    def __init__(self, *, amount_col: str, cost_col: str):
        self.amount_col = amount_col
        self.cost_col = cost_col
        self.required_columns = (amount_col, cost_col)

    def compute(self, df_test, y_pred, target_col):
        return float((y_pred * df_test[self.amount_col] - df_test[self.cost_col]).mean())
```

Only create `projects/<project_name>/eval/metrics.py` when the project needs
custom metric classes; otherwise `config.py` owns `EVAL`. When a custom metric
class is needed, place it in the project-local metrics module and import it in
`config.py`:

```python
from projects.<project_name>.eval.metrics import ProjectedRevenue

EVAL = EvalSpec(
    primary={"revenue_rank": ProjectedRevenue(amount_col="AMT", cost_col="COST")},
    metrics=[Auc()],
)
```

## Why This Matters

The loop optimizes the primary metric. A weak metric produces a strong-looking
model that solves the wrong problem, so define `EVAL` deliberately before
running AutoML.
