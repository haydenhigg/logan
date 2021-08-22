# logan
Performant logistic regression in Go.

## interface
- `func New(learningRate float64) *Model`
- `func NewL2Regularized(learningRate, l2Parameter float64) *Model`
- Model
  - `Weights []float64`
  - `Bias float64`
  - `func (mdl *Model) TrainBatch(inputs [][]float64, outputs []float64, epochs int)`
  - `func (mdl *Model) TrainMiniBatch(inputs [][]float64, outputs []float64, epochs, batchSize int)`
  - `func (mdl *Model) TrainSGD(inputs [][]float64, outputs []float64, epochs int)`
  - `func (mdl *Model) Train(input []float64, output float64)`
  - `func (mdl *Model) Predict(input []float64) float64`
- `func Marshal(mdl *Model) ([]byte, error)`
- `func Unmarshal(data []byte) (*Model, error)`

## notes
- All output values used as training parameters should be 0 or 1.
- `Predict` returns the probability, **not** 0 or 1.
- Input standardization is **always** performed.
- `l2Parameter` should be between 0 (no regularization) and 1 (full regularization).