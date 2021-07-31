# logan
Performant logistic regression in Go.

## interface
- `func New(learningRate float64) *Model`
- Model
  - `Weights []float64`
  - `Bias float64`
  - `func TrainBatch(inputs [][]float64, outputs []float64, epochs int)`
  - `func TrainMiniBatch(inputs [][]float64, outputs []float64, epochs, batchSize int)`
  - `func TrainSGD(inputs [][]float64, outputs []float64, epochs int)`
  - `func Predict(input []float64) float64`

## notes
- `outputs` is `[]float64`, but each value should be 0 or 1
- `Predict([]float64) float64` returns the raw probability, **not** 0 or 1