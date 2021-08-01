package logan

import (
	"math/rand"
	"time"
)

type Model struct {
	Weights []float64
	Bias	float64
	eta		float64
	rand	*rand.Rand
}

func New(learningRate float64) *Model {
	return &Model {
		Weights: []float64 {},
		Bias: 0,
		eta: learningRate,
		rand: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (mdl *Model) TrainBatch(inputs [][]float64, outputs []float64, epochs int) {
	n := len(inputs[0])

	if len(mdl.Weights) == 0 {
		mdl.Weights = make([]float64, n)
	}

	batchEta := mdl.eta / float64(len(inputs))
	gradient := struct { weights []float64; bias float64 } {
		weights: make([]float64, n),
		bias: 0,
	}

	var delta float64

	for epoch := 0; epoch < epochs; epoch++ {
		for i, input := range inputs {
			delta = mdl.Predict(input) - outputs[i]

			for j, feature := range input {
				gradient.weights[j] -= delta * feature
			}

			gradient.bias -= delta
		}

		for j, partial := range gradient.weights {
			mdl.Weights[j] += batchEta * partial
			gradient.weights[j] = 0
		}

		mdl.Bias += batchEta * gradient.bias
		gradient.bias = 0
	}
}

func (mdl *Model) TrainMiniBatch(inputs [][]float64, outputs []float64, epochs, batchSize int) {
	m := len(inputs)

	if batchSize >= m {
		mdl.TrainBatch(inputs, outputs, epochs)
		return
	} else if batchSize <= 1 {
		mdl.TrainSGD(inputs, outputs, epochs)
		return
	}

	n := len(inputs[0])

	if len(mdl.Weights) == 0 {
		mdl.Weights = make([]float64, n)
	}

	batchEta := mdl.eta / float64(batchSize)
	gradient := struct { weights []float64; bias float64 } {
		weights: make([]float64, n),
		bias: 0,
	}

	var index int
	var delta float64

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < batchSize; i++ {
			index = mdl.rand.Intn(m)
			delta = mdl.Predict(inputs[index]) - outputs[index]

			for j, feature := range inputs[index] {
				gradient.weights[j] -= delta * feature
			}

			gradient.bias -= delta
		}

		for j, partial := range gradient.weights {
			mdl.Weights[j] += batchEta * partial
			gradient.weights[j] = 0
		}

		mdl.Bias += batchEta * gradient.bias
		gradient.bias = 0
	}
}

func (mdl *Model) TrainSGD(inputs [][]float64, outputs []float64, epochs int) {
	m := len(inputs)

	if len(mdl.Weights) == 0 {
		mdl.Weights = make([]float64, len(inputs[0]))
	}

	var index int

	for epoch := 0; epoch < epochs; epoch++ {
		index = mdl.rand.Intn(m)
		mdl.Train(inputs[index], outputs[index])
	}
}

func (mdl *Model) Train(input []float64, output float64) {
	delta := mdl.eta * (mdl.Predict(input) - output)

	for j, feature := range input {
		mdl.Weights[j] -= delta * feature
	}

	mdl.Bias -= delta
}

func (mdl *Model) Predict(input []float64) float64 {
	return sigmoid(dot(input, mdl.Weights) + mdl.Bias)
}
