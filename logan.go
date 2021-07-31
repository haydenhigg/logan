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

func (mdl *Model) TrainBatch(inputs [][]float64, outputs []float64, iters int) {
	m := float64(len(inputs))

	if m == 0. {
		return
	}

	n := len(inputs[0])

	if len(mdl.Weights) == 0 {
		mdl.Weights = make([]float64, n)
	}

	batchEta := mdl.eta / m

	gradient := make([]float64, n)
	biasGradient := 0.

	var delta float64

	for iter := 0; iter < iters; iter++ {
		for i, input := range inputs {
			delta = mdl.Predict(input) - outputs[i]

			for j, feature := range input {
				gradient[j] -= delta * feature
			}

			biasGradient -= delta
		}

		for j := range gradient {
			mdl.Weights[j] += batchEta * gradient[j]
			gradient[j] = 0
		}

		mdl.Bias += batchEta * biasGradient
		biasGradient = 0
	}
}

func (mdl *Model) TrainSGD(inputs [][]float64, outputs []float64, epochs int) {
	m := len(inputs)

	if m == 0 {
		return
	}

	if len(mdl.Weights) == 0 {
		mdl.Weights = make([]float64, len(inputs[0]))
	}

	var index int
	var delta float64

	for epoch := 0; epoch < epochs; epoch++ {
		index = mdl.rand.Intn(m)
		delta = mdl.eta * (mdl.Predict(inputs[index]) - outputs[index])

		for j, feature := range inputs[index] {
			mdl.Weights[j] -= delta * feature
		}

		mdl.Bias -= delta
	}
}

func (mdl *Model) Predict(input []float64) float64 {
	return sigmoid(dot(input, mdl.Weights) + mdl.Bias)
}
