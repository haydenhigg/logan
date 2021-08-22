package logan

import "math"

func subtract(a, b []float64) []float64 {
	difference := make([]float64, len(a))

	for i, x := range a {
		difference[i] = x - b[i]
	}

	return difference
}

func dot(a, b []float64) float64 {
	s := 0.

	for i, x := range a {
		s += x * b[i]
	}

	return s
}

func sigmoid(z float64) float64 {
	return 1. / (1. + math.Exp(-z))
}

func (mdl *Model) calculateMeans(inputs [][]float64) {
	mdl.Means = make([]float64, len(inputs[0]))

	for i := range inputs {
		for j, feature := range inputs[i] {
			mdl.Means[j] += feature
		}
	}

	m := float64(len(inputs))

	for j := range mdl.Means {
		mdl.Means[j] /= m
	}
}

func (mdl *Model) update(gradient *Gradient, input []float64, output float64) {
	scaled := subtract(input, mdl.Means)
	delta := mdl.predict(scaled) - output

	for j, feature := range scaled {
		gradient.weights[j] -= delta * feature
	}

	gradient.bias -= delta
}

func (mdl *Model) descend(gradient *Gradient, coefficient float64) {
	for j, partial := range gradient.weights {
		mdl.Weights[j] += coefficient * partial - mdl.L2Parameter * mdl.Weights[j]
		gradient.weights[j] = 0
	}

	mdl.Bias += coefficient * gradient.bias
	gradient.bias = 0
}
