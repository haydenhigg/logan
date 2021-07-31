package logan

import "math"

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
