package logan

func (mdl *Model) TrainBatch(inputs [][]float64, outputs []float64, epochs int) {
	mdl.calculateMeans(inputs)

	n := len(inputs[0])

	if len(mdl.Weights) == 0 {
		mdl.Weights = make([]float64, n)
	}

	eta := mdl.LearningRate / float64(len(inputs))
	gradient := Gradient { weights: make([]float64, n) }

	for epoch := 0; epoch < epochs; epoch++ {
		for i, input := range inputs {
			mdl.update(&gradient, input, outputs[i])
		}

		mdl.descend(&gradient, eta)
	}
}

func (mdl *Model) TrainMiniBatch(inputs [][]float64, outputs []float64, epochs, batchSize int) {
	mdl.calculateMeans(inputs)
	
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

	eta := mdl.LearningRate / float64(batchSize)
	gradient := Gradient { weights: make([]float64, n) }

	var index int

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < batchSize; i++ {
			index = mdl.rand.Intn(m)
			mdl.update(&gradient, inputs[index], outputs[index])
		}

		mdl.descend(&gradient, eta)
	}
}

func (mdl *Model) TrainSGD(inputs [][]float64, outputs []float64, epochs int) {
	mdl.calculateMeans(inputs)
	
	m := len(inputs)

	if len(mdl.Weights) == 0 {
		mdl.Weights = make([]float64, len(inputs[0]))
	}

	var index int

	for epoch := 0; epoch < epochs; epoch++ {
		index = mdl.rand.Intn(m)
		mdl.train(inputs[index], outputs[index])
	}
}

func (mdl *Model) train(input []float64, output float64) {
	scaled := subtract(input, mdl.Means)
	delta := mdl.predict(scaled) - output

	for j, feature := range scaled {
		mdl.Weights[j] -= delta * feature + mdl.L2Parameter * mdl.Weights[j]
	}

	mdl.Bias -= delta
}

func (mdl *Model) predict(input []float64) float64 {
	return sigmoid(dot(input, mdl.Weights) + mdl.Bias)
}

func (mdl *Model) Predict(input []float64) float64 {
	return mdl.predict(subtract(input, mdl.Means))
}
